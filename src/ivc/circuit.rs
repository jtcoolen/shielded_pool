//! Generic IVC circuits: leaf, node, and decider.
//!
//! Each circuit wraps a user-provided step function with the framework's
//! recursive partial verification and accumulator folding.  The user never
//! constructs these directly — [`super::engine::IvcProver`] does.

use std::collections::BTreeMap;

use midnight_circuits::instructions::{AssignmentInstructions, PublicInputInstructions};
use midnight_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{Circuit, ConstraintSystem, Error},
};

use super::{
    Acc, AssignedNative, C, DECIDER_CHILD_ARITY, DeciderStep, F, FoldStep, LEAF_CLIENT_ARITY,
    LeafStep, NODE_CHILD_ARITY, VkData,
    ctx::{
        AggCircuitConfig, IvcCtx, RpvInput, configure_ivc_circuit, expose_node_outputs,
        recursive_partial_verify,
    },
};

////////////////////////////////////////////////////////////////////////////////
// Framework witness — fields managed by the framework, not the application
////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Debug)]
pub struct FrameworkWitness {
    pub child_vk: VkData,
    pub child_vk_name: String,
    pub child_proofs: Vec<Value<Vec<u8>>>,
    pub child_pi_accs: Vec<Value<Acc>>,
    pub fixed_base_names: Vec<String>,
}

impl FrameworkWitness {
    pub fn without_witnesses(&self) -> Self {
        Self {
            child_vk: self.child_vk.clone(),
            child_vk_name: self.child_vk_name.clone(),
            child_proofs: self.child_proofs.iter().map(|_| Value::unknown()).collect(),
            child_pi_accs: self
                .child_pi_accs
                .iter()
                .map(|_| Value::unknown())
                .collect(),
            fixed_base_names: self.fixed_base_names.clone(),
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Shared synthesis core
//
// Every IVC circuit follows the same three-phase pattern:
//   1. Run the application's step function  → (state, child_pis)
//   2. Framework appends Merkle digest to the state
//   3. Recursive partial verify + expose outputs
////////////////////////////////////////////////////////////////////////////////

/// Output produced by the step-phase closure inside `synthesize_node`.
struct StepPhaseOutput {
    full_state: Vec<AssignedNative<F>>,
    child_pis: Vec<Vec<AssignedNative<F>>>,
}

fn synthesize_node<const K: u32, L: Layouter<F>>(
    config: AggCircuitConfig,
    layouter: &mut L,
    fw: &FrameworkWitness,
    children_are_client_proofs: bool,
    fixed_bases: Option<&BTreeMap<String, C>>,
    step_phase: impl FnOnce(&IvcCtx, &mut L) -> Result<StepPhaseOutput, Error>,
) -> Result<(), Error> {
    let ctx = IvcCtx::new(&config, (K as usize).saturating_sub(1));
    let assigned_vk = ctx.verifier.assign_fixed_vk(
        layouter,
        &fw.child_vk_name,
        &fw.child_vk.domain,
        &fw.child_vk.cs,
        fw.child_vk.transcript_repr,
    )?;

    let out = step_phase(&ctx, layouter)?;

    let rpv = recursive_partial_verify(
        &ctx,
        layouter,
        RpvInput {
            assigned_vk: &assigned_vk,
            children_are_client_proofs,
            fixed_base_names: &fw.fixed_base_names,
            child_base_pis: out.child_pis,
            child_proofs: fw.child_proofs.clone(),
            child_pi_accs: fw.child_pi_accs.clone(),
        },
    )?;

    let mut next_acc = rpv.next_acc;
    if let Some(fixed_bases) = fixed_bases {
        let assigned_fixed_bases: BTreeMap<_, _> = fixed_bases
            .iter()
            .map(|(name, base)| {
                let assigned = ctx.curve.assign_fixed(layouter, *base)?;
                Ok((name.clone(), assigned))
            })
            .collect::<Result<_, Error>>()?;
        next_acc.resolve_fixed_bases(&assigned_fixed_bases);
        next_acc.collapse(layouter, &ctx.curve, &ctx.scalar)?;
    }

    expose_node_outputs(&ctx, layouter, out.full_state, &next_acc)?;
    ctx.load(layouter)
}

////////////////////////////////////////////////////////////////////////////////
// IvcLeafCircuit<L, K>
//
// Subcircuit at the leaves of the binary tree.  Verifies six raw client
// proofs, runs the application's LeafStep, and produces the base Merkle
// digest H(H(H(h(x0), h(x1)), H(h(x2), h(x3))), H(h(x4), h(x5))).
////////////////////////////////////////////////////////////////////////////////

#[derive(Clone)]
pub struct IvcLeafCircuit<L: LeafStep, const K: u32> {
    pub(crate) step: L,
    pub(crate) client_items: [Value<Vec<F>>; LEAF_CLIENT_ARITY],
    pub(crate) witness: Value<L::Witness>,
    pub(crate) client_pi_width: usize,
    pub(crate) fixed_bases: BTreeMap<String, C>,
    pub(crate) fw: FrameworkWitness,
}

impl<L: LeafStep, const K: u32> Circuit<F> for IvcLeafCircuit<L, K> {
    type Config = AggCircuitConfig;
    type FloorPlanner = SimpleFloorPlanner;
    type Params = ();

    fn without_witnesses(&self) -> Self {
        Self {
            step: self.step.clone(),
            client_items: std::array::from_fn(|_| Value::unknown()),
            witness: Value::unknown(),
            client_pi_width: self.client_pi_width,
            fixed_bases: self.fixed_bases.clone(),
            fw: self.fw.without_witnesses(),
        }
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        configure_ivc_circuit(meta)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        let step = self.step.clone();
        let items = self.client_items.clone();
        let witness = self.witness.clone();
        let width = self.client_pi_width;

        synthesize_node::<K, _>(
            config,
            &mut layouter,
            &self.fw,
            true, // children ARE client proofs
            Some(&self.fixed_bases),
            |ctx, layouter| {
                // 1. Assign client PIs
                let client_pis = items
                    .iter()
                    .map(|item| assign_value_vec(ctx, layouter, item, width))
                    .collect::<Result<Vec<_>, _>>()?;

                // 2. Application step (only sees client PIs)
                let app_state = step.synthesize(ctx, layouter, &client_pis, witness)?;

                // 3. Framework: Merkle hashes
                if client_pis.is_empty() {
                    return Err(Error::Synthesis("leaf arity must be > 0".to_string()));
                }
                let mut layer = client_pis
                    .iter()
                    .map(|pi| ctx.hash_many(layouter, pi))
                    .collect::<Result<Vec<_>, _>>()?;
                while layer.len() > 1 {
                    let mut next = Vec::with_capacity((layer.len() + 1) / 2);
                    let mut i = 0usize;
                    while i + 1 < layer.len() {
                        next.push(ctx.hash2(layouter, &layer[i], &layer[i + 1])?);
                        i += 2;
                    }
                    if i < layer.len() {
                        next.push(layer[i].clone());
                    }
                    layer = next;
                }
                let digest = layer
                    .pop()
                    .ok_or_else(|| Error::Synthesis("missing leaf digest".to_string()))?;

                // 4. Full state = [app_state..., digest]
                let mut full = app_state;
                full.push(digest);

                Ok(StepPhaseOutput {
                    full_state: full,
                    child_pis: client_pis,
                })
            },
        )
    }
}

////////////////////////////////////////////////////////////////////////////////
// IvcNodeCircuit<Fo, K>
//
// Subcircuit at internal nodes.  Verifies four child aggregation proofs,
// runs the application's FoldStep on the app-state portions, and computes
// the parent Merkle digest H(H(d0, d1), H(d2, d3)).
////////////////////////////////////////////////////////////////////////////////

#[derive(Clone)]
pub struct IvcNodeCircuit<Fo: FoldStep, const K: u32> {
    pub(crate) step: Fo,
    pub(crate) app_state_width: usize,
    pub(crate) child_states: Vec<Vec<Value<F>>>,
    pub(crate) fw: FrameworkWitness,
}

impl<Fo: FoldStep, const K: u32> Circuit<F> for IvcNodeCircuit<Fo, K> {
    type Config = AggCircuitConfig;
    type FloorPlanner = SimpleFloorPlanner;
    type Params = ();

    fn without_witnesses(&self) -> Self {
        let full_width = self.app_state_width + 1;
        Self {
            step: self.step.clone(),
            app_state_width: self.app_state_width,
            child_states: vec![vec![Value::unknown(); full_width]; NODE_CHILD_ARITY],
            fw: self.fw.without_witnesses(),
        }
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        configure_ivc_circuit(meta)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        let step = self.step.clone();
        let child_vals = self.child_states.clone();
        let w = self.app_state_width;

        synthesize_node::<K, _>(
            config,
            &mut layouter,
            &self.fw,
            false, // children are NOT client proofs
            None,
            |ctx, layouter| {
                // 1. Assign full child states [app..., digest]
                if child_vals.len() != NODE_CHILD_ARITY {
                    return Err(Error::Synthesis("node arity mismatch".to_string()));
                }
                let child_full_states = child_vals
                    .iter()
                    .map(|vals| assign_values(ctx, layouter, vals))
                    .collect::<Result<Vec<_>, _>>()?;

                // 2. Application fold over all child app states
                let mut app_state = child_full_states[0][..w].to_vec();
                for child in child_full_states.iter().skip(1) {
                    app_state = step.synthesize(ctx, layouter, &app_state, &child[..w])?;
                }

                // 3. Framework: parent Merkle digest
                if !child_full_states.len().is_power_of_two() {
                    return Err(Error::Synthesis(
                        "node arity must be a power of two".to_string(),
                    ));
                }
                let mut layer: Vec<AssignedNative<F>> = child_full_states
                    .iter()
                    .map(|full| full[w].clone())
                    .collect();
                while layer.len() > 1 {
                    let mut next = Vec::with_capacity(layer.len() / 2);
                    for pair in layer.chunks_exact(2) {
                        next.push(ctx.hash2(layouter, &pair[0], &pair[1])?);
                    }
                    layer = next;
                }
                let digest = layer
                    .pop()
                    .ok_or_else(|| Error::Synthesis("missing node digest".to_string()))?;

                let mut full = app_state;
                full.push(digest);

                Ok(StepPhaseOutput {
                    full_state: full,
                    child_pis: child_full_states,
                })
            },
        )
    }
}

////////////////////////////////////////////////////////////////////////////////
// IvcDeciderCircuit<D, K>
//
// Final "wrap" circuit.  Verifies the two top-level aggregation proofs,
// runs the DeciderStep which produces application-specific final PIs,
// and exposes the final accumulator.
////////////////////////////////////////////////////////////////////////////////

#[derive(Clone)]
pub struct IvcDeciderCircuit<D: DeciderStep, const K: u32> {
    pub step: D,
    pub app_state_width: usize,
    pub child_states: Vec<Vec<Value<F>>>,
    pub witness: Value<D::Witness>,
    pub fixed_bases: BTreeMap<String, C>,
    pub fw: FrameworkWitness,
}

impl<D: DeciderStep, const K: u32> Circuit<F> for IvcDeciderCircuit<D, K> {
    type Config = AggCircuitConfig;
    type FloorPlanner = SimpleFloorPlanner;
    type Params = ();

    fn without_witnesses(&self) -> Self {
        let full_width = self.app_state_width + 1;
        Self {
            step: self.step.clone(),
            app_state_width: self.app_state_width,
            child_states: vec![vec![Value::unknown(); full_width]; DECIDER_CHILD_ARITY],
            witness: Value::unknown(),
            fixed_bases: self.fixed_bases.clone(),
            fw: self.fw.without_witnesses(),
        }
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        configure_ivc_circuit(meta)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        let ctx = IvcCtx::new(&config, (K as usize).saturating_sub(1));
        let assigned_vk = ctx.verifier.assign_fixed_vk(
            &mut layouter,
            &self.fw.child_vk_name,
            &self.fw.child_vk.domain,
            &self.fw.child_vk.cs,
            self.fw.child_vk.transcript_repr,
        )?;

        let w = self.app_state_width;

        if self.child_states.len() != DECIDER_CHILD_ARITY {
            return Err(Error::Synthesis("decider arity mismatch".to_string()));
        }
        // Assign full child states
        let child_full_states: Vec<Vec<AssignedNative<F>>> = self
            .child_states
            .iter()
            .map(|vals| assign_values(&ctx, &mut layouter, vals))
            .collect::<Result<Vec<_>, _>>()?;

        // Compute the final Merkle root from children's digests
        let mut layer: Vec<AssignedNative<F>> = child_full_states
            .iter()
            .map(|full| full[w].clone())
            .collect();
        while layer.len() > 1 {
            let mut next = Vec::with_capacity((layer.len() + 1) / 2);
            let mut i = 0usize;
            while i + 1 < layer.len() {
                next.push(ctx.hash2(&mut layouter, &layer[i], &layer[i + 1])?);
                i += 2;
            }
            if i < layer.len() {
                next.push(layer[i].clone());
            }
            layer = next;
        }
        let merkle_root = layer
            .pop()
            .ok_or_else(|| Error::Synthesis("missing decider merkle root".to_string()))?;

        // Run the decider step (gets full states + Merkle root)
        let final_pi = self.step.synthesize(
            &ctx,
            &mut layouter,
            &child_full_states,
            &merkle_root,
            self.witness.clone(),
        )?;

        // Expose the decider's public inputs
        ctx.expose_native(&mut layouter, final_pi)?;

        // RPV for the two top children (which are aggregation proofs)
        let rpv = recursive_partial_verify(
            &ctx,
            &mut layouter,
            RpvInput {
                assigned_vk: &assigned_vk,
                children_are_client_proofs: false,
                fixed_base_names: &self.fw.fixed_base_names,
                child_base_pis: child_full_states,
                child_proofs: self.fw.child_proofs.clone(),
                child_pi_accs: self.fw.child_pi_accs.clone(),
            },
        )?;

        // Resolve fixed bases and fully collapse accumulator before exposure.
        let assigned_fixed_bases: BTreeMap<_, _> = self
            .fixed_bases
            .iter()
            .map(|(name, base)| {
                let assigned = ctx.curve.assign_fixed(&mut layouter, *base)?;
                Ok((name.clone(), assigned))
            })
            .collect::<Result<_, Error>>()?;

        let mut final_acc = rpv.next_acc;
        final_acc.resolve_fixed_bases(&assigned_fixed_bases);
        final_acc.collapse(&mut layouter, &ctx.curve, &ctx.scalar)?;

        // Expose accumulator PI
        let acc_pi = ctx.verifier.as_public_input(&mut layouter, &final_acc)?;
        ctx.expose_native(&mut layouter, acc_pi)?;

        ctx.load(&mut layouter)
    }
}

////////////////////////////////////////////////////////////////////////////////
// Helpers
////////////////////////////////////////////////////////////////////////////////

fn assign_values(
    ctx: &IvcCtx,
    layouter: &mut impl Layouter<F>,
    values: &[Value<F>],
) -> Result<Vec<AssignedNative<F>>, Error> {
    values.iter().map(|v| ctx.assign(layouter, *v)).collect()
}

fn assign_value_vec(
    ctx: &IvcCtx,
    layouter: &mut impl Layouter<F>,
    items: &Value<Vec<F>>,
    width: usize,
) -> Result<Vec<AssignedNative<F>>, Error> {
    (0..width)
        .map(|i| {
            let v = items.as_ref().map(|vs| vs[i]);
            ctx.assign(layouter, v)
        })
        .collect()
}
