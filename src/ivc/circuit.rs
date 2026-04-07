//! Generic IVC circuits: leaf, node, and decider.
//!
//! Each circuit wraps a user-provided step function with the framework's
//! recursive partial verification and accumulator folding.  The user never
//! constructs these directly — [`super::engine::IvcProver`] does.

use midnight_circuits::instructions::PublicInputInstructions;
use midnight_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{Circuit, ConstraintSystem, Error},
};

use super::{
    Acc, AssignedNative, DeciderStep, F, FoldStep, LeafStep, VkData,
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

    expose_node_outputs(&ctx, layouter, out.full_state, &rpv.next_acc)?;
    ctx.load(layouter)
}

////////////////////////////////////////////////////////////////////////////////
// IvcLeafCircuit<L, K>
//
// Subcircuit at the leaves of the binary tree.  Verifies four raw client
// proofs, runs the application's LeafStep, and produces the base Merkle
// digest H(H(h(x0), h(x1)), H(h(x2), h(x3))).
////////////////////////////////////////////////////////////////////////////////

#[derive(Clone)]
pub struct IvcLeafCircuit<L: LeafStep, const K: u32> {
    pub(crate) step: L,
    pub(crate) client_items: [Value<Vec<F>>; 4],
    pub(crate) witness: Value<L::Witness>,
    pub(crate) client_pi_width: usize,
    pub(crate) fw: FrameworkWitness,
}

impl<L: LeafStep, const K: u32> Circuit<F> for IvcLeafCircuit<L, K> {
    type Config = AggCircuitConfig;
    type FloorPlanner = SimpleFloorPlanner;
    type Params = ();

    fn without_witnesses(&self) -> Self {
        Self {
            step: self.step.clone(),
            client_items: [
                Value::unknown(),
                Value::unknown(),
                Value::unknown(),
                Value::unknown(),
            ],
            witness: Value::unknown(),
            client_pi_width: self.client_pi_width,
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
            |ctx, layouter| {
                // 1. Assign client PIs
                let p0 = assign_value_vec(ctx, layouter, &items[0], width)?;
                let p1 = assign_value_vec(ctx, layouter, &items[1], width)?;
                let p2 = assign_value_vec(ctx, layouter, &items[2], width)?;
                let p3 = assign_value_vec(ctx, layouter, &items[3], width)?;

                // 2. Application step (only sees client PIs)
                let app_state = step.synthesize(ctx, layouter, &p0, &p1, &p2, &p3, witness)?;

                // 3. Framework: Merkle hashes
                let h0 = ctx.hash_many(layouter, &p0)?;
                let h1 = ctx.hash_many(layouter, &p1)?;
                let h2 = ctx.hash_many(layouter, &p2)?;
                let h3 = ctx.hash_many(layouter, &p3)?;
                let h01 = ctx.hash2(layouter, &h0, &h1)?;
                let h23 = ctx.hash2(layouter, &h2, &h3)?;
                let digest = ctx.hash2(layouter, &h01, &h23)?;

                // 4. Full state = [app_state..., digest]
                let mut full = app_state;
                full.push(digest);

                Ok(StepPhaseOutput {
                    full_state: full,
                    child_pis: vec![p0, p1, p2, p3],
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
    pub(crate) child0_state: Vec<Value<F>>,
    pub(crate) child1_state: Vec<Value<F>>,
    pub(crate) child2_state: Vec<Value<F>>,
    pub(crate) child3_state: Vec<Value<F>>,
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
            child0_state: vec![Value::unknown(); full_width],
            child1_state: vec![Value::unknown(); full_width],
            child2_state: vec![Value::unknown(); full_width],
            child3_state: vec![Value::unknown(); full_width],
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
        let child0_vals = self.child0_state.clone();
        let child1_vals = self.child1_state.clone();
        let child2_vals = self.child2_state.clone();
        let child3_vals = self.child3_state.clone();
        let w = self.app_state_width;

        synthesize_node::<K, _>(
            config,
            &mut layouter,
            &self.fw,
            false, // children are NOT client proofs
            |ctx, layouter| {
                // 1. Assign full child states [app..., digest]
                let child0_full = assign_values(ctx, layouter, &child0_vals)?;
                let child1_full = assign_values(ctx, layouter, &child1_vals)?;
                let child2_full = assign_values(ctx, layouter, &child2_vals)?;
                let child3_full = assign_values(ctx, layouter, &child3_vals)?;

                // 2. Split into app state and Merkle digest
                let (child0_app, child0_digest) = child0_full.split_at(w);
                let (child1_app, child1_digest) = child1_full.split_at(w);
                let (child2_app, child2_digest) = child2_full.split_at(w);
                let (child3_app, child3_digest) = child3_full.split_at(w);

                // 3. Application fold (only sees app states)
                let app01 = step.synthesize(ctx, layouter, child0_app, child1_app)?;
                let app012 = step.synthesize(ctx, layouter, &app01, child2_app)?;
                let app_state = step.synthesize(ctx, layouter, &app012, child3_app)?;

                // 4. Framework: parent Merkle digest
                let d01 = ctx.hash2(layouter, &child0_digest[0], &child1_digest[0])?;
                let d23 = ctx.hash2(layouter, &child2_digest[0], &child3_digest[0])?;
                let digest = ctx.hash2(layouter, &d01, &d23)?;

                let mut full = app_state;
                full.push(digest);

                Ok(StepPhaseOutput {
                    full_state: full,
                    child_pis: vec![child0_full, child1_full, child2_full, child3_full],
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
    pub left_child_state: Vec<Value<F>>,
    pub right_child_state: Vec<Value<F>>,
    pub witness: Value<D::Witness>,
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
            left_child_state: vec![Value::unknown(); full_width],
            right_child_state: vec![Value::unknown(); full_width],
            witness: Value::unknown(),
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

        // Assign full child states
        let left_full = assign_values(&ctx, &mut layouter, &self.left_child_state)?;
        let right_full = assign_values(&ctx, &mut layouter, &self.right_child_state)?;

        // Compute the final Merkle root from children's digests
        let left_digest = &left_full[w];
        let right_digest = &right_full[w];
        let merkle_root = ctx.hash2(&mut layouter, left_digest, right_digest)?;

        // Run the decider step (gets full states + Merkle root)
        let final_pi = self.step.synthesize(
            &ctx,
            &mut layouter,
            &left_full,
            &right_full,
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
                child_base_pis: vec![left_full, right_full],
                child_proofs: self.fw.child_proofs.clone(),
                child_pi_accs: self.fw.child_pi_accs.clone(),
            },
        )?;

        // Expose accumulator PI
        let acc_pi = ctx.verifier.as_public_input(&mut layouter, &rpv.next_acc)?;
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
