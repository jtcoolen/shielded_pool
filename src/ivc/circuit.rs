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
    Acc, DeciderStep, F, FoldStep, LeafStep, VkData,
    ctx::{
        AggCircuitConfig, IvcCtx, RpvInput,
        configure_ivc_circuit, expose_node_outputs, recursive_partial_verify,
    },
    AssignedNative,
};

////////////////////////////////////////////////////////////////////////////////
// Framework witness — fields managed by the framework, not the application
////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Debug)]
pub struct FrameworkWitness {
    pub child_vk: VkData,
    pub child_vk_name: String,
    pub left_proof: Value<Vec<u8>>,
    pub right_proof: Value<Vec<u8>>,
    pub left_pi_acc: Value<Acc>,
    pub right_pi_acc: Value<Acc>,
    pub fixed_base_names: Vec<String>,
}

impl FrameworkWitness {
    pub fn without_witnesses(&self) -> Self {
        Self {
            child_vk: self.child_vk.clone(),
            child_vk_name: self.child_vk_name.clone(),
            left_proof: Value::unknown(),
            right_proof: Value::unknown(),
            left_pi_acc: Value::unknown(),
            right_pi_acc: Value::unknown(),
            fixed_base_names: self.fixed_base_names.clone(),
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Shared synthesis core
//
// Every IVC circuit follows the same three-phase pattern:
//   1. Run the application's step function  → (state, left_child_pi, right_child_pi)
//   2. Framework appends Merkle digest to the state
//   3. Recursive partial verify + expose outputs
////////////////////////////////////////////////////////////////////////////////

/// Output produced by the step-phase closure inside `synthesize_node`.
struct StepPhaseOutput {
    full_state: Vec<AssignedNative<F>>,
    left_child_pi: Vec<AssignedNative<F>>,
    right_child_pi: Vec<AssignedNative<F>>,
}

fn synthesize_node<const K: u32, L: Layouter<F>>(
    config: AggCircuitConfig,
    layouter: &mut L,
    fw: &FrameworkWitness,
    children_are_client_proofs: bool,
    step_phase: impl FnOnce(&IvcCtx, &mut L) -> Result<StepPhaseOutput, Error>,
) -> Result<(), Error> {
    let ctx = IvcCtx::new(&config, (K as usize).saturating_sub(1));
    let assigned_vk = ctx.verifier.assign_vk_to_fixed(
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
            left_base_pi: out.left_child_pi,
            right_base_pi: out.right_child_pi,
            left_proof: fw.left_proof.clone(),
            right_proof: fw.right_proof.clone(),
            left_pi_acc: fw.left_pi_acc.clone(),
            right_pi_acc: fw.right_pi_acc.clone(),
        },
    )?;

    expose_node_outputs(&ctx, layouter, out.full_state, &rpv.next_acc)?;
    ctx.load(layouter)
}

////////////////////////////////////////////////////////////////////////////////
// IvcLeafCircuit<L, K>
//
// Subcircuit at the leaves of the binary tree.  Verifies two raw client
// proofs, runs the application's LeafStep, and produces the base Merkle
// digest H(h(x_left), h(x_right)).
////////////////////////////////////////////////////////////////////////////////

#[derive(Clone)]
pub struct IvcLeafCircuit<L: LeafStep, const K: u32> {
    pub(crate) step: L,
    pub(crate) left_client_items: Value<Vec<F>>,
    pub(crate) right_client_items: Value<Vec<F>>,
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
            left_client_items: Value::unknown(),
            right_client_items: Value::unknown(),
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
        let left_items = self.left_client_items.clone();
        let right_items = self.right_client_items.clone();
        let witness = self.witness.clone();
        let width = self.client_pi_width;

        synthesize_node::<K, _>(
            config,
            &mut layouter,
            &self.fw,
            true, // children ARE client proofs
            |ctx, layouter| {
                // 1. Assign client PIs
                let left_pi = assign_value_vec(ctx, layouter, &left_items, width)?;
                let right_pi = assign_value_vec(ctx, layouter, &right_items, width)?;

                // 2. Application step (only sees client PIs)
                let app_state = step.synthesize(ctx, layouter, &left_pi, &right_pi, witness)?;

                // 3. Framework: Merkle hashes
                let left_hash = ctx.hash_many(layouter, &left_pi)?;
                let right_hash = ctx.hash_many(layouter, &right_pi)?;
                let digest = ctx.hash2(layouter, &left_hash, &right_hash)?;

                // 4. Full state = [app_state..., digest]
                let mut full = app_state;
                full.push(digest);

                Ok(StepPhaseOutput {
                    full_state: full,
                    left_child_pi: left_pi,
                    right_child_pi: right_pi,
                })
            },
        )
    }
}

////////////////////////////////////////////////////////////////////////////////
// IvcNodeCircuit<Fo, K>
//
// Subcircuit at internal nodes.  Verifies two child aggregation proofs,
// runs the application's FoldStep on the app-state portions, and computes
// the parent Merkle digest H(left_digest, right_digest).
////////////////////////////////////////////////////////////////////////////////

#[derive(Clone)]
pub struct IvcNodeCircuit<Fo: FoldStep, const K: u32> {
    pub(crate) step: Fo,
    pub(crate) app_state_width: usize,
    pub(crate) left_child_state: Vec<Value<F>>,
    pub(crate) right_child_state: Vec<Value<F>>,
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
            left_child_state: vec![Value::unknown(); full_width],
            right_child_state: vec![Value::unknown(); full_width],
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
        let left_vals = self.left_child_state.clone();
        let right_vals = self.right_child_state.clone();
        let w = self.app_state_width;

        synthesize_node::<K, _>(
            config,
            &mut layouter,
            &self.fw,
            false, // children are NOT client proofs
            |ctx, layouter| {
                // 1. Assign full child states [app..., digest]
                let left_full = assign_values(ctx, layouter, &left_vals)?;
                let right_full = assign_values(ctx, layouter, &right_vals)?;

                // 2. Split into app state and Merkle digest
                let (left_app, left_digest) = left_full.split_at(w);
                let (right_app, right_digest) = right_full.split_at(w);

                // 3. Application fold (only sees app states)
                let app_state = step.synthesize(ctx, layouter, left_app, right_app)?;

                // 4. Framework: parent Merkle digest
                let digest = ctx.hash2(layouter, &left_digest[0], &right_digest[0])?;

                let mut full = app_state;
                full.push(digest);

                Ok(StepPhaseOutput {
                    full_state: full,
                    left_child_pi: left_full,
                    right_child_pi: right_full,
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
        let assigned_vk = ctx.verifier.assign_vk_to_fixed(
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
                left_base_pi: left_full,
                right_base_pi: right_full,
                left_proof: self.fw.left_proof.clone(),
                right_proof: self.fw.right_proof.clone(),
                left_pi_acc: self.fw.left_pi_acc.clone(),
                right_pi_acc: self.fw.right_pi_acc.clone(),
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
