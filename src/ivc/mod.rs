//! Generic IVC framework for proof aggregation with Merkle instance commitment.
//!
//! Given 2^d proofs for the same relation R(x, w), constructs an aggregated
//! proof for:
//!
//! ```text
//! PoK{ {(w_i, x_i)}_i : R(x_i, w_i) = 1  ∧  MerkleHash({x_i}_i) = H }
//! ```
//!
//! # Architecture
//!
//! The framework is split into three layers:
//!
//! 1. **Step traits** ([`LeafStep`], [`FoldStep`], [`DeciderStep`]) — the
//!    application implements these to define its state transition.  The Merkle
//!    instance commitment (`H`) is managed by the framework automatically.
//!
//! 2. **Generic circuits** ([`IvcLeafCircuit`], [`IvcNodeCircuit`],
//!    [`IvcDeciderCircuit`]) — each wraps a step function with recursive
//!    partial verification and accumulator folding.  The application never
//!    constructs these directly.
//!
//! 3. **Engine** ([`IvcSetup`], [`IvcProver`]) — one-shot key generation and
//!    parallel tree proof construction.

pub mod ctx;
pub mod circuit;
pub mod engine;

// Re-export the public API.
pub use ctx::IvcCtx;
pub use circuit::IvcDeciderCircuit;
pub use engine::IvcProver;

use std::fmt::Debug;

use midnight_circuits::{
    ecc::foreign::ForeignEccChip,
    field::{
        NativeGadget,
        decomposition::chip::P2RDecompositionChip,
        native::NativeChip,
    },
    hash::poseidon::PoseidonChip,
    map::cpu::MapMt,
    types::{AssignedForeignPoint, AssignedNative},
    verifier::{Accumulator, BlstrsEmulation, SelfEmulation},
};
use midnight_proofs::{
    circuit::Value,
    plonk::{ConstraintSystem, Error},
    poly::EvaluationDomain,
};

////////////////////////////////////////////////////////////////////////////////
// Shared type aliases (single source of truth for the whole crate)
////////////////////////////////////////////////////////////////////////////////

pub type S = BlstrsEmulation;
pub type F = <S as SelfEmulation>::F;
pub type C = <S as SelfEmulation>::C;
pub type E = <S as SelfEmulation>::Engine;

pub type NG = NativeGadget<F, P2RDecompositionChip<F>, NativeChip<F>>;
pub type CurveChip = ForeignEccChip<F, C, C, NG, NG>;
pub type MapGadget = midnight_circuits::map::map_gadget::MapGadget<F, NG, PoseidonChip<F>>;
pub type IdPoint = AssignedForeignPoint<
    midnight_curves::Fq,
    midnight_curves::G1Projective,
    midnight_curves::G1Projective,
>;

pub type Map = MapMt<F, PoseidonChip<F>>;
pub type Acc = Accumulator<S>;

////////////////////////////////////////////////////////////////////////////////
// Core types
////////////////////////////////////////////////////////////////////////////////

/// Verification key data needed by aggregation circuits.
#[derive(Clone, Debug)]
pub struct VkData {
    pub domain: EvaluationDomain<F>,
    pub cs: ConstraintSystem<F>,
    pub transcript_repr: F,
}

/// A client proof with its public inputs and precomputed instance hash.
#[derive(Clone, Debug)]
pub struct ClientProof {
    pub proof: Vec<u8>,
    pub public_inputs: Vec<F>,
    pub instance_hash: F,
}

/// A node in the IVC proof tree (produced by leaf or internal provers).
#[derive(Clone, Debug)]
pub struct TreeNode<S> {
    pub app_state: S,
    pub merkle_digest: F,
    pub proof: Vec<u8>,
    pub proof_acc: Acc,
    pub pi_acc: Acc,
}

/// Full node state on the host side (application state + framework-managed digest).
#[derive(Clone, Debug)]
pub struct NodeState<S> {
    pub app_state: S,
    pub merkle_digest: F,
}

#[allow(dead_code)]
impl<S: HostState> NodeState<S> {
    /// Flatten into the field representation used as circuit public inputs
    /// (before the accumulator PI).
    pub fn to_fields(&self) -> Vec<F> {
        let mut fields = self.app_state.to_fields();
        fields.push(self.merkle_digest);
        fields
    }

    pub fn full_width(&self) -> usize {
        S::WIDTH + 1
    }
}

/// Result of proving the full binary tree (everything except the decider).
#[derive(Clone, Debug)]
pub struct TreeResult<S> {
    pub left_top: TreeNode<S>,
    pub right_top: TreeNode<S>,
    pub root_state: NodeState<S>,
}

////////////////////////////////////////////////////////////////////////////////
// Host-side state trait
////////////////////////////////////////////////////////////////////////////////

/// Application state that can be serialized to/from field elements.
///
/// Implemented by the application (e.g. `RollupAppState`).
#[allow(dead_code)]
pub trait HostState: Clone + Debug + Send + Sync {
    const WIDTH: usize;
    fn to_fields(&self) -> Vec<F>;
    fn from_fields(fields: &[F]) -> Self;
}

////////////////////////////////////////////////////////////////////////////////
// Step function traits
////////////////////////////////////////////////////////////////////////////////

/// Leaf step: processes two client proof instances.
///
/// The framework assigns the client PIs, hashes them for the Merkle
/// commitment, and handles recursive partial verification.  The step
/// function only defines the application-specific state transition.
///
/// # Contract
///
/// `synthesize` receives the already-assigned client public inputs and
/// must return the **application state** (a `Vec<AssignedNative<F>>` of
/// length `HostState::WIDTH`).  The Merkle digest is appended by the
/// framework.
pub trait LeafStep: Clone + Send + Sync {
    type Witness: Clone + Send;

    fn synthesize<L: midnight_proofs::circuit::Layouter<F>>(
        &self,
        ctx: &IvcCtx,
        layouter: &mut L,
        left_pi: &[AssignedNative<F>],
        right_pi: &[AssignedNative<F>],
        witness: Value<Self::Witness>,
    ) -> Result<Vec<AssignedNative<F>>, Error>;
}

/// Fold step: merges two child application states.
///
/// The framework assigns the full child states (including the
/// Merkle digest), splits off the digest, passes only the application
/// fields to `synthesize`, then computes the parent digest as
/// `H(left_digest, right_digest)`.
///
/// # Contract
///
/// `synthesize` receives two slices of length `HostState::WIDTH` and
/// returns a new application state of the same length.
pub trait FoldStep: Clone + Send + Sync {
    fn synthesize<L: midnight_proofs::circuit::Layouter<F>>(
        &self,
        ctx: &IvcCtx,
        layouter: &mut L,
        left_app_state: &[AssignedNative<F>],
        right_app_state: &[AssignedNative<F>],
    ) -> Result<Vec<AssignedNative<F>>, Error>;
}

/// Decider step: final wrap circuit.
///
/// Receives both children's full states (app + digest) plus the
/// framework-computed Merkle root, and produces the final on-chain
/// public inputs.  The framework handles RPV and accumulator
/// exposure after this.
pub trait DeciderStep: Clone + Send + Sync {
    type Witness: Clone + Send;

    fn synthesize<L: midnight_proofs::circuit::Layouter<F>>(
        &self,
        ctx: &IvcCtx,
        layouter: &mut L,
        left_full_state: &[AssignedNative<F>],
        right_full_state: &[AssignedNative<F>],
        merkle_root: &AssignedNative<F>,
        witness: Value<Self::Witness>,
    ) -> Result<Vec<AssignedNative<F>>, Error>;
}

////////////////////////////////////////////////////////////////////////////////
// Host-side step traits (for the prover's planning layer)
////////////////////////////////////////////////////////////////////////////////

/// Host-side leaf planning: validates client proofs and computes the
/// expected application state.
#[allow(dead_code)]
pub trait HostLeafStep {
    type AppState: HostState;

    fn plan_pair(
        &self,
        left: &ClientProof,
        right: &ClientProof,
    ) -> Result<Self::AppState, engine::AggregationError>;
}

/// Host-side fold: validates stitching and computes the merged state.
#[allow(dead_code)]
pub trait HostFoldStep {
    type AppState: HostState;

    fn validate_and_merge(
        &self,
        left: &NodeState<Self::AppState>,
        right: &NodeState<Self::AppState>,
    ) -> Result<Self::AppState, engine::AggregationError>;
}
