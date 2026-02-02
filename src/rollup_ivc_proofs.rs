use std::{collections::BTreeMap, time::Instant};

use ff::Field;
use group::Group;
use rand::rngs::OsRng;
use rayon::prelude::*;
use thiserror::Error;

use midnight_circuits::{
    hash::poseidon::{PoseidonChip, PoseidonState},
    instructions::map::MapCPU,
    types::Instantiable,
    verifier::{Accumulator, AssignedAccumulator, BlstrsEmulation, SelfEmulation},
};
use midnight_curves::Bls12;
use midnight_proofs::{
    circuit::Value,
    plonk::{VerifyingKey, create_proof},
    poly::kzg::{KZGCommitmentScheme, params::ParamsKZG},
    transcript::{CircuitTranscript, Transcript},
};

use crate::{
    rollup_ivc_circuits::{self, AGG_STATE_WIDTH, VkData},
    setup_ivc,
};

pub type S = BlstrsEmulation;
type F = <S as SelfEmulation>::F;
type C = <S as SelfEmulation>::C;
type E = <S as SelfEmulation>::Engine;
type Map = midnight_circuits::map::cpu::MapMt<F, PoseidonChip<F>>;

#[repr(transparent)]
#[derive(Clone)]
struct SendableMap(Map);

// SAFETY: used only to move cloned maps into rayon tasks.
// Each task works on its own cloned `Map` instance.
unsafe impl Send for SendableMap {}
unsafe impl Sync for SendableMap {}

impl SendableMap {
    fn clone_inner(&self) -> Map {
        self.0.clone()
    }
}

#[derive(Debug, Error)]
pub enum AggregationError {
    #[error("need at least one client proof")]
    Empty,

    #[error("client proofs length must be a power of two (got {len})")]
    NotPowerOfTwo { len: usize },

    #[error("client_proofs len must match cached setup (expected {expected}, got {got})")]
    LenMismatch { expected: usize, got: usize },

    #[error("merged final agg requires at least 4 client proofs (got {got})")]
    NeedAtLeastFour { got: usize },

    #[error("max_agg_level mismatch (expected {expected}, got {got})")]
    MaxAggLevelMismatch { expected: usize, got: usize },

    #[error("leaf {leaf} {side} tx root not in historic roots set")]
    HistoricRootMissing { leaf: usize, side: &'static str },

    #[error("leaf {leaf} {side} instance hash mismatch")]
    InstanceMismatch { leaf: usize, side: &'static str },

    #[error("commit boundary mismatch")]
    CommitBoundaryMismatch,

    #[error("null boundary mismatch")]
    NullBoundaryMismatch,

    #[error("roots_set_root mismatch")]
    RootsSetRootMismatch,

    #[error("unexpected AggState field count (expected {expected}, got {got})")]
    AggStateFieldCount { expected: usize, got: usize },

    #[error("root subroot mismatch with recomputed Poseidon tree root")]
    RootPoseidonTreeMismatch,

    #[error("verification prepare failed: {0}")]
    PrepareFailed(&'static str),

    #[error("dual MSM did not check")]
    DualMsmDidNotCheck,

    #[error("accumulator failed final check")]
    AccumulatorFinalCheckFailed,

    #[error("PI accumulator did not check: {0}")]
    PiAccumulatorDidNotCheck(&'static str),

    #[error("leaf AGG proof failed")]
    LeafAggProofFailed,

    #[error("internal AGG proof failed")]
    InternalAggProofFailed,

    #[error("leaf {leaf} swap pair rejected: {reason}")]
    SwapPairRejected { leaf: usize, reason: &'static str },
}

fn ensure(cond: bool, err: AggregationError) -> Result<(), AggregationError> {
    if cond { Ok(()) } else { Err(err) }
}

fn hash_pair(a: F, b: F) -> F {
    use midnight_circuits::instructions::hash::HashCPU;
    <PoseidonChip<F> as HashCPU<F, F>>::hash(&[a, b])
}

// Host-side: single Poseidon hash of all CLIENT_ITEMS_WIDTH public items
fn host_instance_hash(items: [F; rollup_ivc_circuits::CLIENT_ITEMS_WIDTH]) -> F {
    use midnight_circuits::instructions::hash::HashCPU;
    <PoseidonChip<F> as HashCPU<F, F>>::hash(&items)
}

fn poseidon_tree_root(leaves: &[F]) -> Result<F, AggregationError> {
    ensure(!leaves.is_empty(), AggregationError::Empty)?;
    ensure(
        leaves.len().is_power_of_two(),
        AggregationError::NotPowerOfTwo { len: leaves.len() },
    )?;

    fn step(level: Vec<F>) -> Vec<F> {
        level
            .chunks_exact(2)
            .map(|pair| match pair {
                [a, b] => hash_pair(*a, *b),
                _ => unreachable!("chunks_exact(2)"),
            })
            .collect()
    }

    fn go(level: Vec<F>) -> F {
        match level.as_slice() {
            [root] => *root,
            _ => go(step(level)),
        }
    }

    Ok(go(leaves.to_vec()))
}

fn map_insert_many(map: Map, entries: impl IntoIterator<Item = (F, F)>) -> Map {
    entries.into_iter().fold(map, |mut m, (k, v)| {
        m.insert(&k, &v);
        m
    })
}

fn apply_tx_effects(commit_map: Map, null_map: Map, items: [F; 7]) -> (Map, Map) {
    let [_tx_root, _x1, _x2, c1, c2, nf1, nf2] = items;
    let commit_map = map_insert_many(commit_map, [(c1, F::ONE), (c2, F::ONE)]);
    let null_map = map_insert_many(null_map, [(nf1, F::ONE), (nf2, F::ONE)]);
    (commit_map, null_map)
}

fn apply_tx_effects10(
    commit_map: Map,
    null_map: Map,
    items: [F; rollup_ivc_circuits::CLIENT_ITEMS_WIDTH],
) -> (Map, Map) {
    //  Order: [root_before, pk_bx, pk_by, new_c1, new_c2, nf1, nf2, sterms, vto, side]
    let c1 = items[3];
    let c2 = items[4];
    let nf1 = items[5];
    let nf2 = items[6];
    let commit_map = map_insert_many(commit_map, [(c1, F::ONE), (c2, F::ONE)]);
    let null_map = map_insert_many(null_map, [(nf1, F::ONE), (nf2, F::ONE)]);
    (commit_map, null_map)
}

fn check_pair_swap_or_transfer(
    leaf: usize,
    left: &[F; rollup_ivc_circuits::CLIENT_ITEMS_WIDTH],
    right: &[F; rollup_ivc_circuits::CLIENT_ITEMS_WIDTH],
) -> Result<(), AggregationError> {
    // Indices for swap extension
    let l_sterms = left[7];
    let l_vto = left[8];
    let l_side = left[9];
    let r_sterms = right[7];
    let r_vto = right[8];
    let r_side = right[9];

    // SPEC: transfer-mode is indicated by sterms == 0; in that case vto==0 and side==0 must hold.
    let l_is_transfer = l_sterms == F::ZERO;
    let r_is_transfer = r_sterms == F::ZERO;

    // Mixed modes are invalid (must be both transfer or both swap).
    if l_is_transfer != r_is_transfer {
        return Err(AggregationError::SwapPairRejected {
            leaf,
            reason: "mixed transfer/swap pair",
        });
    }

    if l_is_transfer && r_is_transfer {
        ensure(
            l_vto == F::ZERO,
            AggregationError::SwapPairRejected {
                leaf,
                reason: "transfer mode: left vto != 0",
            },
        )?;
        ensure(
            l_side == F::ZERO,
            AggregationError::SwapPairRejected {
                leaf,
                reason: "transfer mode: left side != 0",
            },
        )?;
        ensure(
            r_vto == F::ZERO,
            AggregationError::SwapPairRejected {
                leaf,
                reason: "transfer mode: right vto != 0",
            },
        )?;
        ensure(
            r_side == F::ZERO,
            AggregationError::SwapPairRejected {
                leaf,
                reason: "transfer mode: right side != 0",
            },
        )?;
        return Ok(());
    }

    // Swap case: matching intent digest; side markers must be opposite; vto is per-leg expiry.
    ensure(
        l_sterms == r_sterms,
        AggregationError::SwapPairRejected {
            leaf,
            reason: "sterms mismatch",
        },
    )?;
    ensure(
        l_sterms != F::ZERO,
        AggregationError::SwapPairRejected {
            leaf,
            reason: "sterms is zero",
        },
    )?;

    ensure(
        (l_side == F::ZERO) || (l_side == F::ONE),
        AggregationError::SwapPairRejected {
            leaf,
            reason: "left side not in {0,1}",
        },
    )?;
    ensure(
        (r_side == F::ZERO) || (r_side == F::ONE),
        AggregationError::SwapPairRejected {
            leaf,
            reason: "right side not in {0,1}",
        },
    )?;
    ensure(
        l_side + r_side == F::ONE,
        AggregationError::SwapPairRejected {
            leaf,
            reason: "side markers not opposite (side_L + side_R != 1)",
        },
    )?;

    Ok(())
}

fn verify_and_extract_acc(
    srs: &ParamsKZG<Bls12>,
    vk: &VerifyingKey<F, KZGCommitmentScheme<E>>,
    fixed_bases: &BTreeMap<String, C>,
    proof: &[u8],
    public_inputs: &[F],
) -> Result<Accumulator<S>, AggregationError> {
    let mut transcript = CircuitTranscript::<PoseidonState<F>>::init_from_bytes(proof);
    let committed_bases: &[&[C]] = &[&[C::identity()]];
    let instances: &[&[&[F]]] = &[&[public_inputs]];

    let dual_msm = midnight_proofs::plonk::prepare::<
        F,
        KZGCommitmentScheme<E>,
        CircuitTranscript<PoseidonState<F>>,
    >(vk, committed_bases, instances, &mut transcript)
    .map_err(|_| AggregationError::PrepareFailed("midnight_proofs::plonk::prepare"))?;

    ensure(
        dual_msm.clone().check(&srs.verifier_params()),
        AggregationError::DualMsmDidNotCheck,
    )?;

    let mut acc: Accumulator<S> = dual_msm.into();
    acc.extract_fixed_bases(fixed_bases);
    acc.collapse();

    ensure(
        acc.check(&srs.s_g2().into(), fixed_bases),
        AggregationError::AccumulatorFinalCheckFailed,
    )?;

    Ok(acc)
}

fn collapse_acc(mut acc: Accumulator<S>) -> Accumulator<S> {
    acc.collapse();
    acc
}

fn agg_state_as_values(
    state: &rollup_ivc_circuits::AggState,
) -> Result<[Value<F>; AGG_STATE_WIDTH], AggregationError> {
    let fields = state.to_fields();
    match fields.as_slice() {
        [a, b, c, d, e, f, g] => Ok([
            Value::known(a.clone()),
            Value::known(b.clone()),
            Value::known(c.clone()),
            Value::known(d.clone()),
            Value::known(e.clone()),
            Value::known(f.clone()),
            Value::known(g.clone()),
        ]),
        _ => Err(AggregationError::AggStateFieldCount {
            expected: rollup_ivc_circuits::AGG_STATE_WIDTH,
            got: fields.len(),
        }),
    }
}

#[derive(Clone, Debug)]
pub struct AggPublicInputs {
    pub state: rollup_ivc_circuits::AggState,
    pub pi_acc: rollup_ivc_circuits::AggAccumulator,
}

impl AggPublicInputs {
    pub fn to_fields(&self) -> Vec<F> {
        self.state
            .to_fields()
            .into_iter()
            .chain(AssignedAccumulator::as_public_input(&self.pi_acc).into_iter())
            .collect()
    }
}

////////////////////////////////////////////////////////////////////////////////
// Host-side structures
////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Debug)]
pub struct TreeNode {
    pub state: rollup_ivc_circuits::AggState,
    pub proof: Vec<u8>,
    pub proof_acc: rollup_ivc_circuits::AggAccumulator,
    pub pi_acc: rollup_ivc_circuits::AggAccumulator,
}

#[derive(Clone, Debug)]
pub struct ClientProof {
    pub state: F,
    pub proof: Vec<u8>,
    pub public_items: [F; rollup_ivc_circuits::CLIENT_ITEMS_WIDTH],
}

#[derive(Clone, Debug)]
pub struct AggregationResult {
    pub root_state: rollup_ivc_circuits::AggState,
    pub left_top: TreeNode,
    pub right_top: TreeNode,
    pub child_vk: VkData,
    pub child_vk_name: String,
    pub fixed_base_names: Vec<String>,
    pub fixed_bases: BTreeMap<String, C>,
}

////////////////////////////////////////////////////////////////////////////////
// Planning layer (validated, less error by construction)
////////////////////////////////////////////////////////////////////////////////

#[derive(Clone)]
struct LeafPlan<'a> {
    i: usize,
    children: (&'a ClientProof, &'a ClientProof),
    pre_commitment_map: SendableMap,
    pre_nullifier_map: SendableMap,
    pre_roots_map: SendableMap,
    expected_state: rollup_ivc_circuits::AggState,
}

fn validate_inputs(
    setup: &setup_ivc::AggSetup,
    client_proofs: &[ClientProof],
) -> Result<(), AggregationError> {
    ensure(
        client_proofs.len() == setup.num_leaves,
        AggregationError::LenMismatch {
            expected: setup.num_leaves,
            got: client_proofs.len(),
        },
    )?;
    ensure(!client_proofs.is_empty(), AggregationError::Empty)?;
    ensure(
        client_proofs.len().is_power_of_two(),
        AggregationError::NotPowerOfTwo {
            len: client_proofs.len(),
        },
    )?;

    let num_leaves = client_proofs.len();
    let max_level = (num_leaves as u32).trailing_zeros() as usize;
    ensure(
        max_level >= 2,
        AggregationError::NeedAtLeastFour { got: num_leaves },
    )?;

    let max_agg_level = max_level - 1;
    ensure(
        max_agg_level == setup.max_agg_level,
        AggregationError::MaxAggLevelMismatch {
            expected: setup.max_agg_level,
            got: max_agg_level,
        },
    )?;

    Ok(())
}

fn check_client(
    roots: &Map,
    leaf: usize,
    side: &'static str,
    proof: &ClientProof,
) -> Result<(), AggregationError> {
    let tx_root = proof.public_items[0];

    ensure(
        roots.get(&tx_root) == F::ONE,
        AggregationError::HistoricRootMissing { leaf, side },
    )?;

    ensure(
        host_instance_hash(proof.public_items) == proof.state,
        AggregationError::InstanceMismatch { leaf, side },
    )?;

    Ok(())
}

fn plan_leaves<'a>(
    client_proofs: &'a [ClientProof],
    pre_commitment_map: Map,
    pre_nullifier_map: Map,
    roots_map: Map,
    roots_set_root: F,
    block_level: F,
) -> Result<Vec<LeafPlan<'a>>, AggregationError> {
    let init = (
        Vec::with_capacity(client_proofs.len() / 2),
        pre_commitment_map,
        pre_nullifier_map,
    );

    let (plans, _c, _n) = client_proofs.chunks_exact(2).enumerate().try_fold(
        init,
        |(mut plans, c_map, n_map), (i, chunk)| {
            let (left, right) = match chunk {
                [l, r] => (l, r),
                _ => unreachable!("chunks_exact(2)"),
            };

            check_client(&roots_map, i, "left", left)?;
            check_client(&roots_map, i, "right", right)?;
            check_pair_swap_or_transfer(i, &left.public_items, &right.public_items)?;

            let c_pre = c_map.succinct_repr();
            let n_pre = n_map.succinct_repr();

            let pre_c_for_leaf = c_map.clone();
            let pre_n_for_leaf = n_map.clone();

            let (c1, n1) = apply_tx_effects10(c_map, n_map, left.public_items);
            let (c2, n2) = apply_tx_effects10(c1, n1, right.public_items);

            let expected_state = rollup_ivc_circuits::AggState {
                c_pre,
                c_post: c2.succinct_repr(),
                n_pre,
                n_post: n2.succinct_repr(),
                subroot: hash_pair(left.state, right.state),
                commitment_roots_set_root: roots_set_root,
                block_level,
            };

            plans.push(LeafPlan {
                i,
                children: (left, right),
                pre_commitment_map: SendableMap(pre_c_for_leaf),
                pre_nullifier_map: SendableMap(pre_n_for_leaf),
                pre_roots_map: SendableMap(roots_map.clone()),
                expected_state,
            });

            Ok((plans, c2, n2))
        },
    )?;

    Ok(plans)
}

////////////////////////////////////////////////////////////////////////////////
// Proof construction
////////////////////////////////////////////////////////////////////////////////

fn prove_leaf(
    setup: &setup_ivc::AggSetup,
    leaf_srs: &ParamsKZG<Bls12>,
    leaf_vk: &VerifyingKey<F, KZGCommitmentScheme<E>>,
    plan: &LeafPlan<'_>,
) -> Result<TreeNode, AggregationError> {
    let agg_srs1 = &setup.agg_srs_leaf;
    let agg_srs2 = &setup.agg_srs_internal;

    let leaf_keys = setup.agg_store.get(1);
    let leaf_pk = leaf_keys.pk.clone();
    let leaf_vk_arc = leaf_keys.vk.clone();

    let fixed_base_names = setup.fixed_base_names.clone();
    let fixed_bases = setup.fixed_bases.clone();
    let trivial = setup.trivial_combined.clone();

    let leaf_vk_data = setup.leaf_vk_data.clone();
    let leaf_vk_name = setup.leaf_vk_name.clone();
    let leaf_fixed_bases = setup.leaf_fixed_bases.clone();

    let (left, right) = plan.children;

    let circuit = rollup_ivc_circuits::LeafAggCircuit {
        child_vk: leaf_vk_data,
        child_vk_name: leaf_vk_name,

        left_items: Value::known(left.public_items),
        right_items: Value::known(right.public_items),

        pre_commitment_map: Value::known(plan.pre_commitment_map.clone_inner()),
        pre_nullifier_map: Value::known(plan.pre_nullifier_map.clone_inner()),
        pre_commitment_roots_set_map: Value::known(plan.pre_roots_map.clone_inner()),
        block_level: Value::known(plan.expected_state.block_level),

        left_proof: Value::known(left.proof.clone()),
        right_proof: Value::known(right.proof.clone()),

        left_pi_acc: Value::known(trivial.clone()),
        right_pi_acc: Value::known(trivial.clone()),

        fixed_base_names: fixed_base_names.clone(),
    };

    let acc_l = verify_and_extract_acc(
        leaf_srs,
        leaf_vk,
        &leaf_fixed_bases,
        &left.proof,
        &left.public_items,
    )?;
    let acc_r = verify_and_extract_acc(
        leaf_srs,
        leaf_vk,
        &leaf_fixed_bases,
        &right.proof,
        &right.public_items,
    )?;

    let pi_acc = collapse_acc(Accumulator::accumulate(&[
        acc_l,
        trivial.clone(),
        acc_r,
        trivial.clone(),
    ]));

    ensure(
        pi_acc.check(&agg_srs2.s_g2().into(), &fixed_bases),
        AggregationError::PiAccumulatorDidNotCheck("leaf accumulated PI"),
    )?;

    let public_inputs = AggPublicInputs {
        state: plan.expected_state,
        pi_acc: pi_acc.clone(),
    };
    let public_inputs_fields = public_inputs.to_fields();

    let start = Instant::now();
    let proof = {
        let mut transcript = CircuitTranscript::<PoseidonState<F>>::init();
        create_proof::<
            F,
            KZGCommitmentScheme<E>,
            CircuitTranscript<PoseidonState<F>>,
            rollup_ivc_circuits::LeafAggCircuit,
        >(
            agg_srs1,
            leaf_pk.as_ref(),
            &[circuit],
            1,
            &[&[&[], &public_inputs_fields]],
            OsRng,
            &mut transcript,
        )
        .map_err(|_| AggregationError::LeafAggProofFailed)?;
        transcript.finalize()
    };

    println!("Leaf AGG {} created in {:?}", plan.i, start.elapsed());

    let proof_acc = verify_and_extract_acc(
        agg_srs1,
        leaf_vk_arc.as_ref(),
        &leaf_keys.fixed_bases,
        &proof,
        &public_inputs_fields,
    )?;

    Ok(TreeNode {
        state: public_inputs.state,
        proof,
        proof_acc,
        pi_acc,
    })
}

fn prove_parent(
    setup: &setup_ivc::AggSetup,
    parent_level: usize,
    child_level: usize,
    children: (&TreeNode, &TreeNode),
) -> Result<TreeNode, AggregationError> {
    let (left, right) = children;

    ensure(
        left.state.c_post == right.state.c_pre,
        AggregationError::CommitBoundaryMismatch,
    )?;
    ensure(
        left.state.n_post == right.state.n_pre,
        AggregationError::NullBoundaryMismatch,
    )?;
    ensure(
        left.state.commitment_roots_set_root == right.state.commitment_roots_set_root,
        AggregationError::RootsSetRootMismatch,
    )?;
    ensure(
        left.state.block_level == right.state.block_level,
        AggregationError::RootsSetRootMismatch, // reuse (or add a dedicated error if you prefer)
    )?;

    let child_keys = setup.agg_store.get(child_level);
    let parent_keys = setup.agg_store.get(parent_level);

    let agg_srs2 = &setup.agg_srs_internal;
    let fixed_base_names = setup.fixed_base_names.clone();
    let fixed_bases = setup.fixed_bases.clone();

    let state = rollup_ivc_circuits::AggState {
        c_pre: left.state.c_pre,
        c_post: right.state.c_post,
        n_pre: left.state.n_pre,
        n_post: right.state.n_post,
        subroot: hash_pair(left.state.subroot, right.state.subroot),
        commitment_roots_set_root: left.state.commitment_roots_set_root,
        block_level: left.state.block_level,
    };

    let circuit = rollup_ivc_circuits::InternalAggCircuit {
        child_vk: child_keys.vk_data.clone(),
        child_vk_name: child_keys.name.clone(),

        left_child_state: agg_state_as_values(&left.state)?,
        right_child_state: agg_state_as_values(&right.state)?,

        left_proof: Value::known(left.proof.clone()),
        right_proof: Value::known(right.proof.clone()),

        left_pi_acc: Value::known(left.pi_acc.clone()),
        right_pi_acc: Value::known(right.pi_acc.clone()),

        fixed_base_names: fixed_base_names.clone(),
    };

    let pi_acc = collapse_acc(Accumulator::accumulate(&[
        left.proof_acc.clone(),
        left.pi_acc.clone(),
        right.proof_acc.clone(),
        right.pi_acc.clone(),
    ]));

    ensure(
        pi_acc.check(&agg_srs2.s_g2().into(), &fixed_bases),
        AggregationError::PiAccumulatorDidNotCheck("internal accumulated PI"),
    )?;

    let public_inputs = AggPublicInputs {
        state,
        pi_acc: pi_acc.clone(),
    };
    let public_inputs_fields = public_inputs.to_fields();

    let start = Instant::now();
    let proof = {
        let mut transcript = CircuitTranscript::<PoseidonState<F>>::init();
        create_proof::<
            F,
            KZGCommitmentScheme<E>,
            CircuitTranscript<PoseidonState<F>>,
            rollup_ivc_circuits::InternalAggCircuit,
        >(
            agg_srs2,
            parent_keys.pk.as_ref(),
            &[circuit],
            1,
            &[&[&[], &public_inputs_fields]],
            OsRng,
            &mut transcript,
        )
        .map_err(|_| AggregationError::InternalAggProofFailed)?;
        transcript.finalize()
    };

    println!(
        "Internal level {} node created in {:?}",
        parent_level,
        start.elapsed()
    );

    let proof_acc = verify_and_extract_acc(
        agg_srs2,
        parent_keys.vk.as_ref(),
        &parent_keys.fixed_bases,
        &proof,
        &public_inputs_fields,
    )?;

    Ok(TreeNode {
        state: public_inputs.state,
        proof,
        proof_acc,
        pi_acc,
    })
}

/// Reduce one level: [n0,n1,n2,n3,...] -> [p0,p1,...], pairing without indexing.
fn build_next_level(
    setup: &setup_ivc::AggSetup,
    parent_level: usize,
    child_level: usize,
    nodes: Vec<TreeNode>,
) -> Result<Vec<TreeNode>, AggregationError> {
    nodes
        .par_chunks_exact(2)
        .map(|chunk| match chunk {
            [l, r] => prove_parent(setup, parent_level, child_level, (l, r)),
            _ => unreachable!("par_chunks_exact(2)"),
        })
        .collect()
}

/// Recursively build internal levels to the top pair, returning a tuple (left,right) instead
/// of “unpacking from a vec”.
fn build_to_top_pair(
    setup: &setup_ivc::AggSetup,
    child_level: usize,
    nodes: Vec<TreeNode>,
) -> Result<(usize, (TreeNode, TreeNode)), AggregationError> {
    match nodes.as_slice() {
        [left, right] => Ok((child_level, (left.clone(), right.clone()))),
        _ => {
            let parent_level = child_level + 1;
            let next = build_next_level(setup, parent_level, child_level, nodes)?;
            build_to_top_pair(setup, parent_level, next)
        }
    }
}

fn finalize(
    setup: &setup_ivc::AggSetup,
    client_proofs: &[ClientProof],
    child_level: usize,
    top: (TreeNode, TreeNode),
) -> Result<AggregationResult, AggregationError> {
    let (left_top, right_top) = top;

    let root_state = rollup_ivc_circuits::AggState {
        c_pre: left_top.state.c_pre,
        c_post: right_top.state.c_post,
        n_pre: left_top.state.n_pre,
        n_post: right_top.state.n_post,
        subroot: hash_pair(left_top.state.subroot, right_top.state.subroot),
        commitment_roots_set_root: left_top.state.commitment_roots_set_root,
        block_level: left_top.state.block_level,
    };

    let expected_root =
        poseidon_tree_root(&client_proofs.iter().map(|p| p.state).collect::<Vec<F>>())?;

    ensure(
        root_state.subroot == expected_root,
        AggregationError::RootPoseidonTreeMismatch,
    )?;

    let child_keys = setup.agg_store.get(child_level);
    let child_vk = VkData {
        domain: child_keys.vk_data.domain.clone(),
        cs: child_keys.vk_data.cs.clone(),
        transcript_repr: child_keys.vk_data.transcript_repr,
    };

    Ok(AggregationResult {
        root_state,
        left_top,
        right_top,
        child_vk,
        child_vk_name: child_keys.name.clone(),
        fixed_base_names: setup.fixed_base_names.clone(),
        fixed_bases: setup.fixed_bases.clone(),
    })
}

////////////////////////////////////////////////////////////////////////////////
// Public API
////////////////////////////////////////////////////////////////////////////////

pub fn try_aggregate_client_proofs_cached(
    setup: &setup_ivc::AggSetup,
    leaf_srs: &ParamsKZG<Bls12>,
    leaf_vk: &VerifyingKey<F, KZGCommitmentScheme<E>>,
    client_proofs: &[ClientProof],
    pre_commitment_map: Map,
    pre_nullifier_map: Map,
    pre_commitment_roots_map: Map,
    batch_block_level: F,
) -> Result<AggregationResult, AggregationError> {
    validate_inputs(setup, client_proofs)?;

    // Bind a single roots-set root across the whole agg tree (host side).
    let roots_set_root = pre_commitment_roots_map.succinct_repr();

    // 1) plan leaves (invariants checked once)
    let leaf_plans = plan_leaves(
        client_proofs,
        pre_commitment_map,
        pre_nullifier_map,
        pre_commitment_roots_map,
        roots_set_root,
        batch_block_level,
    )?;

    // 2) prove leaves (parallel)
    let leaf_nodes = leaf_plans
        .par_iter()
        .map(|p| prove_leaf(setup, leaf_srs, leaf_vk, p))
        .collect::<Result<Vec<_>, _>>()?;

    // 3) build internal levels to (left,right) tuple
    let (child_level, top_pair) = build_to_top_pair(setup, 1, leaf_nodes)?;

    // 4) finalize
    finalize(setup, client_proofs, child_level, top_pair)
}

/// Backwards-compatible wrapper that preserves the old signature.
pub fn aggregate_client_proofs_cached(
    setup: &setup_ivc::AggSetup,
    leaf_srs: &ParamsKZG<Bls12>,
    leaf_vk: &VerifyingKey<F, KZGCommitmentScheme<E>>,
    client_proofs: &[ClientProof],
    pre_commitment_map: Map,
    pre_nullifier_map: Map,
    pre_commitment_roots_map: Map,
    batch_block_level: F,
) -> AggregationResult {
    try_aggregate_client_proofs_cached(
        setup,
        leaf_srs,
        leaf_vk,
        client_proofs,
        pre_commitment_map,
        pre_nullifier_map,
        pre_commitment_roots_map,
        batch_block_level,
    )
    .expect("aggregation failed") // TODO remove expect
}
