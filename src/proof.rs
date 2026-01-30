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
    plonk::{ConstraintSystem, VerifyingKey, create_proof},
    poly::{
        EvaluationDomain,
        kzg::{KZGCommitmentScheme, params::ParamsKZG},
    },
    transcript::{CircuitTranscript, Transcript},
};

use crate::{rollup_ivc, setup};

pub type S = BlstrsEmulation;
type F = <S as SelfEmulation>::F;
type C = <S as SelfEmulation>::C;
type E = <S as SelfEmulation>::Engine;
type Map = midnight_circuits::map::cpu::MapMt<F, PoseidonChip<F>>;

#[repr(transparent)]
#[derive(Clone)]
struct SendableMap(Map);

// SAFETY: `SendableMap` is only used by cloning the inner `Map` per task.
// The code assumes cloning yields independent, thread-safe instances for read-only use
// and for local mutation within each worker thread.
unsafe impl Send for SendableMap {}
unsafe impl Sync for SendableMap {}

impl SendableMap {
    fn clone_inner(&self) -> Map {
        self.0.clone()
    }
}

#[derive(Debug, Clone, Copy)]
enum Side {
    Left,
    Right,
}
impl Side {
    const fn as_str(self) -> &'static str {
        match self {
            Side::Left => "left",
            Side::Right => "right",
        }
    }
}

#[derive(Debug, Error)]
pub enum AggregationError {
    #[error("need at least one client proof")]
    ClientProofsEmpty,

    #[error("client proofs length must be a power of two (got {len})")]
    ClientProofsNotPowerOfTwo { len: usize },

    #[error("client_proofs len must match cached setup (expected {expected}, got {got})")]
    ClientProofsLenMismatch { expected: usize, got: usize },

    #[error("merged final agg requires at least 4 client proofs (got {got})")]
    NeedAtLeastFourClientProofs { got: usize },

    #[error("max_agg_level mismatch (expected {expected}, got {got})")]
    MaxAggLevelMismatch { expected: usize, got: usize },

    #[error("leaf {leaf} {side} tx root not in historic roots set")]
    HistoricRootMissing { leaf: usize, side: &'static str },

    #[error("{side} client instance mismatch (leaf {leaf})")]
    InstanceMismatch { leaf: usize, side: &'static str },

    #[error("leaf {leaf} planned subroot mismatch")]
    PlannedSubrootMismatch { leaf: usize },

    #[error("commit boundary mismatch")]
    CommitBoundaryMismatch,

    #[error("null boundary mismatch")]
    NullBoundaryMismatch,

    #[error("roots_set_root mismatch")]
    RootsSetRootMismatch,

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

    #[error("unexpected AggState field count (expected {expected}, got {got})")]
    UnexpectedAggStateFieldCount { expected: usize, got: usize },

    #[error("expected to stop at top pair (got {got})")]
    ExpectedTopPair { got: usize },
}

fn ensure(cond: bool, err: AggregationError) -> Result<(), AggregationError> {
    if cond { Ok(()) } else { Err(err) }
}

fn hash_pair(a: F, b: F) -> F {
    use midnight_circuits::instructions::hash::HashCPU;
    <PoseidonChip<F> as HashCPU<F, F>>::hash(&[a, b])
}

fn poseidon_tree_root(leaf_states: &[F]) -> Result<F, AggregationError> {
    ensure(!leaf_states.is_empty(), AggregationError::ClientProofsEmpty)?;
    ensure(
        leaf_states.len().is_power_of_two(),
        AggregationError::ClientProofsNotPowerOfTwo {
            len: leaf_states.len(),
        },
    )?;

    fn reduce_level(level: Vec<F>) -> Vec<F> {
        level
            .chunks_exact(2)
            .map(|chunk| match chunk {
                [a, b] => hash_pair(*a, *b),
                _ => unreachable!("chunks_exact(2) guarantees pairs"),
            })
            .collect()
    }

    fn go(level: Vec<F>) -> F {
        match level.as_slice() {
            [root] => *root,
            _ => go(reduce_level(level)),
        }
    }

    Ok(go(leaf_states.to_vec()))
}

// ✅ Single Poseidon hash of all 7 would-be public inputs (host-side)
fn host_instance_hash(items: [F; 7]) -> F {
    use midnight_circuits::instructions::hash::HashCPU;
    <PoseidonChip<F> as HashCPU<F, F>>::hash(&items)
}

fn map_inserted(mut map: Map, key: F, value: F) -> Map {
    map.insert(&key, &value);
    map
}

fn map_insert_many(map: Map, entries: impl IntoIterator<Item = (F, F)>) -> Map {
    entries
        .into_iter()
        .fold(map, |m, (k, v)| map_inserted(m, k, v))
}

fn apply_tx_effects(commit_map: Map, null_map: Map, items: [F; 7]) -> (Map, Map) {
    let [_tx_root, _x1, _x2, c1, c2, nf1, nf2] = items;
    let commit_map = map_insert_many(commit_map, [(c1, F::ONE), (c2, F::ONE)]);
    let null_map = map_insert_many(null_map, [(nf1, F::ONE), (nf2, F::ONE)]);
    (commit_map, null_map)
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

fn state_to_value_array(state: &rollup_ivc::AggState) -> Result<[Value<F>; 6], AggregationError> {
    let fields = state.to_fields();
    match fields.as_slice() {
        [a, b, c, d, e, f] => Ok([
            a.clone(),
            b.clone(),
            c.clone(),
            d.clone(),
            e.clone(),
            f.clone(),
        ]
        .map(Value::known)),
        _ => Err(AggregationError::UnexpectedAggStateFieldCount {
            expected: 6,
            got: fields.len(),
        }),
    }
}

#[derive(Clone, Debug)]
pub struct AggPublicInputs {
    pub state: rollup_ivc::AggState,
    pub pi_acc: rollup_ivc::AggAccumulator,
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
// Host-side structures + aggregation
////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Debug)]
pub struct TreeNode {
    pub state: rollup_ivc::AggState,
    pub proof: Vec<u8>,
    pub proof_acc: rollup_ivc::AggAccumulator,
    pub pi_acc: rollup_ivc::AggAccumulator,
}

#[derive(Clone, Debug)]
pub struct ClientProof {
    pub state: F,
    pub proof: Vec<u8>,
    pub public_items: [F; 7],
}

#[derive(Clone)]
struct LeafPlan {
    i: usize,
    left_items: [F; 7],
    right_items: [F; 7],
    pre_commitment_map: SendableMap,
    pre_nullifier_map: SendableMap,
    pre_roots_map: SendableMap,
    expected_state: rollup_ivc::AggState,
    left_state: F,
    right_state: F,
    left_proof: Vec<u8>,
    right_proof: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct AggregationResult {
    pub root_state: rollup_ivc::AggState,
    pub left_top: TreeNode,
    pub right_top: TreeNode,
    pub child_vk: (EvaluationDomain<F>, ConstraintSystem<F>, F),
    pub child_vk_name: String,
    pub child_level: usize,
    pub fixed_base_names: Vec<String>,
    pub fixed_bases: BTreeMap<String, C>,
}

struct ValidatedDims {
    num_leaves: usize,
    max_agg_level: usize,
}

fn log_start(pre_commitment_map: &Map, pre_nullifier_map: &Map, client_proofs: &[ClientProof]) {
    println!(
        "[agg] start c_pre(map) = {:?}",
        pre_commitment_map.succinct_repr()
    );
    println!(
        "[agg] start n_pre(map) = {:?}",
        pre_nullifier_map.succinct_repr()
    );

    if let Some(first) = client_proofs.first() {
        let [root_before, ..] = first.public_items;
        println!("[agg] client_proofs[0].root_before = {:?}", root_before);
    }
}

fn validate_dims(
    setup: &setup::AggSetup,
    client_proofs: &[ClientProof],
) -> Result<ValidatedDims, AggregationError> {
    ensure(
        client_proofs.len() == setup.num_leaves,
        AggregationError::ClientProofsLenMismatch {
            expected: setup.num_leaves,
            got: client_proofs.len(),
        },
    )?;
    ensure(
        !client_proofs.is_empty(),
        AggregationError::ClientProofsEmpty,
    )?;
    ensure(
        client_proofs.len().is_power_of_two(),
        AggregationError::ClientProofsNotPowerOfTwo {
            len: client_proofs.len(),
        },
    )?;

    let num_leaves = client_proofs.len();
    let max_level: usize = (num_leaves as u32).trailing_zeros() as usize;

    ensure(
        max_level >= 2,
        AggregationError::NeedAtLeastFourClientProofs { got: num_leaves },
    )?;

    let max_agg_level = max_level - 1;

    ensure(
        max_agg_level == setup.max_agg_level,
        AggregationError::MaxAggLevelMismatch {
            expected: setup.max_agg_level,
            got: max_agg_level,
        },
    )?;

    Ok(ValidatedDims {
        num_leaves,
        max_agg_level,
    })
}

fn check_roots_membership(
    roots_map: &Map,
    leaf: usize,
    side: Side,
    items: [F; 7],
) -> Result<(), AggregationError> {
    let [tx_root, ..] = items;
    ensure(
        roots_map.get(&tx_root) == F::ONE,
        AggregationError::HistoricRootMissing {
            leaf,
            side: side.as_str(),
        },
    )
}

fn plan_leaf_level(
    client_proofs: &[ClientProof],
    pre_commitment_map: Map,
    pre_nullifier_map: Map,
    pre_commitment_roots_map: Map,
    batch_roots_set_root: F,
) -> Result<Vec<LeafPlan>, AggregationError> {
    let init = (
        Vec::with_capacity(client_proofs.len() / 2),
        pre_commitment_map,
        pre_nullifier_map,
    );

    let (plans, _final_commit, _final_null) = client_proofs.chunks_exact(2).enumerate().try_fold(
        init,
        |(mut plans, commit_map, null_map), (i, pair)| {
            let (left, right) = match pair {
                [l, r] => (l, r),
                _ => unreachable!("chunks_exact(2) guarantees pairs"),
            };

            check_roots_membership(&pre_commitment_roots_map, i, Side::Left, left.public_items)?;
            check_roots_membership(
                &pre_commitment_roots_map,
                i,
                Side::Right,
                right.public_items,
            )?;

            let c_pre = commit_map.succinct_repr();
            let n_pre = null_map.succinct_repr();

            let pre_commit_map_for_leaf = commit_map.clone();
            let pre_null_map_for_leaf = null_map.clone();
            let pre_roots_map_for_leaf = pre_commitment_roots_map.clone();

            let (commit_after_left, null_after_left) =
                apply_tx_effects(commit_map, null_map, left.public_items);
            let (commit_after_both, null_after_both) =
                apply_tx_effects(commit_after_left, null_after_left, right.public_items);

            let c_post = commit_after_both.succinct_repr();
            let n_post = null_after_both.succinct_repr();

            let expected_state = rollup_ivc::AggState {
                c_pre,
                c_post,
                n_pre,
                n_post,
                subroot: hash_pair(left.state, right.state),
                commitment_roots_set_root: batch_roots_set_root,
            };

            let plan = LeafPlan {
                i,
                left_items: left.public_items,
                right_items: right.public_items,
                pre_commitment_map: SendableMap(pre_commit_map_for_leaf),
                pre_nullifier_map: SendableMap(pre_null_map_for_leaf),
                pre_roots_map: SendableMap(pre_roots_map_for_leaf),
                expected_state,
                left_state: left.state,
                right_state: right.state,
                left_proof: left.proof.clone(),
                right_proof: right.proof.clone(),
            };

            plans.push(plan);
            Ok((plans, commit_after_both, null_after_both))
        },
    )?;

    Ok(plans)
}

fn build_leaf_nodes(
    setup: &setup::AggSetup,
    leaf_srs: &ParamsKZG<Bls12>,
    leaf_vk: &VerifyingKey<F, KZGCommitmentScheme<E>>,
    leaf_plans: &[LeafPlan],
) -> Result<Vec<TreeNode>, AggregationError> {
    let agg_srs1 = &setup.agg_srs_leaf;
    let agg_srs2 = &setup.agg_srs_internal;

    let combined_fixed_base_names = setup.fixed_base_names.clone();
    let combined_fixed_bases = setup.fixed_bases.clone();
    let trivial_combined = setup.trivial_combined.clone();

    let leaf_level = 1usize;
    let leaf_keys = setup.agg_store.get(leaf_level);
    let leaf_agg_vk_name = leaf_keys.name.clone();

    let leaf_vk_data_cl = setup.leaf_vk_data.clone();
    let leaf_vk_name_string = setup.leaf_vk_name.clone();
    let leaf_pk = leaf_keys.pk.clone();
    let leaf_vk_arc = leaf_keys.vk.clone();
    let leaf_fixed_bases = setup.leaf_fixed_bases.clone();

    leaf_plans
        .par_iter()
        .map(|p| -> Result<TreeNode, AggregationError> {
            let inst_l = host_instance_hash(p.left_items);
            let inst_r = host_instance_hash(p.right_items);

            ensure(
                inst_l == p.left_state,
                AggregationError::InstanceMismatch {
                    leaf: p.i,
                    side: Side::Left.as_str(),
                },
            )?;
            ensure(
                inst_r == p.right_state,
                AggregationError::InstanceMismatch {
                    leaf: p.i,
                    side: Side::Right.as_str(),
                },
            )?;

            let planned_subroot = hash_pair(inst_l, inst_r);
            ensure(
                planned_subroot == p.expected_state.subroot,
                AggregationError::PlannedSubrootMismatch { leaf: p.i },
            )?;

            let circuit = rollup_ivc::LeafAggCircuit {
                child_vk: leaf_vk_data_cl.clone(),
                child_vk_name: leaf_vk_name_string.clone(),

                left_items: Value::known(p.left_items),
                right_items: Value::known(p.right_items),

                pre_commitment_map: Value::known(p.pre_commitment_map.clone_inner()),
                pre_nullifier_map: Value::known(p.pre_nullifier_map.clone_inner()),
                pre_commitment_roots_set_map: Value::known(p.pre_roots_map.clone_inner()),

                left_proof: Value::known(p.left_proof.clone()),
                right_proof: Value::known(p.right_proof.clone()),

                // Naming updated: these are pi-acc witnesses (placeholders for client children).
                left_pi_acc: Value::known(trivial_combined.clone()),
                right_pi_acc: Value::known(trivial_combined.clone()),

                fixed_base_names: combined_fixed_base_names.clone(),
            };

            let proof_acc_left = verify_and_extract_acc(
                leaf_srs,
                leaf_vk,
                &leaf_fixed_bases,
                &p.left_proof,
                &[p.left_state],
            )?;
            let proof_acc_right = verify_and_extract_acc(
                leaf_srs,
                leaf_vk,
                &leaf_fixed_bases,
                &p.right_proof,
                &[p.right_state],
            )?;

            let accumulated_pi = collapse_acc(Accumulator::accumulate(&[
                proof_acc_left,
                trivial_combined.clone(),
                proof_acc_right,
                trivial_combined.clone(),
            ]));

            let public_inputs = AggPublicInputs {
                state: p.expected_state,
                pi_acc: accumulated_pi.clone(),
            };
            let public_inputs_fields = public_inputs.to_fields();

            let start = Instant::now();
            let proof = {
                let mut transcript = CircuitTranscript::<PoseidonState<F>>::init();
                create_proof::<
                    F,
                    KZGCommitmentScheme<E>,
                    CircuitTranscript<PoseidonState<F>>,
                    rollup_ivc::LeafAggCircuit,
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

            println!("proof size (bytes): {}", proof.len());
            println!(
                "Leaf AGG {} ({}) created in {:?}",
                p.i,
                leaf_agg_vk_name,
                start.elapsed()
            );

            ensure(
                accumulated_pi.check(&agg_srs2.s_g2().into(), &combined_fixed_bases),
                AggregationError::PiAccumulatorDidNotCheck("leaf accumulated PI"),
            )?;

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
                pi_acc: accumulated_pi,
            })
        })
        .collect::<Result<Vec<_>, _>>()
}

fn build_internal_levels(
    setup: &setup::AggSetup,
    agg_srs2: &ParamsKZG<Bls12>,
    combined_fixed_base_names: Vec<String>,
    combined_fixed_bases: BTreeMap<String, C>,
    child_level: usize,
    current_level: Vec<TreeNode>,
) -> Result<(usize, Vec<TreeNode>), AggregationError> {
    if current_level.len() <= 2 {
        return Ok((child_level, current_level));
    }

    let parent_level = child_level + 1;

    let parent_keys = setup.agg_store.get(parent_level);
    let parent_vk_name = parent_keys.name.clone();

    println!(
        "\nBuilding AGG level {} ({}) with {} nodes...",
        parent_level,
        parent_vk_name,
        current_level.len() / 2
    );

    let child_keys = setup.agg_store.get(child_level);
    let child_vk_data = child_keys.vk_data.clone();
    let child_vk_name = child_keys.name.clone();

    let parent_pk = parent_keys.pk.clone();
    let parent_vk = parent_keys.vk.clone();

    let next_level: Vec<TreeNode> = current_level
        .par_chunks_exact(2)
        .enumerate()
        .map(|(i, chunk)| -> Result<TreeNode, AggregationError> {
            let (left, right) = match chunk {
                [l, r] => (l, r),
                _ => unreachable!("par_chunks_exact(2) guarantees pairs"),
            };

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

            let state = rollup_ivc::AggState {
                c_pre: left.state.c_pre,
                c_post: right.state.c_post,
                n_pre: left.state.n_pre,
                n_post: right.state.n_post,
                subroot: hash_pair(left.state.subroot, right.state.subroot),
                commitment_roots_set_root: left.state.commitment_roots_set_root,
            };

            let left_child_state = state_to_value_array(&left.state)?;
            let right_child_state = state_to_value_array(&right.state)?;

            let circuit = rollup_ivc::InternalAggCircuit {
                child_vk: child_vk_data.clone(),
                child_vk_name: child_vk_name.clone(),

                left_child_state,
                right_child_state,

                left_proof: Value::known(left.proof.clone()),
                right_proof: Value::known(right.proof.clone()),

                // Naming updated: these are child pi-acc witnesses.
                left_pi_acc: Value::known(left.pi_acc.clone()),
                right_pi_acc: Value::known(right.pi_acc.clone()),

                fixed_base_names: combined_fixed_base_names.clone(),
            };

            let accumulated_pi = collapse_acc(Accumulator::accumulate(&[
                left.proof_acc.clone(),
                left.pi_acc.clone(),
                right.proof_acc.clone(),
                right.pi_acc.clone(),
            ]));

            let public_inputs = AggPublicInputs {
                state,
                pi_acc: accumulated_pi.clone(),
            };
            let public_inputs_fields = public_inputs.to_fields();

            let start = Instant::now();
            let proof = {
                let mut transcript = CircuitTranscript::<PoseidonState<F>>::init();
                create_proof::<
                    F,
                    KZGCommitmentScheme<E>,
                    CircuitTranscript<PoseidonState<F>>,
                    rollup_ivc::InternalAggCircuit,
                >(
                    agg_srs2,
                    parent_pk.as_ref(),
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
                "Level {} node {} ({}) created in {:?}",
                parent_level,
                i,
                parent_vk_name,
                start.elapsed()
            );

            ensure(
                accumulated_pi.check(&agg_srs2.s_g2().into(), &combined_fixed_bases),
                AggregationError::PiAccumulatorDidNotCheck("internal level accumulated PI"),
            )?;

            let proof_acc = verify_and_extract_acc(
                agg_srs2,
                parent_vk.as_ref(),
                &parent_keys.fixed_bases,
                &proof,
                &public_inputs_fields,
            )?;

            Ok(TreeNode {
                state: public_inputs.state,
                proof,
                proof_acc,
                pi_acc: accumulated_pi,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    build_internal_levels(
        setup,
        agg_srs2,
        combined_fixed_base_names,
        combined_fixed_bases,
        parent_level,
        next_level,
    )
}

fn finalize_result(
    setup: &setup::AggSetup,
    client_proofs: &[ClientProof],
    child_level: usize,
    top_pair: Vec<TreeNode>,
) -> Result<AggregationResult, AggregationError> {
    ensure(
        top_pair.len() == 2,
        AggregationError::ExpectedTopPair {
            got: top_pair.len(),
        },
    )?;

    let (left_top, right_top) = match top_pair.as_slice() {
        [l, r] => (l.clone(), r.clone()),
        _ => unreachable!("checked len == 2"),
    };

    ensure(
        left_top.state.c_post == right_top.state.c_pre,
        AggregationError::CommitBoundaryMismatch,
    )?;
    ensure(
        left_top.state.n_post == right_top.state.n_pre,
        AggregationError::NullBoundaryMismatch,
    )?;
    ensure(
        left_top.state.commitment_roots_set_root == right_top.state.commitment_roots_set_root,
        AggregationError::RootsSetRootMismatch,
    )?;

    let root_state = rollup_ivc::AggState {
        c_pre: left_top.state.c_pre,
        c_post: right_top.state.c_post,
        n_pre: left_top.state.n_pre,
        n_post: right_top.state.n_post,
        subroot: hash_pair(left_top.state.subroot, right_top.state.subroot),
        commitment_roots_set_root: left_top.state.commitment_roots_set_root,
    };

    let leaf_states = client_proofs.iter().map(|p| p.state).collect::<Vec<F>>();
    let expected_root = poseidon_tree_root(&leaf_states)?;

    ensure(
        root_state.subroot == expected_root,
        AggregationError::RootPoseidonTreeMismatch,
    )?;

    let child_keys = setup.agg_store.get(child_level);
    let child_vk_tuple = (
        child_keys.vk_data.domain.clone(),
        child_keys.vk_data.cs.clone(),
        child_keys.vk_data.transcript_repr,
    );

    Ok(AggregationResult {
        root_state,
        left_top,
        right_top,
        child_vk: child_vk_tuple,
        child_vk_name: child_keys.name.clone(),
        child_level,
        fixed_base_names: setup.fixed_base_names.clone(),
        fixed_bases: setup.fixed_bases.clone(),
    })
}

/// Aggregation using cached keys (`AggSetup`). No vk/pk computation occurs here.
///
/// NOTE: Added `pre_commitment_roots_map` so leaf circuits can accept “lagging” tx roots
/// (proofs that reference any historic confirmed root).
pub fn try_aggregate_client_proofs_cached(
    setup: &setup::AggSetup,
    leaf_srs: &ParamsKZG<Bls12>,
    leaf_vk: &VerifyingKey<F, KZGCommitmentScheme<E>>,
    client_proofs: &[ClientProof],
    pre_commitment_map: Map,
    pre_nullifier_map: Map,
    pre_commitment_roots_map: Map,
) -> Result<AggregationResult, AggregationError> {
    // logging + validation (kept out of the “tree construction”)
    log_start(&pre_commitment_map, &pre_nullifier_map, client_proofs);
    let dims = validate_dims(setup, client_proofs)?;

    // ---- tree construction (plans -> leaf proofs -> internal proofs -> finalize)
    let batch_roots_set_root = pre_commitment_roots_map.succinct_repr();

    println!("\nCreating {} leaf AGG nodes...", dims.num_leaves / 2);

    let leaf_plans = plan_leaf_level(
        client_proofs,
        pre_commitment_map,
        pre_nullifier_map,
        pre_commitment_roots_map,
        batch_roots_set_root,
    )?;

    let leaf_nodes = build_leaf_nodes(setup, leaf_srs, leaf_vk, &leaf_plans)?;

    let (child_level, top_pair) = build_internal_levels(
        setup,
        &setup.agg_srs_internal,
        setup.fixed_base_names.clone(),
        setup.fixed_bases.clone(),
        1usize,
        leaf_nodes,
    )?;

    finalize_result(setup, client_proofs, child_level, top_pair)
}

/// Backwards-compatible wrapper that preserves the old signature.
pub fn aggregate_client_proofs_cached(
    setup: &setup::AggSetup,
    leaf_srs: &ParamsKZG<Bls12>,
    leaf_vk: &VerifyingKey<F, KZGCommitmentScheme<E>>,
    client_proofs: &[ClientProof],
    pre_commitment_map: Map,
    pre_nullifier_map: Map,
    pre_commitment_roots_map: Map,
) -> AggregationResult {
    try_aggregate_client_proofs_cached(
        setup,
        leaf_srs,
        leaf_vk,
        client_proofs,
        pre_commitment_map,
        pre_nullifier_map,
        pre_commitment_roots_map,
    )
    .expect("aggregation failed")
}
