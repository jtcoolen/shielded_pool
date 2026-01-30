use std::time::Instant;

use ff::Field;
use group::Group;

use midnight_circuits::{
    hash::poseidon::{PoseidonChip, PoseidonState},
    instructions::map::MapCPU,
    types::Instantiable,
};
use midnight_proofs::plonk::create_proof;
use midnight_proofs::{circuit::Value, transcript::Transcript};
use rand::rngs::OsRng;
use rayon::{
    iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator},
    slice::ParallelSlice,
};

use core::array;

use midnight_circuits::verifier::{
    Accumulator, AssignedAccumulator, BlstrsEmulation, SelfEmulation,
};
use midnight_curves::Bls12;
use midnight_proofs::poly::kzg::params::ParamsKZG;
use midnight_proofs::{
    plonk::{ConstraintSystem, VerifyingKey},
    poly::{EvaluationDomain, kzg::KZGCommitmentScheme},
    transcript::CircuitTranscript,
};
use std::collections::BTreeMap;

use crate::{rollup_ivc, setup};

pub type S = BlstrsEmulation;
type F = <S as SelfEmulation>::F;
type C = <S as SelfEmulation>::C;
type E = <S as SelfEmulation>::Engine;
type Map = midnight_circuits::map::cpu::MapMt<F, PoseidonChip<F>>;

#[repr(transparent)]
#[derive(Clone)]
struct SendableMap(Map);
unsafe impl Send for SendableMap {}
unsafe impl Sync for SendableMap {}
impl SendableMap {
    fn clone_inner(&self) -> Map {
        self.0.clone()
    }
}

#[derive(Clone, Debug)]
pub struct AggPublicInputs {
    pub state: rollup_ivc::AggState,
    pub pi_acc: rollup_ivc::AggAccumulator,
}
impl AggPublicInputs {
    pub fn to_fields(&self) -> Vec<F> {
        let mut out = Vec::new();
        out.extend_from_slice(&self.state.to_fields());
        out.extend(AssignedAccumulator::as_public_input(&self.pi_acc));
        out
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

fn poseidon_tree_root(leaf_states: &[F]) -> F {
    use midnight_circuits::instructions::hash::HashCPU;
    assert!(!leaf_states.is_empty(), "Need at least one leaf");
    assert!(
        leaf_states.len().is_power_of_two(),
        "Leaves must be power of two"
    );

    let mut level_states = leaf_states.to_vec();
    while level_states.len() > 1 {
        level_states = level_states
            .chunks(2)
            .map(|pair| <PoseidonChip<F> as HashCPU<F, F>>::hash(&[pair[0], pair[1]]))
            .collect();
    }
    level_states[0]
}

// ✅ Single Poseidon hash of all 7 would-be public inputs (host-side)
fn host_instance_hash(items: [F; 7]) -> F {
    use midnight_circuits::instructions::hash::HashCPU;
    <PoseidonChip<F> as HashCPU<F, F>>::hash(&items)
}

fn verify_and_extract_acc(
    srs: &ParamsKZG<Bls12>,
    vk: &VerifyingKey<F, KZGCommitmentScheme<E>>,
    fixed_bases: &BTreeMap<String, C>,
    proof: &[u8],
    public_inputs: &[F],
) -> Accumulator<S> {
    let mut transcript = CircuitTranscript::<PoseidonState<F>>::init_from_bytes(proof);
    let committed_bases: &[&[C]] = &[&[C::identity()]];
    let instances: &[&[&[F]]] = &[&[public_inputs]];

    let dual_msm = midnight_proofs::plonk::prepare::<
        F,
        KZGCommitmentScheme<E>,
        CircuitTranscript<PoseidonState<F>>,
    >(vk, committed_bases, instances, &mut transcript)
    .expect("Verification (prepare) failed");

    assert!(
        dual_msm.clone().check(&srs.verifier_params()),
        "dual MSM did not check"
    );

    let mut acc: Accumulator<S> = dual_msm.into();
    acc.extract_fixed_bases(fixed_bases);
    acc.collapse();

    assert!(
        acc.check(&srs.s_g2().into(), fixed_bases),
        "Accumulator failed final check against fixed bases"
    );

    acc
}

/// Aggregation using cached keys (`AggSetup`). No vk/pk computation occurs here.
///
/// NOTE: Added `pre_commitment_roots_map` so leaf circuits can accept “lagging” tx roots
/// (proofs that reference any historic confirmed root).
pub fn aggregate_client_proofs_cached(
    setup: &setup::AggSetup,
    leaf_srs: &ParamsKZG<Bls12>,
    leaf_vk: &VerifyingKey<F, KZGCommitmentScheme<E>>,
    client_proofs: &[ClientProof],
    pre_commitment_map: Map,
    pre_nullifier_map: Map,
    pre_commitment_roots_map: Map,
) -> AggregationResult {
    use midnight_circuits::instructions::hash::HashCPU;

    println!(
        "[agg] start c_pre(map) = {:?}",
        pre_commitment_map.succinct_repr()
    );
    println!(
        "[agg] start n_pre(map) = {:?}",
        pre_nullifier_map.succinct_repr()
    );
    println!(
        "[agg] client_proofs[0].root_before = {:?}",
        client_proofs[0].public_items[0]
    );

    assert_eq!(
        client_proofs.len(),
        setup.num_leaves,
        "client_proofs len must match cached setup (expected {}, got {})",
        setup.num_leaves,
        client_proofs.len()
    );
    assert!(!client_proofs.is_empty(), "Need at least one client proof");
    assert!(
        client_proofs.len().is_power_of_two(),
        "Client proofs must be power of two"
    );

    let num_leaves = client_proofs.len();
    let max_level: usize = (num_leaves as u32).trailing_zeros() as usize;
    assert!(
        max_level >= 2,
        "Merged final agg requires at least 4 client proofs"
    );
    let max_agg_level: usize = max_level - 1;
    assert_eq!(
        max_agg_level, setup.max_agg_level,
        "max_agg_level mismatch with cached setup"
    );

    let agg_srs1 = &setup.agg_srs_leaf;
    let agg_srs2 = &setup.agg_srs_internal;

    let combined_fixed_base_names = setup.fixed_base_names.clone();
    let combined_fixed_bases = setup.fixed_bases.clone();
    let trivial_combined = setup.trivial_combined.clone();

    // ---- FIX (Issue 1): bind a single roots-set root across whole agg tree (host side)
    let batch_roots_set_root = pre_commitment_roots_map.succinct_repr();

    println!("\nCreating {} leaf AGG nodes...", num_leaves / 2);

    let leaf_level = 1usize;
    let leaf_keys = setup.agg_store.get(leaf_level);
    let leaf_agg_vk_name = leaf_keys.name.clone();

    let mut rolling_commit_map = pre_commitment_map.clone();
    let mut rolling_null_map = pre_nullifier_map.clone();

    let mut leaf_plans: Vec<LeafPlan> = Vec::with_capacity(num_leaves / 2);

    for i in 0..num_leaves / 2 {
        let left = &client_proofs[i * 2];
        let right = &client_proofs[i * 2 + 1];

        let c_pre = rolling_commit_map.succinct_repr();
        let n_pre = rolling_null_map.succinct_repr();

        // NEW (host-side sanity): both tx roots must be in the historic roots set
        assert_eq!(
            pre_commitment_roots_map.get(&left.public_items[0]),
            F::ONE,
            "leaf {} left tx root not in historic roots set",
            i
        );
        assert_eq!(
            pre_commitment_roots_map.get(&right.public_items[0]),
            F::ONE,
            "leaf {} right tx root not in historic roots set",
            i
        );

        let pre_commit_map_for_leaf = rolling_commit_map.clone();
        let pre_null_map_for_leaf = rolling_null_map.clone();
        let pre_roots_map_for_leaf = pre_commitment_roots_map.clone();

        // Apply BOTH tx’s effects to the rolling rollup state (independent of tx-root in proofs)
        rolling_commit_map.insert(&left.public_items[3], &F::ONE);
        rolling_commit_map.insert(&left.public_items[4], &F::ONE);

        for nf in [left.public_items[5], left.public_items[6]] {
            let _old = rolling_null_map.get(&nf);
            rolling_null_map.insert(&nf, &F::ONE);
        }

        rolling_commit_map.insert(&right.public_items[3], &F::ONE);
        rolling_commit_map.insert(&right.public_items[4], &F::ONE);

        for nf in [right.public_items[5], right.public_items[6]] {
            let _old = rolling_null_map.get(&nf);
            rolling_null_map.insert(&nf, &F::ONE);
        }

        let c_post = rolling_commit_map.succinct_repr();
        let n_post = rolling_null_map.succinct_repr();

        let subroot = <PoseidonChip<F> as HashCPU<F, F>>::hash(&[left.state, right.state]);

        let expected_state = rollup_ivc::AggState {
            c_pre,
            c_post,
            n_pre,
            n_post,
            subroot,
            roots_set_root: batch_roots_set_root, // ---- FIX (Issue 1)
        };

        leaf_plans.push(LeafPlan {
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
        });
    }

    leaf_plans.par_iter().for_each(|p| {
        let inst_l = host_instance_hash(p.left_items);
        let inst_r = host_instance_hash(p.right_items);
        assert_eq!(
            inst_l, p.left_state,
            "left client instance mismatch (leaf {})",
            p.i
        );
        assert_eq!(
            inst_r, p.right_state,
            "right client instance mismatch (leaf {})",
            p.i
        );

        let subroot = <PoseidonChip<F> as HashCPU<F, F>>::hash(&[inst_l, inst_r]);
        assert_eq!(
            subroot, p.expected_state.subroot,
            "leaf {} planned subroot mismatch",
            p.i
        );
    });

    let leaf_vk_data_cl = setup.leaf_vk_data.clone();
    let leaf_vk_name_string = setup.leaf_vk_name.clone();
    let leaf_pk = leaf_keys.pk.clone();
    let leaf_vk_arc = leaf_keys.vk.clone();

    let leaf_fixed_bases = setup.leaf_fixed_bases.clone();

    let mut current_level: Vec<TreeNode> = leaf_plans
        .par_iter()
        .map(|p| {
            let state = p.expected_state;

            let circuit = rollup_ivc::LeafAggCircuit {
                child_vk: leaf_vk_data_cl.clone(),
                child_vk_name: leaf_vk_name_string.clone(),
                left_child_state: array::from_fn(|_| Value::unknown()),
                right_child_state: array::from_fn(|_| Value::unknown()),
                left_items: Value::known(p.left_items),
                right_items: Value::known(p.right_items),
                pre_commitment_map: Value::known(p.pre_commitment_map.clone_inner()),
                pre_nullifier_map: Value::known(p.pre_nullifier_map.clone_inner()),
                pre_commitment_roots_map: Value::known(p.pre_roots_map.clone_inner()),
                left_proof: Value::known(p.left_proof.clone()),
                right_proof: Value::known(p.right_proof.clone()),
                left_acc: Value::known(trivial_combined.clone()),
                right_acc: Value::known(trivial_combined.clone()),
                fixed_base_names: combined_fixed_base_names.clone(),
                is_leaf: true,
            };

            let proof_acc_left = verify_and_extract_acc(
                leaf_srs,
                leaf_vk,
                &leaf_fixed_bases,
                &p.left_proof,
                &[p.left_state],
            );
            let proof_acc_right = verify_and_extract_acc(
                leaf_srs,
                leaf_vk,
                &leaf_fixed_bases,
                &p.right_proof,
                &[p.right_state],
            );

            let mut accumulated_pi = Accumulator::accumulate(&[
                proof_acc_left,
                trivial_combined.clone(),
                proof_acc_right,
                trivial_combined.clone(),
            ]);
            accumulated_pi.collapse();

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
                .expect("Leaf AGG proof failed");
                transcript.finalize()
            };

            println!("proof size (bytes): {}", proof.len());
            println!(
                "Leaf AGG {} ({}) created in {:?}",
                p.i,
                leaf_agg_vk_name,
                start.elapsed()
            );

            assert!(
                accumulated_pi.check(&agg_srs2.s_g2().into(), &combined_fixed_bases),
                "Leaf node {}: accumulated PI accumulator did not check",
                p.i
            );

            let proof_acc = verify_and_extract_acc(
                agg_srs1,
                leaf_vk_arc.as_ref(),
                &leaf_keys.fixed_bases,
                &proof,
                &public_inputs_fields,
            );

            TreeNode {
                state,
                proof,
                proof_acc,
                pi_acc: accumulated_pi,
            }
        })
        .collect();

    let mut child_level: usize = 1;
    while current_level.len() > 2 {
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
            .par_chunks(2)
            .enumerate()
            .map(|(i, pair)| {
                let left = &pair[0];
                let right = &pair[1];

                assert_eq!(
                    left.state.c_post, right.state.c_pre,
                    "commit boundary mismatch"
                );
                assert_eq!(
                    left.state.n_post, right.state.n_pre,
                    "null boundary mismatch"
                );
                // ---- FIX (Issue 1): roots-set root must match across children
                assert_eq!(
                    left.state.roots_set_root, right.state.roots_set_root,
                    "roots_set_root mismatch"
                );

                let state = rollup_ivc::AggState {
                    c_pre: left.state.c_pre,
                    c_post: right.state.c_post,
                    n_pre: left.state.n_pre,
                    n_post: right.state.n_post,
                    subroot: <PoseidonChip<F> as midnight_circuits::instructions::hash::HashCPU<
                        F,
                        F,
                    >>::hash(&[
                        left.state.subroot,
                        right.state.subroot,
                    ]),
                    roots_set_root: left.state.roots_set_root,
                };

                let l_fields = left.state.to_fields();
                let r_fields = right.state.to_fields();

                let circuit = rollup_ivc::InternalAggCircuit {
                    child_vk: child_vk_data.clone(),
                    child_vk_name: child_vk_name.clone(),
                    left_child_state: array::from_fn(|j| Value::known(l_fields[j])),
                    right_child_state: array::from_fn(|j| Value::known(r_fields[j])),
                    left_items: Value::unknown(),
                    right_items: Value::unknown(),
                    pre_commitment_map: Value::unknown(),
                    pre_nullifier_map: Value::unknown(),
                    pre_commitment_roots_map: Value::unknown(),
                    left_proof: Value::known(left.proof.clone()),
                    right_proof: Value::known(right.proof.clone()),
                    left_acc: Value::known(left.pi_acc.clone()),
                    right_acc: Value::known(right.pi_acc.clone()),
                    fixed_base_names: combined_fixed_base_names.clone(),
                    is_leaf: false,
                };

                let mut accumulated_pi = Accumulator::accumulate(&[
                    left.proof_acc.clone(),
                    left.pi_acc.clone(),
                    right.proof_acc.clone(),
                    right.pi_acc.clone(),
                ]);
                accumulated_pi.collapse();

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
                    .expect("Internal AGG proof failed");
                    transcript.finalize()
                };

                println!(
                    "Level {} node {} ({}) created in {:?}",
                    parent_level,
                    i,
                    parent_vk_name,
                    start.elapsed()
                );

                assert!(
                    accumulated_pi.check(&agg_srs2.s_g2().into(), &combined_fixed_bases),
                    "Level {parent_level} node {i}: accumulated PI accumulator did not check"
                );

                let proof_acc = verify_and_extract_acc(
                    agg_srs2,
                    parent_vk.as_ref(),
                    &parent_keys.fixed_bases,
                    &proof,
                    &public_inputs_fields,
                );

                TreeNode {
                    state,
                    proof,
                    proof_acc,
                    pi_acc: accumulated_pi,
                }
            })
            .collect();

        current_level = next_level;
        child_level = parent_level;
    }

    assert_eq!(current_level.len(), 2, "expected to stop at top pair");

    let left_top = current_level[0].clone();
    let right_top = current_level[1].clone();

    assert_eq!(
        left_top.state.c_post, right_top.state.c_pre,
        "top commit boundary mismatch"
    );
    assert_eq!(
        left_top.state.n_post, right_top.state.n_pre,
        "top null boundary mismatch"
    );
    // ---- FIX (Issue 1): roots-set root must match at the top as well
    assert_eq!(
        left_top.state.roots_set_root, right_top.state.roots_set_root,
        "top roots_set_root mismatch"
    );

    let root_subroot = <PoseidonChip<F> as HashCPU<F, F>>::hash(&[
        left_top.state.subroot,
        right_top.state.subroot,
    ]);

    let root_state = rollup_ivc::AggState {
        c_pre: left_top.state.c_pre,
        c_post: right_top.state.c_post,
        n_pre: left_top.state.n_pre,
        n_post: right_top.state.n_post,
        subroot: root_subroot,
        roots_set_root: left_top.state.roots_set_root,
    };

    let leaf_states: Vec<F> = client_proofs.iter().map(|p| p.state).collect();
    let expected_root = poseidon_tree_root(&leaf_states);
    assert_eq!(
        root_state.subroot, expected_root,
        "Root subroot mismatch with recomputed Poseidon tree root"
    );

    let child_keys = setup.agg_store.get(child_level);
    let child_vk_tuple = (
        child_keys.vk_data.domain.clone(),
        child_keys.vk_data.cs.clone(),
        child_keys.vk_data.transcript_repr,
    );

    AggregationResult {
        root_state,
        left_top,
        right_top,
        child_vk: child_vk_tuple,
        child_vk_name: child_keys.name.clone(),
        child_level,
        fixed_base_names: combined_fixed_base_names,
        fixed_bases: combined_fixed_bases,
    }
}
