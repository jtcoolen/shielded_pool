use std::time::Instant;

use ff::Field;
use group::Group;

use midnight_circuits::{
    compact_std_lib::{self, cost_model},
    hash::poseidon::{PoseidonChip, PoseidonState},
    instructions::map::MapCPU,
    map::cpu::MapMt,
    types::{AssignedNativePoint, Instantiable},
};
use midnight_curves::{Fr as JubjubScalar, JubjubExtended as Jubjub, JubjubSubgroup};
use midnight_proofs::plonk::{create_proof, keygen_pk, keygen_vk_with_k, prepare};
use midnight_proofs::{circuit::Value, transcript::Transcript};
use rand::{Rng, SeedableRng, rngs::OsRng};
use rand_chacha::ChaCha8Rng;
use rayon::{
    iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator},
    slice::ParallelSlice,
};

use core::array;

use midnight_circuits::{
    ecc::foreign::ForeignEccChip,
    field::{NativeGadget, decomposition::chip::P2RDecompositionChip, native::NativeChip},
    types::AssignedForeignPoint,
    verifier::{Accumulator, AssignedAccumulator, BlstrsEmulation, SelfEmulation},
};
use midnight_curves::Bls12;
use midnight_proofs::poly::kzg::params::ParamsKZG;
use midnight_proofs::{
    plonk::{ConstraintSystem, VerifyingKey},
    poly::{EvaluationDomain, kzg::KZGCommitmentScheme},
    transcript::CircuitTranscript,
};
use std::collections::BTreeMap;

mod keccak;
mod rollup_ivc;
mod setup;
mod srs;
mod transfer;

pub type S = BlstrsEmulation;
type F = <S as SelfEmulation>::F;
type C = <S as SelfEmulation>::C;
type E = <S as SelfEmulation>::Engine;
type NG = NativeGadget<F, P2RDecompositionChip<F>, NativeChip<F>>;
type Map = midnight_circuits::map::cpu::MapMt<F, PoseidonChip<F>>;

const BATCH_SIZE: usize = 4;

// NEW: probability that a client proof is generated against an older confirmed root
const LAG_TX_PROB: f64 = 0.35;

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

const K_INTERNAL: u32 = 19;

pub const AGG_K: u32 = K_INTERNAL;

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

pub type CurveChip = ForeignEccChip<F, C, C, NG, NG>;
pub type MapGadget = midnight_circuits::map::map_gadget::MapGadget<F, NG, PoseidonChip<F>>;
pub type IdPoint = AssignedForeignPoint<
    midnight_curves::Fq,
    midnight_curves::G1Projective,
    midnight_curves::G1Projective,
>;

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

////////////////////////////////////////////////////////////////////////////////
// Demo main (unchanged)
////////////////////////////////////////////////////////////////////////////////

fn main() {
    const LEAF_VK_NAME: &str = "spend2output2_vk";

    const K: u32 = 14;
    const NUM_ACCOUNTS: usize = 4;
    const NUM_SEED_DEPOSITS_PER_ACCOUNT: usize = 50;
    const NUM_TRANSFERS: usize = 120;

    let srs = srs::filecoin_srs_agg(K).unwrap();
    let relation = transfer::Spend2Output2;
    let vk = compact_std_lib::setup_vk(&srs, &relation);
    let pk = compact_std_lib::setup_pk(&relation, &vk);

    // ✅ Cache AGG keys once (for the fixed batch size).
    let agg_setup = setup::prepare_agg_setup(&srs, vk.vk(), LEAF_VK_NAME, K, BATCH_SIZE);

    // ✅ Cache FINAL aggregation vk/pk once (depends only on cached agg_setup for this batch size).
    let final_agg_srs = srs::filecoin_srs_agg(AGG_K).unwrap();
    let default_final_circuit = rollup_ivc::FinalAggCircuit {
        child_vk: agg_setup.child_vk().clone(),
        child_vk_name: agg_setup.child_vk_name().to_string(),
        child_level: F::from(agg_setup.child_level() as u64),
        left_proof: Value::unknown(),
        right_proof: Value::unknown(),
        left_pi_acc: Value::unknown(),
        right_pi_acc: Value::unknown(),
        fixed_base_names: agg_setup.fixed_base_names().to_vec(),
        left_child_state: Value::unknown(),
        right_child_state: Value::unknown(),
        agg_state: Value::unknown(),
        pre_commitment_roots_map: Value::unknown(),
        post_commitment_roots_root: Value::unknown(),
    };
    let final_vk = keygen_vk_with_k(&final_agg_srs, &default_final_circuit, AGG_K)
        .expect("final vk gen should not fail");
    let final_pk =
        keygen_pk(final_vk.clone(), &default_final_circuit).expect("final pk gen should not fail");

    let mut rng = ChaCha8Rng::from_entropy();
    let asset_id = F::random(&mut rng);

    let mut commitment_map = MapMt::<F, PoseidonChip<F>>::new(&F::ZERO);
    let mut nullifier_map = MapMt::<F, PoseidonChip<F>>::new(&F::ZERO);

    // Confirmed root history (one per “rollup commit”)
    let mut commitment_root_history: Vec<F> = Vec::new();
    // Confirmed map snapshots (for generating lagging proofs)
    let mut commitment_map_history: Vec<MapMt<F, PoseidonChip<F>>> = Vec::new();

    let mut commitment_roots_set = MapMt::<F, PoseidonChip<F>>::new(&F::ZERO);

    let mut accounts: Vec<transfer::Account> = (0..NUM_ACCOUNTS)
        .map(|i| {
            let sk = JubjubScalar::random(&mut OsRng);
            let pk_point = JubjubSubgroup::generator() * sk;
            let fields = AssignedNativePoint::<Jubjub>::as_public_input(&pk_point);
            transfer::Account {
                id: i,
                sk,
                pk_point,
                pk_x: fields[0],
                pk_y: fields[1],
                wallet: vec![],
            }
        })
        .collect();

    for acc in &mut accounts {
        for _ in 0..NUM_SEED_DEPOSITS_PER_ACCOUNT {
            let hi: u128 = rng.r#gen::<u128>() >> (128 - transfer::AMOUNT_GEN_BITS);
            let amt: u128 = hi;
            let utxo = transfer::Utxo {
                asset_id,
                amount: amt,
                randomness: F::random(&mut rng),
            };
            let commit = transfer::host_commit(
                utxo.asset_id,
                utxo.amount,
                acc.pk_x,
                acc.pk_y,
                utxo.randomness,
            );

            commitment_map.insert(&commit, &F::ONE);

            acc.wallet.push(transfer::Note {
                utxo,
                commit,
                spent: false,
                confirmed_at_root_idx: 0,
            });
        }
    }

    let genesis_root = commitment_map.succinct_repr();
    println!("Initial commitment root: {:?}", genesis_root);

    commitment_root_history.push(genesis_root);
    commitment_map_history.push(commitment_map.clone());
    commitment_roots_set.insert(&genesis_root, &F::ONE);

    let choose_sender = |rng: &mut ChaCha8Rng,
                         accs: &mut [transfer::Account],
                         latest_confirmed_root_idx: usize|
     -> Option<usize> {
        let viable: Vec<usize> = accs
            .iter()
            .enumerate()
            .filter(|(_, a)| {
                a.wallet
                    .iter()
                    .filter(|n| !n.spent && n.confirmed_at_root_idx <= latest_confirmed_root_idx)
                    .count()
                    >= 2
            })
            .map(|(i, _)| i)
            .collect();
        if viable.is_empty() {
            None
        } else {
            Some(viable[rng.gen_range(0..viable.len())])
        }
    };

    let mut total_transfers_done = 0usize;
    let mut batch_idx = 0usize;

    'outer: loop {
        if total_transfers_done >= NUM_TRANSFERS {
            break;
        }

        let mut shadow_accounts = accounts.clone();
        let mut shadow_nullifier_map = nullifier_map.clone();
        let mut shadow_commitment_map = commitment_map.clone();

        let pre_nullifier_map_for_batch = shadow_nullifier_map.clone();
        let pre_commitment_map_for_batch = shadow_commitment_map.clone();
        let pre_commitment_roots_map_for_batch = commitment_roots_set.clone();

        let latest_confirmed_root_idx = commitment_root_history.len() - 1;

        let mut client_proofs: Vec<ClientProof> = Vec::new();

        println!(
            "\n=== Starting batch {} from commitment root {:?} ===",
            batch_idx,
            shadow_commitment_map.succinct_repr()
        );

        let mut batch_failed = false;

        for _ in 0..BATCH_SIZE {
            if total_transfers_done >= NUM_TRANSFERS {
                break;
            }

            let sender_idx = match choose_sender(
                &mut rng,
                &mut shadow_accounts,
                latest_confirmed_root_idx,
            ) {
                Some(i) => i,
                None => {
                    println!(
                        "[batch {}] no account has two spendable confirmed notes; stopping batching.",
                        batch_idx
                    );
                    batch_failed = true;
                    break;
                }
            };

            let (i_old1, i_old2) = {
                let unspent: Vec<usize> = shadow_accounts[sender_idx]
                    .wallet
                    .iter()
                    .enumerate()
                    .filter(|(_, n)| {
                        !n.spent && n.confirmed_at_root_idx <= latest_confirmed_root_idx
                    })
                    .map(|(i, _)| i)
                    .collect();
                let a = unspent[rng.gen_range(0..unspent.len())];
                let mut b = unspent[rng.gen_range(0..unspent.len())];
                while b == a {
                    b = unspent[rng.gen_range(0..unspent.len())];
                }
                (a, b)
            };

            let r1 = rng.gen_range(0..NUM_ACCOUNTS);
            let r2 = rng.gen_range(0..NUM_ACCOUNTS);

            let sender = shadow_accounts[sender_idx].clone();
            let old1 = shadow_accounts[sender_idx].wallet[i_old1].clone();
            let old2 = shadow_accounts[sender_idx].wallet[i_old2].clone();

            // NEW: choose a (possibly lagging) confirmed root index to prove against
            let min_root_idx_for_inputs =
                old1.confirmed_at_root_idx.max(old2.confirmed_at_root_idx);

            let root_idx_for_proof = if min_root_idx_for_inputs < latest_confirmed_root_idx
                && rng.gen_bool(LAG_TX_PROB)
            {
                // force lag: pick strictly older than latest when possible
                rng.gen_range(min_root_idx_for_inputs..=latest_confirmed_root_idx - 1)
            } else {
                latest_confirmed_root_idx
            };

            let historic_commit_map = commitment_map_history[root_idx_for_proof].clone();
            let root_before = commitment_root_history[root_idx_for_proof];
            debug_assert_eq!(historic_commit_map.succinct_repr(), root_before);

            if root_idx_for_proof != latest_confirmed_root_idx {
                println!(
                    "[batch {}, tx {}] 🕒 lagging proof root: idx {} (latest {}), root {:?}",
                    batch_idx,
                    total_transfers_done,
                    root_idx_for_proof,
                    latest_confirmed_root_idx,
                    root_before
                );
            }

            let total: u128 = old1.utxo.amount + old2.utxo.amount;
            let out1_amt: u128 = if total == 0 {
                0
            } else {
                rng.gen_range(0..=total)
            };
            let out2_amt: u128 = total - out1_amt;

            let new1 = transfer::Utxo {
                asset_id,
                amount: out1_amt,
                randomness: F::random(&mut rng),
            };
            let new2 = transfer::Utxo {
                asset_id,
                amount: out2_amt,
                randomness: F::random(&mut rng),
            };

            let new1_commit = transfer::host_commit(
                new1.asset_id,
                new1.amount,
                shadow_accounts[r1].pk_x,
                shadow_accounts[r1].pk_y,
                new1.randomness,
            );
            let new2_commit = transfer::host_commit(
                new2.asset_id,
                new2.amount,
                shadow_accounts[r2].pk_x,
                shadow_accounts[r2].pk_y,
                new2.randomness,
            );

            let nf1 = transfer::host_nullify(old1.commit, sender.pk_x, sender.pk_y);
            let nf2 = transfer::host_nullify(old2.commit, sender.pk_x, sender.pk_y);

            shadow_nullifier_map.insert(&nf1, &F::ONE);
            shadow_nullifier_map.insert(&nf2, &F::ONE);

            let alpha = JubjubScalar::random(&mut OsRng);
            let blind_point = JubjubSubgroup::generator() * alpha;
            let pk_blinded_point = sender.pk_point + blind_point;
            let pkb_fields = AssignedNativePoint::<Jubjub>::as_public_input(&pk_blinded_point);
            let pk_bx = pkb_fields[0];
            let pk_by = pkb_fields[1];

            let public_items = [
                root_before,
                pk_bx,
                pk_by,
                new1_commit,
                new2_commit,
                nf1,
                nf2,
            ];
            let instance: F = host_instance_hash(public_items);

            let witness = (
                historic_commit_map,
                sender.sk,
                F::from_bytes_le(&alpha.to_bytes()).unwrap(),
                old1.utxo.clone(),
                old2.utxo.clone(),
                new1.clone(),
                new2.clone(),
                shadow_accounts[r1].pk_point,
                shadow_accounts[r2].pk_point,
            );

            let now = Instant::now();
            let proof = compact_std_lib::prove::<transfer::Spend2Output2, PoseidonState<F>>(
                &srs, &pk, &relation, &instance, witness, OsRng,
            )
            .expect("Proof generation failed");
            println!(
                "[batch {}, tx {}] proof gen: {:?}",
                batch_idx,
                total_transfers_done,
                now.elapsed()
            );

            let stats = cost_model(&transfer::Spend2Output2);
            println!("client circuit stats: {:?}", stats);

            client_proofs.push(ClientProof {
                state: instance,
                proof: proof.clone(),
                public_items,
            });

            // Apply to shadow rollup state (current commitment/nullifier sets)
            shadow_commitment_map.insert(&new1_commit, &F::ONE);
            shadow_commitment_map.insert(&new2_commit, &F::ONE);

            shadow_accounts[sender_idx].wallet[i_old1].spent = true;
            shadow_accounts[sender_idx].wallet[i_old2].spent = true;

            let confirm_at_idx = commitment_root_history.len();
            shadow_accounts[r1].wallet.push(transfer::Note {
                utxo: new1,
                commit: new1_commit,
                spent: false,
                confirmed_at_root_idx: confirm_at_idx,
            });
            shadow_accounts[r2].wallet.push(transfer::Note {
                utxo: new2,
                commit: new2_commit,
                spent: false,
                confirmed_at_root_idx: confirm_at_idx,
            });

            println!(
                "[batch {}, tx {}] shadow commitment root updated: {:?}",
                batch_idx,
                total_transfers_done,
                shadow_commitment_map.succinct_repr()
            );

            total_transfers_done += 1;
        }

        if batch_failed || client_proofs.is_empty() {
            break 'outer;
        }

        assert_eq!(
            client_proofs.len(),
            BATCH_SIZE,
            "Batch not completely filled"
        );
        assert!(
            client_proofs.len().is_power_of_two(),
            "Batch size must be a power of two"
        );

        let now = Instant::now();
        let agg_result = aggregate_client_proofs_cached(
            &agg_setup,
            &srs,
            vk.vk(),
            &client_proofs,
            pre_commitment_map_for_batch.clone(),
            pre_nullifier_map_for_batch.clone(),
            pre_commitment_roots_map_for_batch.clone(),
        );
        println!(
            "Batch {} aggregated (up to top pair) in {:?}",
            batch_idx,
            now.elapsed()
        );
        println!(
            "Batch {} computed subroot (host): {:?}",
            batch_idx, agg_result.root_state.subroot
        );

        let pre_roots_set_root = pre_commitment_roots_map_for_batch.succinct_repr();

        assert_eq!(
            pre_commitment_roots_map_for_batch.get(&agg_result.root_state.c_post),
            F::ZERO,
            "Replay guard: c_post already present in roots set"
        );
        let mut shadow_commitment_roots_set = pre_commitment_roots_map_for_batch.clone();
        shadow_commitment_roots_set.insert(&agg_result.root_state.c_post, &F::ONE);
        let post_roots_set_root = shadow_commitment_roots_set.succinct_repr();

        {
            use midnight_proofs::poly::kzg::KZGCommitmentScheme;
            use midnight_proofs::transcript::CircuitTranscript;

            let mut final_acc: rollup_ivc::AggAccumulator =
                rollup_ivc::AggAccumulator::accumulate(&[
                    agg_result.left_top.proof_acc.clone(),
                    agg_result.left_top.pi_acc.clone(),
                    agg_result.right_top.proof_acc.clone(),
                    agg_result.right_top.pi_acc.clone(),
                ]);
            final_acc.collapse();
            let final_acc_pi = rollup_ivc::accumulator_as_public_input(&final_acc);

            let final_circuit = rollup_ivc::FinalAggCircuit {
                child_vk: agg_result.child_vk.clone(),
                child_vk_name: agg_result.child_vk_name.clone(),
                child_level: F::from(agg_result.child_level as u64),
                left_proof: Value::known(agg_result.left_top.proof.clone()),
                right_proof: Value::known(agg_result.right_top.proof.clone()),
                left_pi_acc: Value::known(agg_result.left_top.pi_acc.clone()),
                right_pi_acc: Value::known(agg_result.right_top.pi_acc.clone()),
                fixed_base_names: agg_result.fixed_base_names.clone(),
                left_child_state: Value::known(agg_result.left_top.state),
                right_child_state: Value::known(agg_result.right_top.state),
                agg_state: Value::known(agg_result.root_state),
                pre_commitment_roots_map: Value::known(pre_commitment_roots_map_for_batch.clone()),
                post_commitment_roots_root: Value::known(post_roots_set_root),
            };

            let mut final_public_inputs: Vec<F> = vec![
                agg_result.root_state.c_pre,
                agg_result.root_state.c_post,
                agg_result.root_state.n_pre,
                agg_result.root_state.n_post,
                agg_result.root_state.subroot,
                pre_roots_set_root,
                post_roots_set_root,
            ];
            final_public_inputs.extend(final_acc_pi.clone());

            // ✅ Use cached final_pk/final_vk/final_agg_srs (no per-batch keygen).
            let final_proof_bytes = {
                let mut transcript = CircuitTranscript::<keccak::KeccakTranscript>::init();
                create_proof::<
                    F,
                    KZGCommitmentScheme<midnight_curves::Bls12>,
                    CircuitTranscript<keccak::KeccakTranscript>,
                    rollup_ivc::FinalAggCircuit,
                >(
                    &final_agg_srs,
                    &final_pk,
                    &[final_circuit],
                    1,
                    &[&[&[], &final_public_inputs]],
                    OsRng,
                    &mut transcript,
                )
                .expect("Final aggregation proof generation should not fail");
                transcript.finalize()
            };

            println!("final proof size (bytes): {}", final_proof_bytes.len());

            let mut transcript =
                CircuitTranscript::<keccak::KeccakTranscript>::init_from_bytes(&final_proof_bytes);
            let committed_bases: &[&[midnight_curves::G1Projective]] =
                &[&[midnight_curves::G1Projective::identity()]];
            let instances: &[&[&[F]]] = &[&[&final_public_inputs]];

            let dual_msm = prepare::<
                F,
                KZGCommitmentScheme<midnight_curves::Bls12>,
                CircuitTranscript<keccak::KeccakTranscript>,
            >(&final_vk, committed_bases, instances, &mut transcript)
            .expect("Final aggregation verification preparation failed");

            assert!(
                dual_msm.check(&final_agg_srs.verifier_params()),
                "Final proof must verify"
            );

            assert!(
                final_acc.check(&final_agg_srs.s_g2().into(), &agg_result.fixed_bases),
                "Final aggregation accumulator must verify"
            );

            println!(
                "\n✅ Final aggregation (MERGED) proof for batch {} verified.\n\
                 Batch subroot (host): {:?}\n\
                    Commitment-set transition: {:?} -> {:?}\n\
                    Nullifier-set transition: {:?} -> {:?}\n\
                    Historic-roots-set transition: {:?} -> {:?}\n\
                    Final accumulator PI length: {} field elements",
                batch_idx,
                agg_result.root_state.subroot,
                agg_result.root_state.c_pre,
                agg_result.root_state.c_post,
                agg_result.root_state.n_pre,
                agg_result.root_state.n_post,
                pre_roots_set_root,
                post_roots_set_root,
                final_acc_pi.len()
            );
        }

        // Commit batch to “chain state”
        accounts = shadow_accounts;
        nullifier_map = shadow_nullifier_map;
        commitment_map = shadow_commitment_map;

        commitment_roots_set = shadow_commitment_roots_set;
        commitment_root_history.push(commitment_map.succinct_repr());
        commitment_map_history.push(commitment_map.clone());

        println!(
            "After batch {} committed commitment root: {:?}",
            batch_idx,
            commitment_map.succinct_repr()
        );

        // NEW: Demonstrate replay protection:
        println!("REPLAY attempt:");
        println!("  c_pre  = {:?}", agg_result.root_state.c_pre);
        println!("  c_post = {:?}", agg_result.root_state.c_post);
        println!("  n_pre  = {:?}", agg_result.root_state.n_pre);
        println!("  n_post = {:?}", agg_result.root_state.n_post);

        println!(
            "  roots_set has c_pre?  {:?}",
            commitment_roots_set.get(&agg_result.root_state.c_pre)
        );
        println!(
            "  roots_set has c_post? {:?}",
            commitment_roots_set.get(&agg_result.root_state.c_post)
        );

        println!(
            "  nullifier_map root == n_post? {}",
            nullifier_map.succinct_repr() == agg_result.root_state.n_post
        );

        let replay_commit_map = commitment_map.clone(); // current POST state
        let replay_null_map = nullifier_map.clone(); // current POST state
        let replay_roots_set_map = commitment_roots_set.clone(); // head AFTER applying batch

        let replay = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = aggregate_client_proofs_cached(
                &agg_setup,
                &srs,
                vk.vk(),
                &client_proofs,
                replay_commit_map.clone(),
                replay_null_map.clone(),
                replay_roots_set_map.clone(),
            );
        }));
        match replay {
            Ok(_) => println!("❌ Replay unexpectedly succeeded (BUG)"),
            Err(_) => println!(
                "✅ Replay correctly rejected (nullifiers already spent / state already advanced)"
            ),
        }

        batch_idx += 1;
    }

    println!(
        "\nFinal commitment root: {:?}",
        commitment_map.succinct_repr()
    );

    for acc in &accounts {
        let bal: u128 = acc
            .wallet
            .iter()
            .filter(|n| !n.spent)
            .fold(0u128, |s, n| s.saturating_add(n.utxo.amount));
        println!(
            "Account {} unspent notes: {}, balance {}",
            acc.id,
            acc.wallet.iter().filter(|n| !n.spent).count(),
            bal
        );
    }
}
