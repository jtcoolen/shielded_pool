use std::time::Instant;

use ff::Field;
use group::Group;
use thiserror::Error;

use midnight_circuits::{
    hash::poseidon::{PoseidonChip, PoseidonState},
    instructions::map::MapCPU,
    map::cpu::MapMt,
    types::{AssignedNativePoint, Instantiable},
};
use midnight_curves::{Fr as JubjubScalar, JubjubExtended as Jubjub, JubjubSubgroup};
use midnight_proofs::{circuit::Value, transcript::Transcript};
use midnight_proofs::{
    plonk::{VerifyingKey, create_proof, keygen_pk, keygen_vk_with_k, prepare},
    poly::kzg::{KZGCommitmentScheme, params::ParamsKZG},
};
use midnight_zk_stdlib::{self, MidnightPK, cost_model};
use rand::{Rng, SeedableRng, rngs::OsRng};
use rand_chacha::ChaCha8Rng;

use midnight_circuits::{
    ecc::foreign::ForeignEccChip,
    field::{NativeGadget, decomposition::chip::P2RDecompositionChip, native::NativeChip},
    types::AssignedForeignPoint,
    verifier::{BlstrsEmulation, SelfEmulation},
};

mod keccak_transcript;
mod rollup_ivc_circuits;
mod rollup_ivc_proofs;
mod setup_ivc;
mod transfer_circuit;
mod trusted_setup;

pub type S = BlstrsEmulation;
type F = <S as SelfEmulation>::F;
type C = <S as SelfEmulation>::C;
type E = <S as SelfEmulation>::Engine;
type NG = NativeGadget<F, P2RDecompositionChip<F>, NativeChip<F>>;

pub type CurveChip = ForeignEccChip<F, C, C, NG, NG>;
pub type MapGadget = midnight_circuits::map::map_gadget::MapGadget<F, NG, PoseidonChip<F>>;
pub type IdPoint = AssignedForeignPoint<
    midnight_curves::Fq,
    midnight_curves::G1Projective,
    midnight_curves::G1Projective,
>;

type CommitmentMap = MapMt<F, PoseidonChip<F>>;

const BATCH_SIZE: usize = 4;

/// Probability that a client proof is generated against an older confirmed root.
const LAG_TX_PROB: f64 = 0.35;

const K_INTERNAL: u32 = 19;
pub const AGG_K: u32 = K_INTERNAL;

#[derive(Debug, Error)]
enum AppError {
    #[error("trusted setup failed: {0}")]
    TrustedSetup(String),

    #[error("key generation failed: {0}")]
    Keygen(String),

    #[error("proof generation failed: {0}")]
    Proof(String),

    #[error("verification preparation failed: {0}")]
    VerificationPrep(String),

    #[error("invalid scalar-to-field conversion")]
    ScalarToField,

    #[error("replay guard failed: {0}")]
    ReplayGuard(String),
}

fn err_string<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

////////////////////////////////////////////////////////////////////////////////
// Host-side structures + helpers
////////////////////////////////////////////////////////////////////////////////

/// Single Poseidon hash of all 7 would-be public inputs (host-side).
fn host_instance_hash(items: [F; 7]) -> F {
    use midnight_circuits::instructions::hash::HashCPU;
    <PoseidonChip<F> as HashCPU<F, F>>::hash(&items)
}

/// A note is spendable if it is unspent and confirmed at or before `latest_confirmed_root_idx`.
fn is_spendable(note: &transfer_circuit::Note, latest_confirmed_root_idx: usize) -> bool {
    !note.spent && note.confirmed_at_root_idx <= latest_confirmed_root_idx
}

/// Return indices of spendable notes for an account.
fn spendable_note_indices(
    account: &transfer_circuit::Account,
    latest_confirmed_root_idx: usize,
) -> Vec<usize> {
    account
        .wallet
        .iter()
        .enumerate()
        .filter(|(_, n)| is_spendable(n, latest_confirmed_root_idx))
        .map(|(i, _)| i)
        .collect()
}

/// Choose a sender index that has at least two spendable notes.
fn choose_sender_idx(
    rng: &mut ChaCha8Rng,
    accounts: &[transfer_circuit::Account],
    latest_confirmed_root_idx: usize,
) -> Option<usize> {
    let viable: Vec<usize> = accounts
        .iter()
        .enumerate()
        .filter(|(_, a)| spendable_note_indices(a, latest_confirmed_root_idx).len() >= 2)
        .map(|(i, _)| i)
        .collect();

    if viable.is_empty() {
        None
    } else {
        Some(viable[rng.gen_range(0..viable.len())])
    }
}

/// Choose two distinct elements from a non-empty slice of candidates.
///
/// # Panics
/// Panics if `candidates` contains fewer than 2 distinct values.
fn choose_two_distinct(rng: &mut ChaCha8Rng, candidates: &[usize]) -> (usize, usize) {
    assert!(candidates.len() >= 2, "need at least 2 candidates");
    let idx_a = rng.gen_range(0..candidates.len());
    let a = candidates[idx_a];
    // Fast path: random pick at a different index.
    let mut idx_b = rng.gen_range(0..candidates.len() - 1);
    if idx_b >= idx_a {
        idx_b += 1;
    }
    let b = candidates[idx_b];
    if b != a {
        return (a, b);
    }
    // Fallback: linear scan for a distinct value (handles duplicate entries).
    for &c in candidates {
        if c != a {
            return (a, c);
        }
    }
    panic!("choose_two_distinct requires at least 2 distinct values in candidates");
}

/// Choose a (possibly lagging) confirmed root index to prove against.
fn choose_root_idx_for_proof(
    rng: &mut ChaCha8Rng,
    min_root_idx_for_inputs: usize,
    latest_confirmed_root_idx: usize,
) -> usize {
    if min_root_idx_for_inputs < latest_confirmed_root_idx && rng.gen_bool(LAG_TX_PROB) {
        // Force lag: pick strictly older than latest when possible.
        rng.gen_range(min_root_idx_for_inputs..=latest_confirmed_root_idx - 1)
    } else {
        latest_confirmed_root_idx
    }
}

fn commitment_for_utxo(utxo: &transfer_circuit::Utxo, pk_x: F, pk_y: F) -> F {
    transfer_circuit::host_commit(utxo.asset_id, utxo.amount, pk_x, pk_y, utxo.randomness)
}

fn nullifier_for_commit(commit: F, pk_x: F, pk_y: F) -> F {
    transfer_circuit::host_nullify(commit, pk_x, pk_y)
}

fn split_amount(rng: &mut ChaCha8Rng, total: u128) -> (u128, u128) {
    if total == 0 {
        (0, 0)
    } else {
        let out1 = rng.gen_range(0..=total);
        (out1, total - out1)
    }
}

fn random_amount(rng: &mut ChaCha8Rng) -> u128 {
    rng.r#gen::<u128>() >> (128 - transfer_circuit::AMOUNT_GEN_BITS)
}

fn blind_pubkey(sender_pk: JubjubSubgroup, alpha: JubjubScalar) -> (JubjubSubgroup, F, F) {
    let blind_point = JubjubSubgroup::generator() * alpha;
    let pk_blinded_point = sender_pk + blind_point;
    let fields = AssignedNativePoint::<Jubjub>::as_public_input(&pk_blinded_point);
    (pk_blinded_point, fields[0], fields[1])
}

fn build_public_items(
    root_before: F,
    pk_bx: F,
    pk_by: F,
    new1_commit: F,
    new2_commit: F,
    nf1: F,
    nf2: F,
) -> ([F; 7], F, transfer_circuit::Spend2Output2PublicInputs) {
    let public_items = [
        root_before,
        pk_bx,
        pk_by,
        new1_commit,
        new2_commit,
        nf1,
        nf2,
    ];
    let state = host_instance_hash(public_items);

    let instance = transfer_circuit::Spend2Output2PublicInputs {
        root: root_before,
        pk_bx,
        pk_by,
        new_c1: new1_commit,
        new_c2: new2_commit,
        nf1,
        nf2,
    };

    (public_items, state, instance)
}

/// Convert a Jubjub scalar to the circuit field `F`.
///
/// `ff::Field::from_bytes_le` returns `subtle::CtOption`, so we convert to `Option` first.
fn scalar_to_field(alpha: JubjubScalar) -> Result<F, AppError> {
    let ct = F::from_bytes_le(&alpha.to_bytes());
    Option::<F>::from(ct).ok_or(AppError::ScalarToField)
}

////////////////////////////////////////////////////////////////////////////////
// State containers
////////////////////////////////////////////////////////////////////////////////

#[derive(Clone)]
struct ChainState {
    /// Single global asset id used for all notes in this demo.
    asset_id: F,

    accounts: Vec<transfer_circuit::Account>,
    commitment_map: CommitmentMap,
    nullifier_map: CommitmentMap,
    commitment_roots_set: CommitmentMap,
    commitment_root_history: Vec<F>,
    commitment_map_history: Vec<CommitmentMap>,

    blk_head: u64,
}

struct BatchPreState {
    pre_commitment_map: CommitmentMap,
    pre_nullifier_map: CommitmentMap,
    pre_roots_set_map: CommitmentMap,
    latest_confirmed_root_idx: usize,
}

fn snapshot_batch_pre_state(state: &ChainState) -> BatchPreState {
    BatchPreState {
        pre_commitment_map: state.commitment_map.clone(),
        pre_nullifier_map: state.nullifier_map.clone(),
        pre_roots_set_map: state.commitment_roots_set.clone(),
        latest_confirmed_root_idx: state.commitment_root_history.len() - 1,
    }
}

fn init_accounts(num_accounts: usize) -> Vec<transfer_circuit::Account> {
    (0..num_accounts)
        .map(|i| {
            let sk = JubjubScalar::random(&mut OsRng);
            let pk_point = JubjubSubgroup::generator() * sk;
            let fields = AssignedNativePoint::<Jubjub>::as_public_input(&pk_point);
            transfer_circuit::Account {
                id: i,
                sk,
                pk_point,
                pk_x: fields[0],
                pk_y: fields[1],
                wallet: vec![],
            }
        })
        .collect()
}

fn seed_deposits(
    rng: &mut ChaCha8Rng,
    accounts: &mut [transfer_circuit::Account],
    commitment_map: &mut CommitmentMap,
    asset_id: F,
    deposits_per_account: usize,
) {
    for acc in accounts.iter_mut() {
        for _ in 0..deposits_per_account {
            let utxo = transfer_circuit::Utxo {
                asset_id,
                amount: random_amount(&mut *rng),
                randomness: F::random(&mut *rng),
            };

            let commit = commitment_for_utxo(&utxo, acc.pk_x, acc.pk_y);
            commitment_map.insert(&commit, &F::ONE);

            acc.wallet.push(transfer_circuit::Note {
                utxo,
                commit,
                spent: false,
                confirmed_at_root_idx: 0,
            });
        }
    }
}

fn init_chain_state(
    rng: &mut ChaCha8Rng,
    num_accounts: usize,
    deposits_per_account: usize,
) -> ChainState {
    let asset_id = F::random(&mut *rng);

    let mut accounts = init_accounts(num_accounts);
    let mut commitment_map = CommitmentMap::new(&F::ZERO);
    let nullifier_map = CommitmentMap::new(&F::ZERO);

    seed_deposits(
        &mut *rng,
        &mut accounts,
        &mut commitment_map,
        asset_id,
        deposits_per_account,
    );

    let genesis_root = commitment_map.succinct_repr();

    let mut commitment_roots_set = CommitmentMap::new(&F::ZERO);
    commitment_roots_set.insert(&genesis_root, &F::ONE);

    ChainState {
        asset_id,
        accounts,
        commitment_map_history: vec![commitment_map.clone()],
        commitment_root_history: vec![genesis_root],
        commitment_roots_set,
        commitment_map,
        nullifier_map,
        blk_head: 0,
    }
}

////////////////////////////////////////////////////////////////////////////////
// Transaction planning + execution
////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Debug)]
struct PlannedTx {
    sender_idx: usize,
    old1_idx: usize,
    old2_idx: usize,
    recipient1_idx: usize,
    recipient2_idx: usize,
    root_idx_for_proof: usize,
}

fn plan_transaction(
    rng: &mut ChaCha8Rng,
    shadow_accounts: &[transfer_circuit::Account],
    latest_confirmed_root_idx: usize,
) -> Option<PlannedTx> {
    let sender_idx = choose_sender_idx(rng, shadow_accounts, latest_confirmed_root_idx)?;
    let spendable = spendable_note_indices(&shadow_accounts[sender_idx], latest_confirmed_root_idx);
    let (old1_idx, old2_idx) = choose_two_distinct(rng, &spendable);

    let recipient1_idx = rng.gen_range(0..shadow_accounts.len());
    let recipient2_idx = rng.gen_range(0..shadow_accounts.len());

    let old1 = &shadow_accounts[sender_idx].wallet[old1_idx];
    let old2 = &shadow_accounts[sender_idx].wallet[old2_idx];
    let min_root_idx_for_inputs = old1.confirmed_at_root_idx.max(old2.confirmed_at_root_idx);

    let root_idx_for_proof =
        choose_root_idx_for_proof(rng, min_root_idx_for_inputs, latest_confirmed_root_idx);

    Some(PlannedTx {
        sender_idx,
        old1_idx,
        old2_idx,
        recipient1_idx,
        recipient2_idx,
        root_idx_for_proof,
    })
}

struct BuiltTx {
    // For proof payload
    public_items: [F; 7],
    state: F,
    instance: transfer_circuit::Spend2Output2PublicInputs,
    witness: (
        CommitmentMap,
        JubjubScalar,
        F,
        transfer_circuit::Utxo,
        transfer_circuit::Utxo,
        transfer_circuit::Utxo,
        transfer_circuit::Utxo,
        JubjubSubgroup,
        JubjubSubgroup,
    ),

    // For state updates
    nf1: F,
    nf2: F,
    new1_commit: F,
    new2_commit: F,
    new1_utxo: transfer_circuit::Utxo,
    new2_utxo: transfer_circuit::Utxo,
}

#[derive(Clone)]
struct TxEffects {
    nf1: F,
    nf2: F,
    new1_commit: F,
    new2_commit: F,
    new1_utxo: transfer_circuit::Utxo,
    new2_utxo: transfer_circuit::Utxo,
    sender_idx: usize,
    old1_idx: usize,
    old2_idx: usize,
    recipient1_idx: usize,
    recipient2_idx: usize,
}

fn effects_from(plan: &PlannedTx, built: &BuiltTx) -> TxEffects {
    TxEffects {
        nf1: built.nf1,
        nf2: built.nf2,
        new1_commit: built.new1_commit,
        new2_commit: built.new2_commit,
        new1_utxo: built.new1_utxo.clone(),
        new2_utxo: built.new2_utxo.clone(),
        sender_idx: plan.sender_idx,
        old1_idx: plan.old1_idx,
        old2_idx: plan.old2_idx,
        recipient1_idx: plan.recipient1_idx,
        recipient2_idx: plan.recipient2_idx,
    }
}

fn build_transaction(
    rng: &mut ChaCha8Rng,
    asset_id: F,
    shadow_accounts: &[transfer_circuit::Account],
    commitment_map_history: &[CommitmentMap],
    commitment_root_history: &[F],
    plan: &PlannedTx,
    latest_confirmed_root_idx: usize,
    batch_idx: usize,
    tx_idx: usize,
) -> Result<(BuiltTx, F), AppError> {
    let sender = shadow_accounts[plan.sender_idx].clone();
    let old1 = shadow_accounts[plan.sender_idx].wallet[plan.old1_idx].clone();
    let old2 = shadow_accounts[plan.sender_idx].wallet[plan.old2_idx].clone();

    let historic_commit_map = commitment_map_history[plan.root_idx_for_proof].clone();
    let root_before = commitment_root_history[plan.root_idx_for_proof];
    debug_assert_eq!(historic_commit_map.succinct_repr(), root_before);

    if plan.root_idx_for_proof != latest_confirmed_root_idx {
        println!(
            "[batch {}, tx {}] 🕒 lagging proof root: idx {} (latest {}), root {:?}",
            batch_idx, tx_idx, plan.root_idx_for_proof, latest_confirmed_root_idx, root_before
        );
    }

    let total = old1
        .utxo
        .amount
        .checked_add(old2.utxo.amount)
        .expect("amount overflow: sum of two input notes exceeds u128");
    let (out1_amt, out2_amt) = split_amount(&mut *rng, total);

    let new1_utxo = transfer_circuit::Utxo {
        asset_id,
        amount: out1_amt,
        randomness: F::random(&mut *rng),
    };
    let new2_utxo = transfer_circuit::Utxo {
        asset_id,
        amount: out2_amt,
        randomness: F::random(&mut *rng),
    };

    let r1 = plan.recipient1_idx;
    let r2 = plan.recipient2_idx;

    let new1_commit = commitment_for_utxo(
        &new1_utxo,
        shadow_accounts[r1].pk_x,
        shadow_accounts[r1].pk_y,
    );
    let new2_commit = commitment_for_utxo(
        &new2_utxo,
        shadow_accounts[r2].pk_x,
        shadow_accounts[r2].pk_y,
    );

    let nf1 = nullifier_for_commit(old1.commit, sender.pk_x, sender.pk_y);
    let nf2 = nullifier_for_commit(old2.commit, sender.pk_x, sender.pk_y);

    let alpha = JubjubScalar::random(&mut OsRng);
    let (_pk_blinded_point, pk_bx, pk_by) = blind_pubkey(sender.pk_point, alpha);
    let alpha_f = scalar_to_field(alpha)?;

    let (public_items, state, instance) = build_public_items(
        root_before,
        pk_bx,
        pk_by,
        new1_commit,
        new2_commit,
        nf1,
        nf2,
    );

    let witness = (
        historic_commit_map,
        sender.sk,
        alpha_f,
        old1.utxo.clone(),
        old2.utxo.clone(),
        new1_utxo.clone(),
        new2_utxo.clone(),
        shadow_accounts[r1].pk_point,
        shadow_accounts[r2].pk_point,
    );

    Ok((
        BuiltTx {
            public_items,
            state,
            instance,
            witness,
            nf1,
            nf2,
            new1_commit,
            new2_commit,
            new1_utxo,
            new2_utxo,
        },
        root_before,
    ))
}

fn apply_tx_effects(
    shadow_accounts: &mut [transfer_circuit::Account],
    shadow_commitment_map: &mut CommitmentMap,
    shadow_nullifier_map: &mut CommitmentMap,
    confirm_at_idx: usize,
    effects: &TxEffects,
) {
    // Nullifiers
    shadow_nullifier_map.insert(&effects.nf1, &F::ONE);
    shadow_nullifier_map.insert(&effects.nf2, &F::ONE);

    // Commitments
    shadow_commitment_map.insert(&effects.new1_commit, &F::ONE);
    shadow_commitment_map.insert(&effects.new2_commit, &F::ONE);

    // Mark spent inputs
    shadow_accounts[effects.sender_idx].wallet[effects.old1_idx].spent = true;
    shadow_accounts[effects.sender_idx].wallet[effects.old2_idx].spent = true;

    // Add newly created notes; they become spendable only after the batch commits.
    shadow_accounts[effects.recipient1_idx]
        .wallet
        .push(transfer_circuit::Note {
            utxo: effects.new1_utxo.clone(),
            commit: effects.new1_commit,
            spent: false,
            confirmed_at_root_idx: confirm_at_idx,
        });

    shadow_accounts[effects.recipient2_idx]
        .wallet
        .push(transfer_circuit::Note {
            utxo: effects.new2_utxo.clone(),
            commit: effects.new2_commit,
            spent: false,
            confirmed_at_root_idx: confirm_at_idx,
        });
}

fn prove_client(
    srs: &ParamsKZG<E>,
    pk: &MidnightPK<transfer_circuit::Spend2Output2>,
    relation: &transfer_circuit::Spend2Output2,
    built: BuiltTx,
) -> Result<rollup_ivc_proofs::ClientProof, AppError> {
    let now = Instant::now();
    let proof = midnight_zk_stdlib::prove::<transfer_circuit::Spend2Output2, PoseidonState<F>>(
        srs,
        pk,
        relation,
        &built.instance,
        built.witness,
        OsRng,
    )
    .map_err(|e| AppError::Proof(err_string(e)))?;

    println!("proof gen: {:?}", now.elapsed());

    Ok(rollup_ivc_proofs::ClientProof {
        state: built.state,
        proof,
        public_items: built.public_items,
    })
}

////////////////////////////////////////////////////////////////////////////////
// Replay demonstration
////////////////////////////////////////////////////////////////////////////////

fn demonstrate_replay_protection(
    agg_setup: &setup_ivc::AggSetup,
    srs: &ParamsKZG<E>,
    vk: &VerifyingKey<F, KZGCommitmentScheme<E>>,
    client_proofs: &[rollup_ivc_proofs::ClientProof],
    commitment_map: CommitmentMap,
    nullifier_map: CommitmentMap,
    roots_set_map: CommitmentMap,
    agg_state: &rollup_ivc_circuits::AggState,
) {
    println!("REPLAY attempt:");
    println!("  c_pre  = {:?}", agg_state.c_pre);
    println!("  c_post = {:?}", agg_state.c_post);
    println!("  n_pre  = {:?}", agg_state.n_pre);
    println!("  n_post = {:?}", agg_state.n_post);
    println!("  blk    = {:?}", agg_state.block_level);

    println!(
        "  roots_set has c_pre?  {:?}",
        roots_set_map.get(&agg_state.c_pre)
    );
    println!(
        "  roots_set has c_post? {:?}",
        roots_set_map.get(&agg_state.c_post)
    );

    println!(
        "  nullifier_map root == n_post? {}",
        nullifier_map.succinct_repr() == agg_state.n_post
    );

    let replay = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = rollup_ivc_proofs::aggregate_client_proofs_cached(
            agg_setup,
            srs,
            vk,
            client_proofs,
            commitment_map,
            nullifier_map,
            roots_set_map,
            F::ZERO,
        );
    }));

    match replay {
        Ok(_) => println!("❌ Replay unexpectedly succeeded (BUG)"),
        Err(_) => println!(
            "✅ Replay correctly rejected (nullifiers already spent / state already advanced)"
        ),
    }
}

////////////////////////////////////////////////////////////////////////////////
// Demo entrypoint
////////////////////////////////////////////////////////////////////////////////

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), AppError> {
    const LEAF_VK_NAME: &str = "spend2output2_vk";

    const K: u32 = 14;
    const NUM_ACCOUNTS: usize = 4;
    const NUM_SEED_DEPOSITS_PER_ACCOUNT: usize = 50;
    const NUM_TRANSFERS: usize = 120;

    // --- Setup leaf circuit keys ---
    let srs =
        trusted_setup::filecoin_srs_agg(K).map_err(|e| AppError::TrustedSetup(err_string(e)))?;
    let relation = transfer_circuit::Spend2Output2;
    let vk = midnight_zk_stdlib::setup_vk(&srs, &relation);
    let pk = midnight_zk_stdlib::setup_pk(&relation, &vk);

    // Cache aggregation setup once (fixed batch size).
    let agg_setup = setup_ivc::prepare_agg_setup(&srs, vk.vk(), LEAF_VK_NAME, K, BATCH_SIZE);

    // Cache final aggregation vk/pk once (depends only on cached agg_setup for this batch size).
    let final_agg_srs = trusted_setup::filecoin_srs_agg(AGG_K)
        .map_err(|e| AppError::TrustedSetup(err_string(e)))?;

    let default_final_circuit = rollup_ivc_circuits::WrapStepCircuit {
        child_vk: agg_setup.child_vk(),
        child_vk_name: agg_setup.child_vk_name().to_string(),
        left_proof: Value::unknown(),
        right_proof: Value::unknown(),
        left_pi_acc: Value::unknown(),
        right_pi_acc: Value::unknown(),
        fixed_base_names: agg_setup.fixed_base_names().to_vec(),
        left_child_state: Value::unknown(),
        right_child_state: Value::unknown(),
        agg_state: Value::unknown(),
        pre_commitment_roots_set_map: Value::unknown(),
        post_commitment_roots_set_root: Value::unknown(),
        blk_post: Value::unknown(),
        blk_pre: Value::unknown(),
    };

    let final_vk = keygen_vk_with_k(&final_agg_srs, &default_final_circuit, AGG_K)
        .map_err(|e| AppError::Keygen(err_string(e)))?;
    let final_pk = keygen_pk(final_vk.clone(), &default_final_circuit)
        .map_err(|e| AppError::Keygen(err_string(e)))?;

    // --- Initialize randomness and chain state ---
    let mut rng = ChaCha8Rng::from_entropy();
    let mut chain = init_chain_state(&mut rng, NUM_ACCOUNTS, NUM_SEED_DEPOSITS_PER_ACCOUNT);

    println!(
        "Initial commitment root: {:?}",
        chain.commitment_root_history[0]
    );

    // Client circuit stats are constant; compute once.
    let client_stats = cost_model(&transfer_circuit::Spend2Output2);
    println!("client circuit stats: {:?}", client_stats);

    // --- Rollup batching loop ---
    let mut total_transfers_done = 0usize;
    let mut batch_idx = 0usize;

    while total_transfers_done < NUM_TRANSFERS {
        let pre = snapshot_batch_pre_state(&chain);

        // Compute this batch's block transition.
        let blk_pre_u64 = chain.blk_head;
        let blk_post_u64 = chain.blk_head + 1;
        let blk_pre_f = F::from(blk_pre_u64);
        let blk_post_f = F::from(blk_post_u64);
        let batch_blk = blk_post_f; // subtree is bound to blk_post per spec

        // Shadow state for the batch.
        let mut shadow_accounts = chain.accounts.clone();
        let mut shadow_nullifier_map = chain.nullifier_map.clone();
        let mut shadow_commitment_map = chain.commitment_map.clone();

        println!(
            "\n=== Starting batch {} from commitment root {:?} ===",
            batch_idx,
            shadow_commitment_map.succinct_repr()
        );

        let mut client_proofs: Vec<rollup_ivc_proofs::ClientProof> = Vec::with_capacity(BATCH_SIZE);
        let mut batch_failed = false;

        for _ in 0..BATCH_SIZE {
            if total_transfers_done >= NUM_TRANSFERS {
                break;
            }

            let plan = match plan_transaction(
                &mut rng,
                &shadow_accounts,
                pre.latest_confirmed_root_idx,
            ) {
                Some(p) => p,
                None => {
                    println!(
                        "[batch {}] no account has two spendable confirmed notes; stopping batching.",
                        batch_idx
                    );
                    batch_failed = true;
                    break;
                }
            };

            // Build ONCE per tx (do not rebuild for effects; that breaks consistency).
            let (built, _root_before) = build_transaction(
                &mut rng,
                chain.asset_id,
                &shadow_accounts,
                &chain.commitment_map_history,
                &chain.commitment_root_history,
                &plan,
                pre.latest_confirmed_root_idx,
                batch_idx,
                total_transfers_done,
            )?;

            let effects = effects_from(&plan, &built);
            let proof = prove_client(&srs, &pk, &relation, built)?;
            client_proofs.push(proof);

            // Apply state transition to shadow rollup state using the exact same tx data as proved.
            apply_tx_effects(
                &mut shadow_accounts,
                &mut shadow_commitment_map,
                &mut shadow_nullifier_map,
                chain.commitment_root_history.len(),
                &effects,
            );

            println!(
                "[batch {}, tx {}] shadow commitment root updated: {:?}",
                batch_idx,
                total_transfers_done,
                shadow_commitment_map.succinct_repr()
            );

            total_transfers_done += 1;
        }

        if batch_failed || client_proofs.is_empty() {
            break;
        }

        if client_proofs.len() != BATCH_SIZE {
            return Err(AppError::ReplayGuard(format!(
                "batch not completely filled: got {}, expected {}",
                client_proofs.len(),
                BATCH_SIZE
            )));
        }
        if !client_proofs.len().is_power_of_two() {
            return Err(AppError::ReplayGuard(
                "batch size must be a power of two".to_string(),
            ));
        }

        // --- Aggregate client proofs (cached setup) ---
        let now = Instant::now();
        let agg_result = rollup_ivc_proofs::aggregate_client_proofs_cached(
            &agg_setup,
            &srs,
            vk.vk(),
            &client_proofs,
            pre.pre_commitment_map.clone(),
            pre.pre_nullifier_map.clone(),
            pre.pre_roots_set_map.clone(),
            batch_blk,
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

        // --- Replay guard via historic roots set update ---
        let pre_roots_set_root = pre.pre_roots_set_map.succinct_repr();
        if pre.pre_roots_set_map.get(&agg_result.root_state.c_post) != F::ZERO {
            return Err(AppError::ReplayGuard(
                "replay guard: c_post already present in roots set".to_string(),
            ));
        }

        let mut shadow_commitment_roots_set = pre.pre_roots_set_map.clone();
        shadow_commitment_roots_set.insert(&agg_result.root_state.c_post, &F::ONE);
        let post_roots_set_root = shadow_commitment_roots_set.succinct_repr();

        // --- Final merged aggregation proof ---
        {
            use midnight_proofs::transcript::CircuitTranscript;

            let mut final_acc: rollup_ivc_circuits::AggAccumulator =
                rollup_ivc_circuits::AggAccumulator::accumulate(&[
                    agg_result.left_top.proof_acc.clone(),
                    agg_result.left_top.pi_acc.clone(),
                    agg_result.right_top.proof_acc.clone(),
                    agg_result.right_top.pi_acc.clone(),
                ]);
            final_acc.collapse();
            let final_acc_pi = rollup_ivc_circuits::accumulator_as_public_input(&final_acc);

            let final_circuit = rollup_ivc_circuits::WrapStepCircuit {
                child_vk: agg_result.child_vk.clone(),
                child_vk_name: agg_result.child_vk_name.clone(),
                left_proof: Value::known(agg_result.left_top.proof.clone()),
                right_proof: Value::known(agg_result.right_top.proof.clone()),
                left_pi_acc: Value::known(agg_result.left_top.pi_acc.clone()),
                right_pi_acc: Value::known(agg_result.right_top.pi_acc.clone()),
                fixed_base_names: agg_result.fixed_base_names.clone(),
                left_child_state: Value::known(agg_result.left_top.state),
                right_child_state: Value::known(agg_result.right_top.state),
                agg_state: Value::known(agg_result.root_state),
                pre_commitment_roots_set_map: Value::known(pre.pre_roots_set_map.clone()),
                post_commitment_roots_set_root: Value::known(post_roots_set_root),
                blk_pre: Value::known(blk_pre_f),
                blk_post: Value::known(blk_post_f),
            };

            let mut final_public_inputs: Vec<F> = vec![
                agg_result.root_state.c_pre,
                agg_result.root_state.c_post,
                agg_result.root_state.n_pre,
                agg_result.root_state.n_post,
                // block counter transition (public) TODO we can expose only one block level
                blk_pre_f,
                blk_post_f,
                // batch subroot (public)
                agg_result.root_state.subroot,
                // historic roots set transition (public)
                pre_roots_set_root,
                post_roots_set_root,
            ];
            final_public_inputs.extend(final_acc_pi.clone());

            let final_proof_bytes = {
                let mut transcript =
                    CircuitTranscript::<keccak_transcript::KeccakTranscript>::init();
                create_proof::<
                    F,
                    KZGCommitmentScheme<E>,
                    CircuitTranscript<keccak_transcript::KeccakTranscript>,
                    rollup_ivc_circuits::WrapStepCircuit,
                >(
                    &final_agg_srs,
                    &final_pk,
                    &[final_circuit],
                    1,
                    &[&[&[], &final_public_inputs]],
                    OsRng,
                    &mut transcript,
                )
                .map_err(|e| AppError::Proof(err_string(e)))?;
                transcript.finalize()
            };

            println!("final proof size (bytes): {}", final_proof_bytes.len());

            let mut transcript =
                CircuitTranscript::<keccak_transcript::KeccakTranscript>::init_from_bytes(
                    &final_proof_bytes,
                );
            let committed_bases: &[&[midnight_curves::G1Projective]] =
                &[&[midnight_curves::G1Projective::identity()]];
            let instances: &[&[&[F]]] = &[&[&final_public_inputs]];

            let dual_msm = prepare::<
                F,
                KZGCommitmentScheme<E>,
                CircuitTranscript<keccak_transcript::KeccakTranscript>,
            >(&final_vk, committed_bases, instances, &mut transcript)
            .map_err(|e| AppError::VerificationPrep(err_string(e)))?;

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
                    Block counter transition: {} -> {}\n\
                    Final accumulator PI length: {} field elements",
                batch_idx,
                agg_result.root_state.subroot,
                agg_result.root_state.c_pre,
                agg_result.root_state.c_post,
                agg_result.root_state.n_pre,
                agg_result.root_state.n_post,
                pre_roots_set_root,
                post_roots_set_root,
                blk_pre_u64,
                blk_post_u64,
                final_acc_pi.len()
            );
        }

        // --- Commit batch to “chain state” ---
        chain.accounts = shadow_accounts;
        chain.nullifier_map = shadow_nullifier_map;
        chain.commitment_map = shadow_commitment_map;

        chain.commitment_roots_set = shadow_commitment_roots_set;
        chain
            .commitment_root_history
            .push(chain.commitment_map.succinct_repr());
        chain
            .commitment_map_history
            .push(chain.commitment_map.clone());

        // Advance the global head block number after accepting the final wrap proof.
        chain.blk_head = blk_post_u64;

        println!(
            "After batch {} committed commitment root: {:?}",
            batch_idx,
            chain.commitment_map.succinct_repr()
        );

        // Demonstrate replay protection using the POST state.
        demonstrate_replay_protection(
            &agg_setup,
            &srs,
            vk.vk(),
            &client_proofs,
            chain.commitment_map.clone(),
            chain.nullifier_map.clone(),
            chain.commitment_roots_set.clone(),
            &agg_result.root_state,
        );

        batch_idx += 1;
    }

    println!(
        "\nFinal commitment root: {:?}",
        chain.commitment_map.succinct_repr()
    );

    for acc in &chain.accounts {
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

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use rand_chacha::ChaCha8Rng;
    use std::panic;

    // -----------------------------
    // Unit tests (helpers/invariants)
    // -----------------------------

    #[test]
    fn unit_split_amount_conserves_total() {
        let mut rng = ChaCha8Rng::seed_from_u64(123);
        for _ in 0..1_000 {
            let total = rng.r#gen::<u128>();
            let (a, b) = split_amount(&mut rng, total);
            assert!(a <= total);
            assert!(b <= total);
            assert_eq!(a.saturating_add(b), total);
        }
    }

    #[test]
    fn unit_choose_two_distinct_returns_distinct() {
        let mut rng = ChaCha8Rng::seed_from_u64(456);
        let candidates = vec![10usize, 11, 12, 13, 14];
        for _ in 0..1_000 {
            let (a, b) = choose_two_distinct(&mut rng, &candidates);
            assert_ne!(a, b);
            assert!(candidates.contains(&a));
            assert!(candidates.contains(&b));
        }
    }

    #[test]
    fn unit_spendable_note_indices_matches_predicate() {
        let asset_id = F::random(&mut ChaCha8Rng::seed_from_u64(1));
        let mut rng = ChaCha8Rng::seed_from_u64(2);

        let mut mk_note = |spent: bool, confirmed: usize| transfer_circuit::Note {
            utxo: transfer_circuit::Utxo {
                asset_id,
                amount: 1,
                randomness: F::random(&mut rng),
            },
            commit: F::random(&mut rng),
            spent,
            confirmed_at_root_idx: confirmed,
        };

        let account = transfer_circuit::Account {
            id: 0,
            sk: JubjubScalar::random(&mut OsRng),
            pk_point: JubjubSubgroup::generator() * JubjubScalar::random(&mut OsRng),
            pk_x: F::ZERO,
            pk_y: F::ZERO,
            wallet: vec![
                mk_note(false, 0), // spendable at latest>=0
                mk_note(false, 3), // spendable only at latest>=3
                mk_note(true, 0),  // never spendable
            ],
        };

        assert_eq!(spendable_note_indices(&account, 0), vec![0]);
        assert_eq!(spendable_note_indices(&account, 2), vec![0]);
        assert_eq!(spendable_note_indices(&account, 3), vec![0, 1]);
    }

    // -----------------------------
    // PBT (property-based tests)
    // -----------------------------

    proptest! {
        #[test]
        fn pbt_split_amount_conserves(seed in any::<u64>(), total in any::<u128>()) {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let (a, b) = split_amount(&mut rng, total);
            prop_assert!(a <= total);
            prop_assert!(b <= total);
            prop_assert_eq!(a.saturating_add(b), total);
        }

        #[test]
        fn pbt_choose_root_idx_for_proof_in_range(
            seed in any::<u64>(),
            min_idx in 0usize..32,
            latest_offset in 0usize..32,
        ) {
            let latest = min_idx + latest_offset;
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let chosen = choose_root_idx_for_proof(&mut rng, min_idx, latest);
            prop_assert!(chosen >= min_idx);
            prop_assert!(chosen <= latest);
        }
    }

    // -----------------------------
    // Integration-style tests (prove + aggregate)
    // These validate the rollup safety properties end-to-end.
    //
    // NOTE: These are "integration-style" but live in the module to access private helpers.
    // -----------------------------

    struct MiniEnv {
        srs: ParamsKZG<E>,
        relation: transfer_circuit::Spend2Output2,
        leaf_vk: VerifyingKey<F, KZGCommitmentScheme<E>>,
        pk: MidnightPK<transfer_circuit::Spend2Output2>,
        agg_setup: setup_ivc::AggSetup,
        leaf_vk_name: &'static str,
        k: u32,
    }

    fn mini_env(k: u32, batch_size: usize) -> Result<MiniEnv, AppError> {
        const LEAF_VK_NAME: &str = "spend2output2_vk_test";

        let srs = trusted_setup::filecoin_srs_agg(k)
            .map_err(|e| AppError::TrustedSetup(err_string(e)))?;

        let relation = transfer_circuit::Spend2Output2;
        let vk_mid = midnight_zk_stdlib::setup_vk(&srs, &relation);
        let pk = midnight_zk_stdlib::setup_pk(&relation, &vk_mid);
        let leaf_vk = vk_mid.vk().clone();

        let agg_setup = setup_ivc::prepare_agg_setup(&srs, &leaf_vk, LEAF_VK_NAME, k, batch_size);

        Ok(MiniEnv {
            srs,
            relation,
            leaf_vk,
            pk,
            agg_setup,
            leaf_vk_name: LEAF_VK_NAME,
            k,
        })
    }

    fn make_chain(seed: u64, num_accounts: usize, deposits_per_account: usize) -> ChainState {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        init_chain_state(&mut rng, num_accounts, deposits_per_account)
    }

    /// Generate a full batch: plans txs, builds witnesses, produces client proofs, applies effects
    /// to obtain the post shadow state, then aggregates.
    fn prove_and_aggregate_one_batch(
        env: &MiniEnv,
        chain: &ChainState,
        seed: u64,
        batch_size: usize,
    ) -> Result<
        (
            BatchPreState,
            Vec<rollup_ivc_proofs::ClientProof>,
            CommitmentMap,
            CommitmentMap,
            rollup_ivc_proofs::AggregationResult,
        ),
        AppError,
    > {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let pre = snapshot_batch_pre_state(chain);

        let mut shadow_accounts = chain.accounts.clone();
        let mut shadow_nullifier_map = chain.nullifier_map.clone();
        let mut shadow_commitment_map = chain.commitment_map.clone();

        let mut client_proofs: Vec<rollup_ivc_proofs::ClientProof> = Vec::with_capacity(batch_size);

        for tx_idx in 0..batch_size {
            let plan = plan_transaction(&mut rng, &shadow_accounts, pre.latest_confirmed_root_idx)
                .expect(
                    "expected at least one viable sender with >=2 spendable notes in test setup",
                );

            // Build tx ONCE; proof must match effects
            let (built, _root_before) = build_transaction(
                &mut rng,
                chain.asset_id,
                &shadow_accounts,
                &chain.commitment_map_history,
                &chain.commitment_root_history,
                &plan,
                pre.latest_confirmed_root_idx,
                /*batch_idx=*/ 0,
                /*tx_idx=*/ tx_idx,
            )?;

            // Local conservation check (safety property: no inflation)
            let in_sum = built
                .witness
                .3
                .amount
                .saturating_add(built.witness.4.amount);
            let out_sum = built
                .witness
                .5
                .amount
                .saturating_add(built.witness.6.amount);
            assert_eq!(in_sum, out_sum, "amount must be conserved per tx");
            assert_eq!(built.witness.3.asset_id, built.witness.5.asset_id);
            assert_eq!(built.witness.4.asset_id, built.witness.6.asset_id);

            let effects = effects_from(&plan, &built);

            let proof = prove_client(&env.srs, &env.pk, &env.relation, built)?;
            client_proofs.push(proof);

            apply_tx_effects(
                &mut shadow_accounts,
                &mut shadow_commitment_map,
                &mut shadow_nullifier_map,
                chain.commitment_root_history.len(), // new outputs confirmed at next root index
                &effects,
            );
        }

        let agg_result = rollup_ivc_proofs::aggregate_client_proofs_cached(
            &env.agg_setup,
            &env.srs,
            &env.leaf_vk,
            &client_proofs,
            pre.pre_commitment_map.clone(),
            pre.pre_nullifier_map.clone(),
            pre.pre_roots_set_map.clone(),
            F::ZERO,
        );

        Ok((
            pre,
            client_proofs,
            shadow_commitment_map,
            shadow_nullifier_map,
            agg_result,
        ))
    }

    /// Positive: batch validity + deterministic root transitions.
    #[test]
    fn integration_batch_validity_roots_match_shadow_state() -> Result<(), AppError> {
        let batch_size = 4; // keep tests light; still a power-of-two
        let env = mini_env(/*k=*/ 14, batch_size)?;
        let chain = make_chain(
            /*seed=*/ 777, /*accounts=*/ 4, /*deposits_per_account=*/ 6,
        );

        let (pre, _client_proofs, post_cmap, post_nmap, agg) =
            prove_and_aggregate_one_batch(&env, &chain, /*seed=*/ 888, batch_size)?;

        // Safety: root pre bindings
        assert_eq!(agg.root_state.c_pre, pre.pre_commitment_map.succinct_repr());
        assert_eq!(agg.root_state.n_pre, pre.pre_nullifier_map.succinct_repr());

        // Safety: post roots equal applying the same txs to the shadow state
        assert_eq!(agg.root_state.c_post, post_cmap.succinct_repr());
        assert_eq!(agg.root_state.n_post, post_nmap.succinct_repr());

        // Safety: proofs must be against an admissible historic root (here genesis is in roots-set)
        assert_ne!(pre.pre_roots_set_map.get(&agg.root_state.c_pre), F::ZERO);

        // Replay-guard precondition: c_post must not already exist in roots-set at pre
        assert_eq!(pre.pre_roots_set_map.get(&agg.root_state.c_post), F::ZERO);

        Ok(())
    }

    /// Negative: replay using POST state should be rejected (nullifiers already spent / root advanced).
    #[test]
    fn negative_replay_aggregation_with_post_state_panics() -> Result<(), AppError> {
        let batch_size = 4;
        let env = mini_env(14, batch_size)?;
        let chain = make_chain(1234, 4, 6);

        let (pre, client_proofs, post_cmap, post_nmap, _agg) =
            prove_and_aggregate_one_batch(&env, &chain, 5678, batch_size)?;

        // Now attempt to re-aggregate the SAME client proofs, but against POST maps.
        // This should fail: leaves prove against old roots and nullifiers are already inserted.
        let replay = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            let _ = rollup_ivc_proofs::aggregate_client_proofs_cached(
                &env.agg_setup,
                &env.srs,
                &env.leaf_vk,
                &client_proofs,
                post_cmap.clone(),
                post_nmap.clone(),
                pre.pre_roots_set_map.clone(),
                F::ZERO,
            );
        }));

        assert!(replay.is_err(), "replay must be rejected");
        Ok(())
    }

    /// Negative: if the aggregator is given the wrong PRE state roots (tampered pre-map),
    /// aggregation must be rejected.
    #[test]
    fn negative_wrong_pre_state_detected_by_head_check() -> Result<(), AppError> {
        let batch_size = 4;
        let env = mini_env(14, batch_size)?;
        let chain = make_chain(42, 4, 6);

        let (pre, client_proofs, _post_cmap, _post_nmap, _agg) =
            prove_and_aggregate_one_batch(&env, &chain, 99, batch_size)?;

        // Tamper pre-state
        let mut wrong_pre_cmap = pre.pre_commitment_map.clone();
        let mut rng = ChaCha8Rng::seed_from_u64(2024);
        wrong_pre_cmap.insert(&F::random(&mut rng), &F::ONE);

        let agg = rollup_ivc_proofs::aggregate_client_proofs_cached(
            &env.agg_setup,
            &env.srs,
            &env.leaf_vk,
            &client_proofs,
            wrong_pre_cmap.clone(),
            pre.pre_nullifier_map.clone(),
            pre.pre_roots_set_map.clone(),
            F::ZERO,
        );

        // The aggregated batch starts from the wrong head (this is what the node must reject).
        assert_eq!(agg.root_state.c_pre, wrong_pre_cmap.succinct_repr());
        assert_ne!(agg.root_state.c_pre, pre.pre_commitment_map.succinct_repr());

        Ok(())
    }

    /// Negative (stronger double-spend): same nullifiers twice inside the same batch must be rejected.
    /// If this test fails, it’s a real bug relative to the stated safety property.
    #[test]
    fn negative_duplicate_nullifiers_within_batch_panics() -> Result<(), AppError> {
        let batch_size = 4;
        let env = mini_env(14, batch_size)?;
        let chain = make_chain(7, 4, 6);
        let pre = snapshot_batch_pre_state(&chain);

        let mut rng = ChaCha8Rng::seed_from_u64(8);
        let shadow_accounts = chain.accounts.clone();

        // Pick a sender and two spendable notes at pre state.
        let sender_idx =
            choose_sender_idx(&mut rng, &shadow_accounts, pre.latest_confirmed_root_idx)
                .expect("need viable sender");
        let spendable =
            spendable_note_indices(&shadow_accounts[sender_idx], pre.latest_confirmed_root_idx);
        let (old1_idx, old2_idx) = choose_two_distinct(&mut rng, &spendable);

        // Force two txs spending the same inputs (=> same nullifiers), but with different outputs.
        let plan = PlannedTx {
            sender_idx,
            old1_idx,
            old2_idx,
            recipient1_idx: rng.gen_range(0..shadow_accounts.len()),
            recipient2_idx: rng.gen_range(0..shadow_accounts.len()),
            root_idx_for_proof: pre.latest_confirmed_root_idx,
        };

        let mut proofs = Vec::with_capacity(batch_size);

        for tx_idx in 0..batch_size {
            let (built, _) = build_transaction(
                &mut rng,
                chain.asset_id,
                &shadow_accounts,
                &chain.commitment_map_history,
                &chain.commitment_root_history,
                &plan,
                pre.latest_confirmed_root_idx,
                0,
                tx_idx,
            )?;
            let proof = prove_client(&env.srs, &env.pk, &env.relation, built)?;
            proofs.push(proof);
        }

        let res = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            let _ = rollup_ivc_proofs::aggregate_client_proofs_cached(
                &env.agg_setup,
                &env.srs,
                &env.leaf_vk,
                &proofs,
                pre.pre_commitment_map.clone(),
                pre.pre_nullifier_map.clone(),
                pre.pre_roots_set_map.clone(),
                F::ZERO,
            );
        }));

        assert!(
            res.is_err(),
            "batch must reject duplicate nullifiers (double-spend) inside the same batch"
        );

        Ok(())
    }
}
