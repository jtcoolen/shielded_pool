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

use midnight_circuits::verifier::{Accumulator, AssignedAccumulator};

mod ivc;
mod keccak_transcript;
mod rollup;
mod transfer_circuit;
mod trusted_setup;

use ivc::{F, E, ClientProof, IvcSetup, IvcProver, TreeResult, IvcDeciderCircuit};
use ivc::engine::{prepare_ivc_setup, host_instance_hash, LeafPlan};
use ivc::circuit::FrameworkWitness;
use rollup::{
    RollupLeafStep, RollupFoldStep, RollupDeciderStep, RollupAppState,
    LeafWitness, DeciderWitness, APP_STATE_WIDTH,
    rollup_host_merge, plan_rollup_leaves,
};

type CommitmentMap = MapMt<F, PoseidonChip<F>>;

const BATCH_SIZE: usize = 4;
const LAG_TX_PROB: f64 = 0.35;
const K_AGG: u32 = 19;

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
    #[error("aggregation failed: {0}")]
    Aggregation(#[from] ivc::engine::AggregationError),
}

fn err_string<Err: std::fmt::Display>(e: Err) -> String {
    e.to_string()
}

////////////////////////////////////////////////////////////////////////////////
// Host-side helpers (unchanged from before)
////////////////////////////////////////////////////////////////////////////////

fn is_spendable(note: &transfer_circuit::Note, latest_confirmed_root_idx: usize) -> bool {
    !note.spent && note.confirmed_at_root_idx <= latest_confirmed_root_idx
}

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
    if viable.is_empty() { None } else { Some(viable[rng.gen_range(0..viable.len())]) }
}

fn choose_two_distinct(rng: &mut ChaCha8Rng, candidates: &[usize]) -> (usize, usize) {
    assert!(candidates.len() >= 2, "need at least 2 candidates");
    let idx_a = rng.gen_range(0..candidates.len());
    let a = candidates[idx_a];
    let mut idx_b = rng.gen_range(0..candidates.len() - 1);
    if idx_b >= idx_a { idx_b += 1; }
    let b = candidates[idx_b];
    if b != a { return (a, b); }
    for &c in candidates { if c != a { return (a, c); } }
    panic!("choose_two_distinct requires at least 2 distinct values");
}

fn choose_root_idx_for_proof(
    rng: &mut ChaCha8Rng, min_idx: usize, latest: usize,
) -> usize {
    if min_idx < latest && rng.gen_bool(LAG_TX_PROB) {
        rng.gen_range(min_idx..=latest - 1)
    } else { latest }
}

fn commitment_for_utxo(utxo: &transfer_circuit::Utxo, pk_x: F, pk_y: F) -> F {
    transfer_circuit::host_commit(utxo.asset_id, utxo.amount, pk_x, pk_y, utxo.randomness)
}

fn nullifier_for_commit(commit: F, pk_x: F, pk_y: F) -> F {
    transfer_circuit::host_nullify(commit, pk_x, pk_y)
}

fn split_amount(rng: &mut ChaCha8Rng, total: u128) -> (u128, u128) {
    if total == 0 { (0, 0) } else { let a = rng.gen_range(0..=total); (a, total - a) }
}

fn random_amount(rng: &mut ChaCha8Rng) -> u128 {
    rng.r#gen::<u128>() >> (128 - transfer_circuit::AMOUNT_GEN_BITS)
}

fn blind_pubkey(sender_pk: JubjubSubgroup, alpha: JubjubScalar) -> (JubjubSubgroup, F, F) {
    let blind_point = JubjubSubgroup::generator() * alpha;
    let pk_blinded = sender_pk + blind_point;
    let fields = AssignedNativePoint::<Jubjub>::as_public_input(&pk_blinded);
    (pk_blinded, fields[0], fields[1])
}

fn scalar_to_field(alpha: JubjubScalar) -> Result<F, AppError> {
    let ct = F::from_bytes_le(&alpha.to_bytes());
    Option::<F>::from(ct).ok_or(AppError::ScalarToField)
}

////////////////////////////////////////////////////////////////////////////////
// Chain state
////////////////////////////////////////////////////////////////////////////////

#[derive(Clone)]
struct ChainState {
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

fn init_chain_state(rng: &mut ChaCha8Rng, num_accounts: usize, deposits: usize) -> ChainState {
    let asset_id = F::random(&mut *rng);
    let mut accounts: Vec<transfer_circuit::Account> = (0..num_accounts)
        .map(|i| {
            let sk = JubjubScalar::random(&mut OsRng);
            let pk = JubjubSubgroup::generator() * sk;
            let f = AssignedNativePoint::<Jubjub>::as_public_input(&pk);
            transfer_circuit::Account { id: i, sk, pk_point: pk, pk_x: f[0], pk_y: f[1], wallet: vec![] }
        })
        .collect();

    let mut commitment_map = CommitmentMap::new(&F::ZERO);
    let nullifier_map = CommitmentMap::new(&F::ZERO);

    for acc in accounts.iter_mut() {
        for _ in 0..deposits {
            let utxo = transfer_circuit::Utxo {
                asset_id, amount: random_amount(rng), randomness: F::random(&mut *rng),
            };
            let commit = commitment_for_utxo(&utxo, acc.pk_x, acc.pk_y);
            commitment_map.insert(&commit, &F::ONE);
            acc.wallet.push(transfer_circuit::Note { utxo, commit, spent: false, confirmed_at_root_idx: 0 });
        }
    }

    let genesis_root = commitment_map.succinct_repr();
    let mut commitment_roots_set = CommitmentMap::new(&F::ZERO);
    commitment_roots_set.insert(&genesis_root, &F::ONE);

    ChainState {
        asset_id, accounts,
        commitment_map_history: vec![commitment_map.clone()],
        commitment_root_history: vec![genesis_root],
        commitment_roots_set, commitment_map, nullifier_map,
        blk_head: 0,
    }
}

////////////////////////////////////////////////////////////////////////////////
// Transaction building + client proving
////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Debug)]
struct PlannedTx {
    sender_idx: usize, old1_idx: usize, old2_idx: usize,
    recipient1_idx: usize, recipient2_idx: usize,
    root_idx_for_proof: usize,
}

#[derive(Clone)]
struct TxEffects {
    nf1: F, nf2: F, new1_commit: F, new2_commit: F,
    new1_utxo: transfer_circuit::Utxo, new2_utxo: transfer_circuit::Utxo,
    sender_idx: usize, old1_idx: usize, old2_idx: usize,
    recipient1_idx: usize, recipient2_idx: usize,
}

fn plan_transaction(
    rng: &mut ChaCha8Rng,
    accounts: &[transfer_circuit::Account],
    latest_confirmed_root_idx: usize,
) -> Option<PlannedTx> {
    let sender_idx = choose_sender_idx(rng, accounts, latest_confirmed_root_idx)?;
    let spendable = spendable_note_indices(&accounts[sender_idx], latest_confirmed_root_idx);
    let (old1_idx, old2_idx) = choose_two_distinct(rng, &spendable);
    let recipient1_idx = rng.gen_range(0..accounts.len());
    let recipient2_idx = rng.gen_range(0..accounts.len());
    let old1 = &accounts[sender_idx].wallet[old1_idx];
    let old2 = &accounts[sender_idx].wallet[old2_idx];
    let min_root = old1.confirmed_at_root_idx.max(old2.confirmed_at_root_idx);
    let root_idx_for_proof = choose_root_idx_for_proof(rng, min_root, latest_confirmed_root_idx);
    Some(PlannedTx { sender_idx, old1_idx, old2_idx, recipient1_idx, recipient2_idx, root_idx_for_proof })
}

fn build_and_prove_tx(
    rng: &mut ChaCha8Rng,
    srs: &ParamsKZG<E>,
    pk: &MidnightPK<transfer_circuit::Spend2Output2>,
    relation: &transfer_circuit::Spend2Output2,
    chain: &ChainState,
    accounts: &[transfer_circuit::Account],
    plan: &PlannedTx,
    batch_idx: usize,
    tx_idx: usize,
    latest_confirmed_root_idx: usize,
) -> Result<(ClientProof, TxEffects), AppError> {
    let sender = accounts[plan.sender_idx].clone();
    let old1 = accounts[plan.sender_idx].wallet[plan.old1_idx].clone();
    let old2 = accounts[plan.sender_idx].wallet[plan.old2_idx].clone();

    let historic_map = chain.commitment_map_history[plan.root_idx_for_proof].clone();
    let root_before = chain.commitment_root_history[plan.root_idx_for_proof];

    if plan.root_idx_for_proof != latest_confirmed_root_idx {
        println!("[batch {batch_idx}, tx {tx_idx}] lagging proof root: idx {} (latest {latest_confirmed_root_idx})", plan.root_idx_for_proof);
    }

    let total = old1.utxo.amount.checked_add(old2.utxo.amount)
        .expect("amount overflow");
    let (out1, out2) = split_amount(rng, total);

    let new1 = transfer_circuit::Utxo { asset_id: chain.asset_id, amount: out1, randomness: F::random(&mut *rng) };
    let new2 = transfer_circuit::Utxo { asset_id: chain.asset_id, amount: out2, randomness: F::random(&mut *rng) };

    let (r1, r2) = (plan.recipient1_idx, plan.recipient2_idx);
    let new1_commit = commitment_for_utxo(&new1, accounts[r1].pk_x, accounts[r1].pk_y);
    let new2_commit = commitment_for_utxo(&new2, accounts[r2].pk_x, accounts[r2].pk_y);
    let nf1 = nullifier_for_commit(old1.commit, sender.pk_x, sender.pk_y);
    let nf2 = nullifier_for_commit(old2.commit, sender.pk_x, sender.pk_y);

    let alpha = JubjubScalar::random(&mut OsRng);
    let (_, pk_bx, pk_by) = blind_pubkey(sender.pk_point, alpha);
    let alpha_f = scalar_to_field(alpha)?;

    let public_items = [root_before, pk_bx, pk_by, new1_commit, new2_commit, nf1, nf2];
    let instance_hash = host_instance_hash(&public_items);
    let instance = transfer_circuit::Spend2Output2PublicInputs {
        root: root_before, pk_bx, pk_by, new_c1: new1_commit, new_c2: new2_commit, nf1, nf2,
    };
    let witness = (historic_map, sender.sk, alpha_f, old1.utxo.clone(), old2.utxo.clone(), new1.clone(), new2.clone(), accounts[r1].pk_point, accounts[r2].pk_point);

    let now = Instant::now();
    let proof_bytes = midnight_zk_stdlib::prove::<transfer_circuit::Spend2Output2, PoseidonState<F>>(
        srs, pk, relation, &instance, witness, OsRng,
    ).map_err(|e| AppError::Proof(err_string(e)))?;
    println!("proof gen: {:?}", now.elapsed());

    let client_proof = ClientProof {
        proof: proof_bytes,
        public_inputs: public_items.to_vec(),
        instance_hash,
    };

    let effects = TxEffects {
        nf1, nf2, new1_commit, new2_commit,
        new1_utxo: new1, new2_utxo: new2,
        sender_idx: plan.sender_idx, old1_idx: plan.old1_idx, old2_idx: plan.old2_idx,
        recipient1_idx: plan.recipient1_idx, recipient2_idx: plan.recipient2_idx,
    };

    Ok((client_proof, effects))
}

fn apply_tx_effects(
    accounts: &mut [transfer_circuit::Account],
    cmap: &mut CommitmentMap,
    nmap: &mut CommitmentMap,
    confirm_at_idx: usize,
    effects: &TxEffects,
) {
    nmap.insert(&effects.nf1, &F::ONE);
    nmap.insert(&effects.nf2, &F::ONE);
    cmap.insert(&effects.new1_commit, &F::ONE);
    cmap.insert(&effects.new2_commit, &F::ONE);
    accounts[effects.sender_idx].wallet[effects.old1_idx].spent = true;
    accounts[effects.sender_idx].wallet[effects.old2_idx].spent = true;
    accounts[effects.recipient1_idx].wallet.push(transfer_circuit::Note {
        utxo: effects.new1_utxo.clone(), commit: effects.new1_commit, spent: false, confirmed_at_root_idx: confirm_at_idx,
    });
    accounts[effects.recipient2_idx].wallet.push(transfer_circuit::Note {
        utxo: effects.new2_utxo.clone(), commit: effects.new2_commit, spent: false, confirmed_at_root_idx: confirm_at_idx,
    });
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
    const NUM_SEED_DEPOSITS: usize = 50;
    const NUM_TRANSFERS: usize = 120;

    // --- Setup leaf circuit keys ---
    let srs = trusted_setup::filecoin_srs_agg(K).map_err(|e| AppError::TrustedSetup(err_string(e)))?;
    let relation = transfer_circuit::Spend2Output2;
    let vk = midnight_zk_stdlib::setup_vk(&srs, &relation);
    let pk = midnight_zk_stdlib::setup_pk(&relation, &vk);

    // --- Setup IVC aggregation (new API) ---
    let ivc_setup = prepare_ivc_setup(
        &RollupLeafStep, &RollupFoldStep,
        &srs, vk.vk(), LEAF_VK_NAME, K,
        K_AGG, K_AGG, BATCH_SIZE,
        APP_STATE_WIDTH, 7,
    ).map_err(|e| AppError::TrustedSetup(err_string(e)))?;

    // --- Setup decider circuit keys ---
    let final_srs = trusted_setup::filecoin_srs_agg(K_AGG).map_err(|e| AppError::TrustedSetup(err_string(e)))?;
    let full_width = APP_STATE_WIDTH + 1;
    let default_decider = IvcDeciderCircuit::<RollupDeciderStep, 19> {
        step: RollupDeciderStep,
        app_state_width: APP_STATE_WIDTH,
        left_child_state: vec![Value::unknown(); full_width],
        right_child_state: vec![Value::unknown(); full_width],
        witness: Value::unknown(),
        fw: FrameworkWitness {
            child_vk: ivc_setup.child_vk(),
            child_vk_name: ivc_setup.child_vk_name().to_string(),
            left_proof: Value::unknown(),
            right_proof: Value::unknown(),
            left_pi_acc: Value::unknown(),
            right_pi_acc: Value::unknown(),
            fixed_base_names: ivc_setup.fixed_base_names().to_vec(),
        },
    };
    let final_vk = keygen_vk_with_k(&final_srs, &default_decider, K_AGG)
        .map_err(|e| AppError::Keygen(err_string(e)))?;
    let final_pk = keygen_pk(final_vk.clone(), &default_decider)
        .map_err(|e| AppError::Keygen(err_string(e)))?;

    // --- Initialize chain state ---
    let mut rng = ChaCha8Rng::from_entropy();
    let mut chain = init_chain_state(&mut rng, NUM_ACCOUNTS, NUM_SEED_DEPOSITS);

    println!("Initial commitment root: {:?}", chain.commitment_root_history[0]);
    let client_stats = cost_model(&transfer_circuit::Spend2Output2);
    println!("client circuit stats: {:?}", client_stats);

    // --- Rollup batching loop ---
    let mut total_done = 0usize;
    let mut batch_idx = 0usize;

    while total_done < NUM_TRANSFERS {
        let pre = snapshot_batch_pre_state(&chain);
        let blk_pre = chain.blk_head;
        let blk_post = blk_pre + 1;
        let blk_post_f = F::from(blk_post);

        let mut shadow_accounts = chain.accounts.clone();
        let mut shadow_cmap = chain.commitment_map.clone();
        let mut shadow_nmap = chain.nullifier_map.clone();

        println!("\n=== Starting batch {batch_idx} from root {:?} ===", shadow_cmap.succinct_repr());

        let mut client_proofs: Vec<ClientProof> = Vec::with_capacity(BATCH_SIZE);
        let mut batch_failed = false;

        for _ in 0..BATCH_SIZE {
            if total_done >= NUM_TRANSFERS { break; }

            let plan = match plan_transaction(&mut rng, &shadow_accounts, pre.latest_confirmed_root_idx) {
                Some(p) => p,
                None => { println!("[batch {batch_idx}] no viable sender"); batch_failed = true; break; }
            };

            let (proof, effects) = build_and_prove_tx(
                &mut rng, &srs, &pk, &relation, &chain, &shadow_accounts,
                &plan, batch_idx, total_done, pre.latest_confirmed_root_idx,
            )?;
            client_proofs.push(proof);

            apply_tx_effects(
                &mut shadow_accounts, &mut shadow_cmap, &mut shadow_nmap,
                chain.commitment_root_history.len(), &effects,
            );
            total_done += 1;
        }

        if batch_failed || client_proofs.is_empty() { break; }
        if client_proofs.len() != BATCH_SIZE {
            return Err(AppError::ReplayGuard(format!("batch incomplete: {}/{BATCH_SIZE}", client_proofs.len())));
        }

        // --- Plan leaves and prove tree via IVC framework ---
        let leaf_plans = plan_rollup_leaves(
            &client_proofs,
            pre.pre_commitment_map.clone(),
            pre.pre_nullifier_map.clone(),
            &pre.pre_roots_set_map,
            blk_post_f,
        )?;

        let now = Instant::now();
        let tree = IvcProver::prove_tree(
            &ivc_setup, &srs, vk.vk(),
            &RollupLeafStep, &RollupFoldStep,
            leaf_plans, rollup_host_merge,
        )?;
        println!("Batch {batch_idx} tree aggregated in {:?}", now.elapsed());

        // --- Replay guard: c_post must not already be in roots set ---
        let pre_roots_set_root = pre.pre_roots_set_map.succinct_repr();
        let c_post = tree.root_state.app_state[1];
        if pre.pre_roots_set_map.get(&c_post) != F::ZERO {
            return Err(AppError::ReplayGuard("c_post already in roots set".into()));
        }

        let mut shadow_roots_set = pre.pre_roots_set_map.clone();
        shadow_roots_set.insert(&c_post, &F::ONE);
        let post_roots_set_root = shadow_roots_set.succinct_repr();

        // --- Final decider proof ---
        {
            use midnight_proofs::transcript::CircuitTranscript;

            let mut final_acc: Accumulator<ivc::S> = Accumulator::accumulate(&[
                tree.left_top.proof_acc.clone(), tree.left_top.pi_acc.clone(),
                tree.right_top.proof_acc.clone(), tree.right_top.pi_acc.clone(),
            ]);
            final_acc.collapse();
            let final_acc_pi = AssignedAccumulator::as_public_input(&final_acc);

            let mut left_full = tree.left_top.app_state.clone();
            left_full.push(tree.left_top.merkle_digest);
            let mut right_full = tree.right_top.app_state.clone();
            right_full.push(tree.right_top.merkle_digest);

            let final_circuit = IvcDeciderCircuit::<RollupDeciderStep, 19> {
                step: RollupDeciderStep,
                app_state_width: APP_STATE_WIDTH,
                left_child_state: left_full.iter().map(|f| Value::known(*f)).collect(),
                right_child_state: right_full.iter().map(|f| Value::known(*f)).collect(),
                witness: Value::known(DeciderWitness {
                    pre_commitment_roots_set_map: pre.pre_roots_set_map.clone(),
                    post_commitment_roots_set_root: post_roots_set_root,
                    blk_pre: F::from(blk_pre),
                    blk_post: blk_post_f,
                }),
                fw: FrameworkWitness {
                    child_vk: ivc_setup.child_vk(),
                    child_vk_name: ivc_setup.child_vk_name().to_string(),
                    left_proof: Value::known(tree.left_top.proof.clone()),
                    right_proof: Value::known(tree.right_top.proof.clone()),
                    left_pi_acc: Value::known(tree.left_top.pi_acc.clone()),
                    right_pi_acc: Value::known(tree.right_top.pi_acc.clone()),
                    fixed_base_names: ivc_setup.fixed_base_names().to_vec(),
                },
            };

            let merkle_root = ivc::engine::host_instance_hash(&[tree.left_top.merkle_digest, tree.right_top.merkle_digest]);
            let _ = merkle_root; // used only for logging

            let mut final_pi: Vec<F> = vec![
                tree.root_state.app_state[0], // c_pre
                tree.root_state.app_state[1], // c_post
                tree.root_state.app_state[2], // n_pre
                tree.root_state.app_state[3], // n_post
                F::from(blk_pre),
                blk_post_f,
                tree.root_state.merkle_digest, // merkle root
                pre_roots_set_root,
                post_roots_set_root,
            ];
            final_pi.extend(final_acc_pi.clone());

            let final_proof_bytes = {
                let mut transcript = CircuitTranscript::<keccak_transcript::KeccakTranscript>::init();
                create_proof::<F, KZGCommitmentScheme<E>, CircuitTranscript<keccak_transcript::KeccakTranscript>, IvcDeciderCircuit<RollupDeciderStep, 19>>(
                    &final_srs, &final_pk, &[final_circuit], 1,
                    &[&[&[], &final_pi]], OsRng, &mut transcript,
                ).map_err(|e| AppError::Proof(err_string(e)))?;
                transcript.finalize()
            };

            println!("final proof size (bytes): {}", final_proof_bytes.len());

            let mut transcript = CircuitTranscript::<keccak_transcript::KeccakTranscript>::init_from_bytes(&final_proof_bytes);
            let committed: &[&[midnight_curves::G1Projective]] = &[&[midnight_curves::G1Projective::identity()]];
            let instances: &[&[&[F]]] = &[&[&final_pi]];

            let dual_msm = prepare::<F, KZGCommitmentScheme<E>, CircuitTranscript<keccak_transcript::KeccakTranscript>>(
                &final_vk, committed, instances, &mut transcript,
            ).map_err(|e| AppError::VerificationPrep(err_string(e)))?;

            assert!(dual_msm.check(&final_srs.verifier_params()), "Final proof must verify");
            assert!(final_acc.check(&final_srs.s_g2().into(), &ivc_setup.fixed_bases), "Final acc must verify");

            println!(
                "\nFinal proof for batch {batch_idx} verified.\n\
                 Commitment-set: {:?} -> {:?}\n\
                 Nullifier-set: {:?} -> {:?}\n\
                 Block: {blk_pre} -> {blk_post}",
                tree.root_state.app_state[0], tree.root_state.app_state[1],
                tree.root_state.app_state[2], tree.root_state.app_state[3],
            );
        }

        // --- Commit batch ---
        chain.accounts = shadow_accounts;
        chain.nullifier_map = shadow_nmap;
        chain.commitment_map = shadow_cmap;
        chain.commitment_roots_set = shadow_roots_set;
        chain.commitment_root_history.push(chain.commitment_map.succinct_repr());
        chain.commitment_map_history.push(chain.commitment_map.clone());
        chain.blk_head = blk_post;

        batch_idx += 1;
    }

    println!("\nFinal commitment root: {:?}", chain.commitment_map.succinct_repr());
    for acc in &chain.accounts {
        let bal: u128 = acc.wallet.iter().filter(|n| !n.spent).fold(0u128, |s, n| s.saturating_add(n.utxo.amount));
        println!("Account {} unspent: {}, balance {bal}", acc.id, acc.wallet.iter().filter(|n| !n.spent).count());
    }

    Ok(())
}
