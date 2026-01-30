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

use midnight_circuits::{
    ecc::foreign::ForeignEccChip,
    field::{NativeGadget, decomposition::chip::P2RDecompositionChip, native::NativeChip},
    types::AssignedForeignPoint,
    verifier::{BlstrsEmulation, SelfEmulation},
};

mod keccak;
mod proof;
mod rollup_ivc;
mod setup;
mod srs;
mod transfer;

pub type S = BlstrsEmulation;
type F = <S as SelfEmulation>::F;
type C = <S as SelfEmulation>::C;
type NG = NativeGadget<F, P2RDecompositionChip<F>, NativeChip<F>>;

const BATCH_SIZE: usize = 4;

// NEW: probability that a client proof is generated against an older confirmed root
const LAG_TX_PROB: f64 = 0.35;

const K_INTERNAL: u32 = 19;

pub const AGG_K: u32 = K_INTERNAL;

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

// ✅ Single Poseidon hash of all 7 would-be public inputs (host-side)
fn host_instance_hash(items: [F; 7]) -> F {
    use midnight_circuits::instructions::hash::HashCPU;
    <PoseidonChip<F> as HashCPU<F, F>>::hash(&items)
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

        let mut client_proofs: Vec<proof::ClientProof> = Vec::new();

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

            client_proofs.push(proof::ClientProof {
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
        let agg_result = proof::aggregate_client_proofs_cached(
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
            let _ = proof::aggregate_client_proofs_cached(
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
