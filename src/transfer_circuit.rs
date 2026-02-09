use ff::{Field, PrimeField};
use group::Group;
use midnight_zk_stdlib::{self, Relation, ZkStdLib, ZkStdLibArch};

use midnight_circuits::{
    biguint::AssignedBigUint,
    ecc::native::AssignedScalarOfNativeCurve,
    hash::poseidon::PoseidonChip,
    instructions::{
        AssertionInstructions, AssignmentInstructions, ConversionInstructions,
        DecompositionInstructions, EccInstructions, PublicInputInstructions, ZeroInstructions,
        hash::HashCPU, map::MapInstructions,
    },
    map::cpu::MapMt,
    types::{AssignedNative, AssignedNativePoint},
};
use midnight_curves::{Fr as JubjubScalar, JubjubExtended as Jubjub, JubjubSubgroup};
use midnight_proofs::circuit::{Layouter, Value};
use midnight_proofs::plonk::Error;

use midnight_circuits::verifier::{BlstrsEmulation, SelfEmulation};

pub type S = BlstrsEmulation;
type F = <S as SelfEmulation>::F;

const UTXO_COMMIT_TAG: u64 = 0x0001;
const UTXO_NULLIFY_TAG: u64 = 0x0002;
const AMOUNT_BITS: u32 = 128;
pub(crate) const AMOUNT_GEN_BITS: u32 = 120;

/// Public inputs (unhashed): these are constrained individually as public inputs, in this exact order.
#[derive(Clone, Copy, Debug)]
pub struct Spend2Output2PublicInputs {
    pub root: F,
    pub pk_bx: F,
    pub pk_by: F,
    pub new_c1: F,
    pub new_c2: F,
    pub nf1: F,
    pub nf2: F,
}

impl Default for Spend2Output2PublicInputs {
    fn default() -> Self {
        Self {
            root: F::ZERO,
            pk_bx: F::ZERO,
            pk_by: F::ZERO,
            new_c1: F::ZERO,
            new_c2: F::ZERO,
            nf1: F::ZERO,
            nf2: F::ZERO,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Utxo {
    pub asset_id: F,
    pub amount: u128,
    pub randomness: F,
}

#[derive(Clone, Default)]
pub struct Spend2Output2;

impl Relation for Spend2Output2 {
    type Instance = Spend2Output2PublicInputs;

    type Witness = (
        MapMt<F, PoseidonChip<F>>,
        JubjubScalar,
        F,
        Utxo,
        Utxo,
        Utxo,
        Utxo,
        JubjubSubgroup,
        JubjubSubgroup,
    );

    fn format_instance(instance: &Self::Instance) -> Result<Vec<F>, Error> {
        Ok(vec![
            instance.root,
            instance.pk_bx,
            instance.pk_by,
            instance.new_c1,
            instance.new_c2,
            instance.nf1,
            instance.nf2,
        ])
    }

    fn circuit(
        &self,
        std_lib: &ZkStdLib,
        layouter: &mut impl Layouter<F>,
        _instance: Value<Self::Instance>,
        witness: Value<Self::Witness>,
    ) -> Result<(), Error> {
        let commit_map_val = witness.clone().map(|(m, _, _, _, _, _, _, _, _)| m);

        let sk_val = witness.clone().map(|(_, sk, _, _, _, _, _, _, _)| sk);
        let alpha_val = witness.clone().map(|(_, _, alpha, _, _, _, _, _, _)| alpha);

        let old1_val = witness.clone().map(|(_, _, _, o1, _, _, _, _, _)| o1);
        let old2_val = witness.clone().map(|(_, _, _, _, o2, _, _, _, _)| o2);
        let new1_val = witness.clone().map(|(_, _, _, _, _, n1, _, _, _)| n1);
        let new2_val = witness.clone().map(|(_, _, _, _, _, _, n2, _, _)| n2);

        let pk1_out_val = witness.clone().map(|(_, _, _, _, _, _, _, k1, _)| k1);
        let pk2_out_val = witness.clone().map(|(_, _, _, _, _, _, _, _, k2)| k2);

        let sk: AssignedScalarOfNativeCurve<Jubjub> = std_lib.jubjub().assign(layouter, sk_val)?;
        let generator = std_lib
            .jubjub()
            .assign_fixed(layouter, JubjubSubgroup::generator())?;
        let pk_sender = std_lib.jubjub().mul(layouter, &sk, &generator)?;
        let pk_sender_fields = std_lib.jubjub().as_public_input(layouter, &pk_sender)?;
        let (pk_sx, pk_sy) = (pk_sender_fields[0].clone(), pk_sender_fields[1].clone());

        let alpha_native_value = std_lib.assign(layouter, alpha_val)?;
        std_lib.assert_non_zero(layouter, &alpha_native_value)?;
        let alpha: AssignedScalarOfNativeCurve<Jubjub> =
            std_lib.jubjub().convert(layouter, &alpha_native_value)?;
        let blind = std_lib.jubjub().mul(layouter, &alpha, &generator)?;
        let pk_blinded = std_lib.jubjub().add(layouter, &pk_sender, &blind)?;
        let pk_blinded_fields = std_lib.jubjub().as_public_input(layouter, &pk_blinded)?;
        let (pk_bx, pk_by) = (pk_blinded_fields[0].clone(), pk_blinded_fields[1].clone());

        let old1_asg = assign_utxo(std_lib, layouter, &old1_val)?;
        let old2_asg = assign_utxo(std_lib, layouter, &old2_val)?;
        let new1_asg = assign_utxo(std_lib, layouter, &new1_val)?;
        let new2_asg = assign_utxo(std_lib, layouter, &new2_val)?;

        let old_c1 = compute_commitment_from_parts(std_lib, layouter, &old1_asg, &pk_sx, &pk_sy)?;
        let old_c2 = compute_commitment_from_parts(std_lib, layouter, &old2_asg, &pk_sx, &pk_sy)?;

        let mut commit_map_gadget = std_lib.map_gadget().clone();
        commit_map_gadget.init(layouter, commit_map_val)?;

        let one = std_lib.assign_fixed(layouter, F::ONE)?;

        let v1 = commit_map_gadget.get(layouter, &old_c1)?;
        let v2 = commit_map_gadget.get(layouter, &old_c2)?;
        std_lib.assert_equal(layouter, &v1, &one)?;
        std_lib.assert_equal(layouter, &v2, &one)?;

        let root = commit_map_gadget.succinct_repr();

        let nf1 = compute_nullifier(std_lib, layouter, &old_c1, &pk_sx, &pk_sy)?;
        let nf2 = compute_nullifier(std_lib, layouter, &old_c2, &pk_sx, &pk_sy)?;
        std_lib.assert_not_equal(layouter, &nf1, &nf2)?;

        let pk1_out: AssignedNativePoint<Jubjub> =
            std_lib.jubjub().assign(layouter, pk1_out_val)?;
        let pk1_fields = std_lib.jubjub().as_public_input(layouter, &pk1_out)?;
        let (pk1x, pk1y) = (pk1_fields[0].clone(), pk1_fields[1].clone());

        let pk2_out: AssignedNativePoint<Jubjub> =
            std_lib.jubjub().assign(layouter, pk2_out_val)?;
        let pk2_fields = std_lib.jubjub().as_public_input(layouter, &pk2_out)?;
        let (pk2x, pk2y) = (pk2_fields[0].clone(), pk2_fields[1].clone());

        let new_c1 = compute_commitment_from_parts(std_lib, layouter, &new1_asg, &pk1x, &pk1y)?;
        let new_c2 = compute_commitment_from_parts(std_lib, layouter, &new2_asg, &pk2x, &pk2y)?;
        std_lib.assert_not_equal(layouter, &new_c1, &new_c2)?;

        check_value_conservation_assigned(
            std_lib, layouter, &old1_asg, &old2_asg, &new1_asg, &new2_asg,
        )?;

        // Unhashed public inputs: constrain each value directly as a public input (in the same order as format_instance).
        std_lib.constrain_as_public_input(layouter, &root)?;
        std_lib.constrain_as_public_input(layouter, &pk_bx)?;
        std_lib.constrain_as_public_input(layouter, &pk_by)?;
        std_lib.constrain_as_public_input(layouter, &new_c1)?;
        std_lib.constrain_as_public_input(layouter, &new_c2)?;
        std_lib.constrain_as_public_input(layouter, &nf1)?;
        std_lib.constrain_as_public_input(layouter, &nf2)?;

        Ok(())
    }

    fn used_chips(&self) -> ZkStdLibArch {
        ZkStdLibArch {
            jubjub: true,
            poseidon: true,
            sha2_256: false,
            sha2_512: false,
            keccak_256: false,
            sha3_256: false,
            blake2b: false,
            secp256k1: false,
            bls12_381: false,
            base64: false,
            nr_pow2range_cols: 1,
            automaton: false,
        }
    }

    fn write_relation<W: std::io::Write>(&self, _writer: &mut W) -> std::io::Result<()> {
        Ok(())
    }
    fn read_relation<R: std::io::Read>(_reader: &mut R) -> std::io::Result<Self> {
        Ok(Self)
    }
}

#[derive(Clone)]
struct AssignedUtxo {
    id: AssignedNative<F>,
    amount_f: AssignedNative<F>,
    amount_big: AssignedBigUint<F>,
    randomness: AssignedNative<F>,
}

fn assign_utxo<L: Layouter<F>>(
    std_lib: &ZkStdLib,
    layouter: &mut L,
    utxo_val: &Value<Utxo>,
) -> Result<AssignedUtxo, Error> {
    let id = std_lib.assign(layouter, utxo_val.clone().map(|u| u.asset_id))?;
    let amount_f = std_lib.assign(layouter, utxo_val.clone().map(|u| F::from_u128(u.amount)))?;
    let randomness = std_lib.assign(layouter, utxo_val.clone().map(|u| u.randomness))?;
    let big = std_lib.biguint();

    let bits_f =
        std_lib.assigned_to_le_bits(layouter, &amount_f, Some(AMOUNT_BITS as usize), true)?;
    let amount_big = big.from_le_bits(layouter, &bits_f)?;

    Ok(AssignedUtxo {
        id,
        amount_f,
        amount_big,
        randomness,
    })
}

fn compute_commitment_from_parts<L: Layouter<F>>(
    std_lib: &ZkStdLib,
    layouter: &mut L,
    utxo: &AssignedUtxo,
    pk_x: &AssignedNative<F>,
    pk_y: &AssignedNative<F>,
) -> Result<AssignedNative<F>, Error> {
    let tag = std_lib.assign_fixed(layouter, F::from(UTXO_COMMIT_TAG))?;
    let zero = std_lib.assign_fixed(layouter, F::ZERO)?;
    let h1 = std_lib.poseidon(layouter, &[tag, utxo.id.clone(), utxo.amount_f.clone()])?;
    let h2 = std_lib.poseidon(
        layouter,
        &[pk_x.clone(), pk_y.clone(), utxo.randomness.clone()],
    )?;
    std_lib.poseidon(layouter, &[h1, h2, zero])
}

fn compute_nullifier<L: Layouter<F>>(
    std_lib: &ZkStdLib,
    layouter: &mut L,
    commitment: &AssignedNative<F>,
    pk_x: &AssignedNative<F>,
    pk_y: &AssignedNative<F>,
) -> Result<AssignedNative<F>, Error> {
    let tag = std_lib.assign_fixed(layouter, F::from(UTXO_NULLIFY_TAG))?;
    let zero = std_lib.assign_fixed(layouter, F::ZERO)?;
    let h = std_lib.poseidon(layouter, &[tag, commitment.clone(), pk_x.clone()])?;
    std_lib.poseidon(layouter, &[h, pk_y.clone(), zero])
}

fn check_value_conservation_assigned<L: Layouter<F>>(
    std_lib: &ZkStdLib,
    layouter: &mut L,
    in1: &AssignedUtxo,
    in2: &AssignedUtxo,
    out1: &AssignedUtxo,
    out2: &AssignedUtxo,
) -> Result<(), Error> {
    std_lib.assert_equal(layouter, &in1.id, &in2.id)?;
    std_lib.assert_equal(layouter, &in1.id, &out1.id)?;
    std_lib.assert_equal(layouter, &in1.id, &out2.id)?;

    let big = std_lib.biguint();
    let sum_in = big.add(layouter, &in1.amount_big, &in2.amount_big)?;
    let sum_out = big.add(layouter, &out1.amount_big, &out2.amount_big)?;
    big.assert_equal(layouter, &sum_in, &sum_out)
}

pub(crate) fn host_commit(id: F, amt_u128: u128, pk_x: F, pk_y: F, rand: F) -> F {
    let tag = F::from(UTXO_COMMIT_TAG);
    let amt_f = F::from_u128(amt_u128);
    let h1 = <PoseidonChip<F> as HashCPU<F, F>>::hash(&[tag, id, amt_f]);
    let h2 = <PoseidonChip<F> as HashCPU<F, F>>::hash(&[pk_x, pk_y, rand]);
    <PoseidonChip<F> as HashCPU<F, F>>::hash(&[h1, h2, F::ZERO])
}

pub(crate) fn host_nullify(commit: F, pk_x: F, pk_y: F) -> F {
    let tag = F::from(UTXO_NULLIFY_TAG);
    let h = <PoseidonChip<F> as HashCPU<F, F>>::hash(&[tag, commit, pk_x]);
    <PoseidonChip<F> as HashCPU<F, F>>::hash(&[h, pk_y, F::ZERO])
}

#[derive(Clone, Debug)]
pub(crate) struct Note {
    pub(crate) utxo: Utxo,
    pub(crate) commit: F,
    pub(crate) spent: bool,
    pub(crate) confirmed_at_root_idx: usize,
}

#[derive(Clone)]
pub(crate) struct Account {
    pub(crate) id: usize,
    pub(crate) sk: JubjubScalar,
    pub(crate) pk_point: JubjubSubgroup,
    pub(crate) pk_x: F,
    pub(crate) pk_y: F,
    pub(crate) wallet: Vec<Note>,
}

#[cfg(test)]
mod tests {
    use crate::transfer_circuit;

    use super::*;
    use midnight_circuits::instructions::map::MapCPU;
    use midnight_circuits::types::Instantiable;
    use proptest::prelude::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use std::ops::Not;
    use std::sync::OnceLock;

    // --- If your project already has a helper for SRS, use it here.
    // In the earlier rollup demo you used trusted_setup::filecoin_srs_agg(K).
    // This test assumes that exists; swap if your crate uses a different SRS provider.
    use midnight_proofs::poly::kzg::params::ParamsKZG;
    use transfer_circuit::Spend2Output2;

    type E = <S as SelfEmulation>::Engine;

    // Poseidon transcript state used by midnight_zk_stdlib
    use midnight_circuits::hash::poseidon::PoseidonState;
    use midnight_zk_stdlib::MidnightPK;

    const K_TEST: u32 = 14;

    #[derive(Clone)]
    struct Env {
        srs: ParamsKZG<E>,
        relation: Spend2Output2,
        vk: midnight_zk_stdlib::MidnightVK,
        pk: MidnightPK<Spend2Output2>,
    }

    static ENV: OnceLock<Env> = OnceLock::new();

    fn env() -> &'static Env {
        ENV.get_or_init(|| {
            // --- swap this line if needed in your repo ---
            let mut srs =
                crate::trusted_setup::filecoin_srs_agg(K_TEST).expect("SRS must load in tests");

            let relation = Spend2Output2;

            // Optional but nice if available in your crate (seen in midnight_zk_stdlib):
            // midnight_zk_stdlib::downsize_srs_for_relation(&mut srs, &relation);

            let vk = midnight_zk_stdlib::setup_vk(&srs, &relation);
            let pk = midnight_zk_stdlib::setup_pk(&relation, &vk);

            Env {
                srs,
                relation,
                vk,
                pk,
            }
        })
    }

    // Treat "reject" as: either proving fails OR verification fails.
    fn accepts(
        instance: &Spend2Output2PublicInputs,
        witness: <Spend2Output2 as midnight_zk_stdlib::Relation>::Witness,
        seed: u64,
    ) -> bool {
        let e = env();
        let mut prover_rng = ChaCha8Rng::seed_from_u64(seed ^ 0xA5A5_A5A5_A5A5_A5A5);

        let proof = match midnight_zk_stdlib::prove::<Spend2Output2, PoseidonState<F>>(
            &e.srs,
            &e.pk,
            &e.relation,
            instance,
            witness,
            &mut prover_rng,
        ) {
            Ok(p) => p,
            Err(_) => return false,
        };

        // This matches the verify API shape shown in docs.rs for midnight_zk_stdlib.
        // midnight_zk_stdlib in your codebase typically mirrors it; if not, swap for your verifier.
        midnight_zk_stdlib::verify::<Spend2Output2, PoseidonState<F>>(
            &e.srs.verifier_params(),
            &e.vk,
            instance,
            None,
            &proof,
        )
        .is_ok()
    }

    fn rejects(
        instance: &Spend2Output2PublicInputs,
        witness: <Spend2Output2 as midnight_zk_stdlib::Relation>::Witness,
        seed: u64,
    ) -> bool {
        !accepts(instance, witness, seed)
    }

    // -----------------------------
    // Host-side helpers for tests
    // -----------------------------

    fn scalar_to_field_opt(alpha: JubjubScalar) -> Option<F> {
        Option::<F>::from(F::from_bytes_le(&alpha.to_bytes()))
    }

    fn jubjub_fields(p: &JubjubSubgroup) -> (F, F) {
        let xy = AssignedNativePoint::<Jubjub>::as_public_input(p);
        (xy[0], xy[1])
    }

    fn rand_amount(rng: &mut ChaCha8Rng) -> u128 {
        // match your sim: AMOUNT_GEN_BITS <= 128 so sum of two fits comfortably
        let mask = if AMOUNT_GEN_BITS == 128 {
            u128::MAX
        } else {
            (1u128 << AMOUNT_GEN_BITS) - 1
        };
        rng.r#gen::<u128>() & mask
    }

    fn split_amount(rng: &mut ChaCha8Rng, total: u128) -> (u128, u128) {
        if total == 0 {
            (0, 0)
        } else {
            let a = rng.gen_range(0..=total);
            (a, total - a)
        }
    }

    fn make_valid_case(
        seed: u64,
    ) -> (
        Spend2Output2PublicInputs,
        <transfer_circuit::Spend2Output2 as Relation>::Witness,
    ) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        // Sender keypair
        let sk = JubjubScalar::random(&mut rng);
        let pk_sender = JubjubSubgroup::generator() * sk;
        let (pk_sx, pk_sy) = jubjub_fields(&pk_sender);

        // Non-zero alpha that is convertible to F
        let (alpha_scalar, alpha_f) = loop {
            let a = JubjubScalar::random(&mut rng);
            if a.is_zero().into() {
                continue;
            }
            if let Some(af) = scalar_to_field_opt(a) {
                if (af.is_zero().not()).into() {
                    break (a, af);
                }
            }
        };

        let pk_blinded = pk_sender + (JubjubSubgroup::generator() * alpha_scalar);
        let (pk_bx, pk_by) = jubjub_fields(&pk_blinded);

        // Asset id shared by all utxos
        let asset_id = F::random(&mut rng);

        // Old inputs
        let old1 = Utxo {
            asset_id,
            amount: rand_amount(&mut rng),
            randomness: F::random(&mut rng),
        };
        let mut old2 = Utxo {
            asset_id,
            amount: rand_amount(&mut rng),
            randomness: F::random(&mut rng),
        };
        // Ensure old1 != old2 so nf1 != nf2 almost surely and also trips the explicit inequality
        if old2.amount == old1.amount && old2.randomness == old1.randomness {
            old2.randomness = F::random(&mut rng);
        }

        let old_c1 = host_commit(old1.asset_id, old1.amount, pk_sx, pk_sy, old1.randomness);
        let old_c2 = host_commit(old2.asset_id, old2.amount, pk_sx, pk_sy, old2.randomness);

        // Commitment map contains both
        let mut commit_map = MapMt::<F, PoseidonChip<F>>::new(&F::ZERO);
        commit_map.insert(&old_c1, &F::ONE);
        commit_map.insert(&old_c2, &F::ONE);
        let root = commit_map.succinct_repr();

        // Outputs
        let k1 = JubjubScalar::random(&mut rng);
        let mut k2 = JubjubScalar::random(&mut rng);
        while k2 == k1 {
            k2 = JubjubScalar::random(&mut rng);
        }
        let pk1_out = JubjubSubgroup::generator() * k1;
        let pk2_out = JubjubSubgroup::generator() * k2;
        let (pk1x, pk1y) = jubjub_fields(&pk1_out);
        let (pk2x, pk2y) = jubjub_fields(&pk2_out);

        let total_in = old1.amount.saturating_add(old2.amount);
        let (out1_amt, out2_amt) = split_amount(&mut rng, total_in);

        let new1 = Utxo {
            asset_id,
            amount: out1_amt,
            randomness: F::random(&mut rng),
        };
        let new2 = Utxo {
            asset_id,
            amount: out2_amt,
            randomness: F::random(&mut rng),
        };

        let new_c1 = host_commit(new1.asset_id, new1.amount, pk1x, pk1y, new1.randomness);
        let mut new_c2 = host_commit(new2.asset_id, new2.amount, pk2x, pk2y, new2.randomness);

        // Ensure new_c1 != new_c2; if collision / accidental equality, tweak randomness
        if new_c2 == new_c1 {
            let tweaked = Utxo {
                randomness: F::random(&mut rng),
                ..new2.clone()
            };
            new_c2 = host_commit(
                tweaked.asset_id,
                tweaked.amount,
                pk2x,
                pk2y,
                tweaked.randomness,
            );
        }

        let nf1 = host_nullify(old_c1, pk_sx, pk_sy);
        let nf2 = host_nullify(old_c2, pk_sx, pk_sy);

        let instance = Spend2Output2PublicInputs {
            root,
            pk_bx,
            pk_by,
            new_c1,
            new_c2,
            nf1,
            nf2,
        };

        let witness = (
            commit_map, sk, alpha_f, old1, old2, new1, new2, pk1_out, pk2_out,
        );

        (instance, witness)
    }

    // -----------------------------
    // Unit tests (cheap, pure)
    // -----------------------------

    #[test]
    fn unit_domain_separation_tags_differ() {
        assert_ne!(UTXO_COMMIT_TAG, UTXO_NULLIFY_TAG);
    }

    proptest! {
        #[test]
        fn pbt_host_commit_deterministic(
            seed in any::<u64>(),
            amt in any::<u128>(),
        ) {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let id = F::random(&mut rng);
            let rand = F::random(&mut rng);
            let pk_x = F::random(&mut rng);
            let pk_y = F::random(&mut rng);

            let c1 = host_commit(id, amt, pk_x, pk_y, rand);
            let c2 = host_commit(id, amt, pk_x, pk_y, rand);
            prop_assert_eq!(c1, c2);
        }

        #[test]
        fn pbt_host_nullify_deterministic(seed in any::<u64>()) {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let commit = F::random(&mut rng);
            let pk_x = F::random(&mut rng);
            let pk_y = F::random(&mut rng);

            let n1 = host_nullify(commit, pk_x, pk_y);
            let n2 = host_nullify(commit, pk_x, pk_y);
            prop_assert_eq!(n1, n2);
        }
    }

    // -----------------------------
    // Circuit-level PBT (accept valid witnesses)
    // -----------------------------

    proptest! {
        #![proptest_config(ProptestConfig { cases: 8, ..ProptestConfig::default() })]
        #[test]
        fn pbt_valid_witness_is_accepted(seed in any::<u64>()) {
            let (instance, witness) = make_valid_case(seed);
            prop_assert!(accepts(&instance, witness, seed));
        }
    }

    // -----------------------------
    // Negative tests (each targets a safety property)
    // -----------------------------

    #[test]
    fn negative_wrong_root_is_rejected() {
        let seed = 1001;
        let (mut instance, witness) = make_valid_case(seed);
        instance.root = F::random(&mut ChaCha8Rng::seed_from_u64(9999)); // tamper
        assert!(rejects(&instance, witness, seed));
    }

    #[test]
    fn negative_missing_input_membership_is_rejected() {
        let seed = 1002;
        let (instance, mut witness) = make_valid_case(seed);

        // Remove one key by replacing map with empty one
        let mut empty = MapMt::<F, PoseidonChip<F>>::new(&F::ZERO);
        // keep types the same; wipe membership
        witness.0 = empty;

        assert!(rejects(&instance, witness, seed));
    }

    #[test]
    fn negative_alpha_zero_is_rejected() {
        let seed = 1003;
        let (instance, mut witness) = make_valid_case(seed);
        witness.2 = F::ZERO; // alpha_f = 0 violates assert_non_zero
        assert!(rejects(&instance, witness, seed));
    }

    #[test]
    fn negative_asset_id_mismatch_is_rejected() {
        let seed = 1004;
        let (instance, mut witness) = make_valid_case(seed);

        // old2 asset differs => check_value_conservation_assigned fails on id equality
        let mut old2 = witness.4.clone();
        old2.asset_id = F::random(&mut ChaCha8Rng::seed_from_u64(55));
        witness.4 = old2;

        assert!(rejects(&instance, witness, seed));
    }

    #[test]
    fn negative_value_conservation_violation_is_rejected() {
        let seed = 1005;
        let (instance, mut witness) = make_valid_case(seed);

        // Increase out1 amount by 1 (wrapping-safe change) => sum_out != sum_in
        let mut new1 = witness.5.clone();
        new1.amount = new1.amount.saturating_add(1);
        witness.5 = new1;

        assert!(rejects(&instance, witness, seed));
    }

    #[test]
    fn negative_duplicate_inputs_nf1_eq_nf2_is_rejected() {
        let seed = 1006;
        let (instance, mut witness) = make_valid_case(seed);

        // Force old2 = old1 => old_c2 = old_c1 => nf2 = nf1, violates assert_not_equal(nf1,nf2)
        witness.4 = witness.3.clone();

        assert!(rejects(&instance, witness, seed));
    }

    #[test]
    fn negative_duplicate_outputs_new_c1_eq_new_c2_is_rejected() {
        let seed = 1007;
        let (instance, mut witness) = make_valid_case(seed);

        // Force new2 = new1 AND pk2_out = pk1_out so commitments match
        witness.6 = witness.5.clone();
        witness.8 = witness.7; // pk2_out = pk1_out

        assert!(rejects(&instance, witness, seed));
    }

    #[test]
    fn negative_public_input_tamper_pk_blinded_is_rejected() {
        let seed = 1008;
        let (mut instance, witness) = make_valid_case(seed);

        instance.pk_bx = F::random(&mut ChaCha8Rng::seed_from_u64(123));
        // (pk_by unchanged) still should fail because circuit constrains both coords
        assert!(rejects(&instance, witness, seed));
    }

    #[test]
    fn negative_public_input_tamper_output_commitment_is_rejected() {
        let seed = 1009;
        let (mut instance, witness) = make_valid_case(seed);

        instance.new_c1 = F::random(&mut ChaCha8Rng::seed_from_u64(321));
        assert!(rejects(&instance, witness, seed));
    }

    #[test]
    fn negative_public_input_tamper_nullifier_is_rejected() {
        let seed = 1010;
        let (mut instance, witness) = make_valid_case(seed);

        instance.nf2 = F::random(&mut ChaCha8Rng::seed_from_u64(777));
        assert!(rejects(&instance, witness, seed));
    }
}
