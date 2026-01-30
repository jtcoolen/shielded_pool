use ff::{Field, PrimeField};
use group::Group;

use midnight_circuits::{
    biguint::AssignedBigUint,
    compact_std_lib::{Relation, ZkStdLib, ZkStdLibArch},
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
            sha256: false,
            sha512: false,
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
