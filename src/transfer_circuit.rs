use ff::{Field, PrimeField};
use group::Group;

use midnight_circuits::{
    biguint::AssignedBigUint,
    compact_std_lib::{Relation, ZkStdLib, ZkStdLibArch},
    ecc::native::AssignedScalarOfNativeCurve,
    hash::poseidon::PoseidonChip,
    instructions::{
        ArithInstructions, AssertionInstructions, AssignmentInstructions, BinaryInstructions,
        ControlFlowInstructions, ConversionInstructions, DecompositionInstructions,
        EccInstructions, PublicInputInstructions, ZeroInstructions, hash::HashCPU,
        map::MapInstructions,
    },
    map::cpu::MapMt,
    types::{AssignedBit, AssignedNative, AssignedNativePoint},
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

pub(crate) const SWAP_TERMS_TAG: u64 = 0x0003;

const VTO_BITS: u32 = 64;

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
    // --- swap extension
    pub sterms: F,
    pub swapcm: F,
    pub vto: F,
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
            sterms: F::ZERO,
            swapcm: F::ZERO,
            vto: F::ZERO,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Utxo {
    pub asset_id: F,
    pub amount: u128,
    pub randomness: F,
}

#[derive(Clone, Debug)]
pub struct SwapTermsWitness {
    /// Asset id spent by pk_a (A -> B leg).
    pub asset_id_a: F,
    /// Asset id spent by pk_b (B -> A leg).
    pub asset_id_b: F,
    pub pk_a: JubjubSubgroup,
    pub pk_b: JubjubSubgroup,
    pub amt_a_to_b: u128,
    pub amt_b_to_a: u128,
}

impl Default for SwapTermsWitness {
    fn default() -> Self {
        Self {
            asset_id_a: F::ZERO,
            asset_id_b: F::ZERO,
            pk_a: JubjubSubgroup::identity(),
            pk_b: JubjubSubgroup::identity(),
            amt_a_to_b: 0,
            amt_b_to_a: 0,
        }
    }
}

/// Canonical witness tuple for this relation.
/// Keeping an alias makes the `Value` plumbing and refactors significantly easier to read.
pub type Spend2Output2Witness = (
    MapMt<F, PoseidonChip<F>>,
    JubjubScalar,
    F,
    Utxo,
    Utxo,
    Utxo,
    Utxo,
    JubjubSubgroup,
    JubjubSubgroup,
    SwapTermsWitness,
);

#[derive(Clone, Default)]
pub struct Spend2Output2;

impl Relation for Spend2Output2 {
    type Instance = Spend2Output2PublicInputs;
    type Witness = Spend2Output2Witness;

    fn format_instance(instance: &Self::Instance) -> Result<Vec<F>, Error> {
        Ok(vec![
            instance.root,
            instance.pk_bx,
            instance.pk_by,
            instance.new_c1,
            instance.new_c2,
            instance.nf1,
            instance.nf2,
            instance.sterms,
            instance.swapcm,
            instance.vto,
        ])
    }

    fn circuit(
        &self,
        std_lib: &ZkStdLib,
        layouter: &mut impl Layouter<F>,
        instance: Value<Self::Instance>,
        witness: Value<Self::Witness>,
    ) -> Result<(), Error> {
        let w = split_witness(witness);

        // ---- Sender keys ----
        let sender = derive_sender_and_blinded_keys(std_lib, layouter, w.sk, w.alpha)?;

        // ---- UTXOs ----
        let (old1, old2, new1, new2) =
            assign_all_utxos(std_lib, layouter, w.old1, w.old2, w.new1, w.new2)?;

        // ---- Inputs: commitments + membership + nullifiers ----
        let (old_c1, old_c2) = compute_input_commitments(
            std_lib,
            layouter,
            &old1,
            &old2,
            &sender.pk_sx,
            &sender.pk_sy,
        )?;

        let root =
            verify_membership_and_compute_root(std_lib, layouter, w.commit_map, &old_c1, &old_c2)?;

        let (nf1, nf2) = compute_distinct_nullifiers(
            std_lib,
            layouter,
            &old_c1,
            &old_c2,
            &sender.pk_sx,
            &sender.pk_sy,
        )?;

        // ---- Outputs: recipient keys, commitments, distinctness ----
        let out1_pk = assign_point_xy_from_value(std_lib, layouter, w.pk1_out)?;
        let out2_pk = assign_point_xy_from_value(std_lib, layouter, w.pk2_out)?;

        let (new_c1, new_c2) = compute_output_commitments_and_assert_distinct(
            std_lib, layouter, &new1, &new2, &out1_pk, &out2_pk,
        )?;

        // ---- Value conservation (always enforced) ----
        assert_value_conservation(std_lib, layouter, &old1, &old2, &new1, &new2)?;

        // ---- Swap public inputs (always constrained as public inputs) ----
        let swap_pi = assign_swap_public_inputs(std_lib, layouter, instance)?;

        // ---- Constrain public inputs in the exact order of `format_instance` ----
        constrain_public_inputs(
            std_lib,
            layouter,
            &PublicCore {
                root,
                pk_bx: sender.pk_bx.clone(),
                pk_by: sender.pk_by.clone(),
                new_c1: new_c1.clone(),
                new_c2: new_c2.clone(),
                nf1: nf1.clone(),
                nf2: nf2.clone(),
            },
            &swap_pi,
        )?;

        // ---- Swap/transfer handling (gated by sterms != 0) ----
        let gates = swap_gates(std_lib, layouter, &swap_pi.sterms)?;
        // Make "transfer" mode (no swap) robust: opaque swap PIs must be zero.
        enforce_transfer_mode_defaults(std_lib, layouter, &gates, &swap_pi)?;
        // in swap mode, make swap legs "self-validating" before rollup:
        // - vto must be a u64
        // - swapcm must bind (sterms, vto) as H2(sterms, vto)
        enforce_swap_mode_public_inputs(std_lib, layouter, &gates, &swap_pi)?;

        // Swap terms witness is still provided even for transfers; swap constraints are gated.
        let terms = assign_swap_terms(std_lib, layouter, w.terms)?;
        enforce_swap_mode(
            std_lib,
            layouter,
            &gates,
            &swap_pi.sterms,
            &old1.id,
            &sender,
            &new1,
            &out1_pk,
            &out2_pk,
            &terms,
        )?;

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

// -----------------------------
// Witness splitting (functional)
// -----------------------------

#[derive(Clone)]
struct WitnessParts {
    commit_map: Value<MapMt<F, PoseidonChip<F>>>,
    sk: Value<JubjubScalar>,
    alpha: Value<F>,
    old1: Value<Utxo>,
    old2: Value<Utxo>,
    new1: Value<Utxo>,
    new2: Value<Utxo>,
    pk1_out: Value<JubjubSubgroup>,
    pk2_out: Value<JubjubSubgroup>,
    terms: Value<SwapTermsWitness>,
}

fn split_witness(w: Value<Spend2Output2Witness>) -> WitnessParts {
    WitnessParts {
        commit_map: w.clone().map(|(m, _, _, _, _, _, _, _, _, _)| m),
        sk: w.clone().map(|(_, sk, _, _, _, _, _, _, _, _)| sk),
        alpha: w.clone().map(|(_, _, a, _, _, _, _, _, _, _)| a),
        old1: w.clone().map(|(_, _, _, o1, _, _, _, _, _, _)| o1),
        old2: w.clone().map(|(_, _, _, _, o2, _, _, _, _, _)| o2),
        new1: w.clone().map(|(_, _, _, _, _, n1, _, _, _, _)| n1),
        new2: w.clone().map(|(_, _, _, _, _, _, n2, _, _, _)| n2),
        pk1_out: w.clone().map(|(_, _, _, _, _, _, _, k1, _, _)| k1),
        pk2_out: w.clone().map(|(_, _, _, _, _, _, _, _, k2, _)| k2),
        terms: w.map(|(_, _, _, _, _, _, _, _, _, t)| t),
    }
}

// -----------------------------
// Assigned domain types
// -----------------------------

#[derive(Clone)]
struct AssignedUtxo {
    id: AssignedNative<F>,
    amount_f: AssignedNative<F>,
    amount_big: AssignedBigUint<F>,
    randomness: AssignedNative<F>,
}

#[derive(Clone)]
struct AssignedPointXY {
    point: AssignedNativePoint<Jubjub>,
    x: AssignedNative<F>,
    y: AssignedNative<F>,
}

#[derive(Clone)]
struct SenderKeys {
    pk_sender: AssignedNativePoint<Jubjub>,
    pk_sx: AssignedNative<F>,
    pk_sy: AssignedNative<F>,

    pk_blinded: AssignedNativePoint<Jubjub>,
    pk_bx: AssignedNative<F>,
    pk_by: AssignedNative<F>,
}

#[derive(Clone)]
struct SwapPublicInputsAssigned {
    sterms: AssignedNative<F>,
    swapcm: AssignedNative<F>,
    vto: AssignedNative<F>,
}

#[derive(Clone)]
struct SwapGates {
    is_swap: AssignedBit<F>,
    not_swap: AssignedBit<F>,
}

#[derive(Clone)]
struct AssignedSwapTerms {
    asset_id_a: AssignedNative<F>,
    asset_id_b: AssignedNative<F>,
    pk_a: AssignedPointXY,
    pk_b: AssignedPointXY,
    amt_a_to_b: AssignedNative<F>,
    amt_b_to_a: AssignedNative<F>,
}

#[derive(Clone)]
struct PublicCore {
    root: AssignedNative<F>,
    pk_bx: AssignedNative<F>,
    pk_by: AssignedNative<F>,
    new_c1: AssignedNative<F>,
    new_c2: AssignedNative<F>,
    nf1: AssignedNative<F>,
    nf2: AssignedNative<F>,
}

// -----------------------------
// Core circuit helpers
// -----------------------------

fn derive_sender_and_blinded_keys<L: Layouter<F>>(
    std_lib: &ZkStdLib,
    layouter: &mut L,
    sk_val: Value<JubjubScalar>,
    alpha_val: Value<F>,
) -> Result<SenderKeys, Error> {
    let sk: AssignedScalarOfNativeCurve<Jubjub> = std_lib.jubjub().assign(layouter, sk_val)?;
    let generator = std_lib
        .jubjub()
        .assign_fixed(layouter, JubjubSubgroup::generator())?;

    let pk_sender = std_lib.jubjub().mul(layouter, &sk, &generator)?;
    let (pk_sx, pk_sy) = point_to_xy(std_lib, layouter, &pk_sender)?;

    let alpha_native = std_lib.assign(layouter, alpha_val)?;
    std_lib.assert_non_zero(layouter, &alpha_native)?;
    let alpha: AssignedScalarOfNativeCurve<Jubjub> =
        std_lib.jubjub().convert(layouter, &alpha_native)?;

    let blind = std_lib.jubjub().mul(layouter, &alpha, &generator)?;
    let pk_blinded = std_lib.jubjub().add(layouter, &pk_sender, &blind)?;
    let (pk_bx, pk_by) = point_to_xy(std_lib, layouter, &pk_blinded)?;

    Ok(SenderKeys {
        pk_sender,
        pk_sx,
        pk_sy,
        pk_blinded,
        pk_bx,
        pk_by,
    })
}

fn assign_all_utxos<L: Layouter<F>>(
    std_lib: &ZkStdLib,
    layouter: &mut L,
    old1: Value<Utxo>,
    old2: Value<Utxo>,
    new1: Value<Utxo>,
    new2: Value<Utxo>,
) -> Result<(AssignedUtxo, AssignedUtxo, AssignedUtxo, AssignedUtxo), Error> {
    let o1 = assign_utxo(std_lib, layouter, &old1)?;
    let o2 = assign_utxo(std_lib, layouter, &old2)?;
    let n1 = assign_utxo(std_lib, layouter, &new1)?;
    let n2 = assign_utxo(std_lib, layouter, &new2)?;
    Ok((o1, o2, n1, n2))
}

fn compute_input_commitments<L: Layouter<F>>(
    std_lib: &ZkStdLib,
    layouter: &mut L,
    old1: &AssignedUtxo,
    old2: &AssignedUtxo,
    sender_x: &AssignedNative<F>,
    sender_y: &AssignedNative<F>,
) -> Result<(AssignedNative<F>, AssignedNative<F>), Error> {
    let c1 = compute_commitment_from_parts(std_lib, layouter, old1, sender_x, sender_y)?;
    let c2 = compute_commitment_from_parts(std_lib, layouter, old2, sender_x, sender_y)?;
    Ok((c1, c2))
}

fn verify_membership_and_compute_root<L: Layouter<F>>(
    std_lib: &ZkStdLib,
    layouter: &mut L,
    commit_map_val: Value<MapMt<F, PoseidonChip<F>>>,
    old_c1: &AssignedNative<F>,
    old_c2: &AssignedNative<F>,
) -> Result<AssignedNative<F>, Error> {
    let mut map = std_lib.map_gadget().clone();
    map.init(layouter, commit_map_val)?;

    let one = std_lib.assign_fixed(layouter, F::ONE)?;
    let v1 = map.get(layouter, old_c1)?;
    let v2 = map.get(layouter, old_c2)?;

    std_lib.assert_equal(layouter, &v1, &one)?;
    std_lib.assert_equal(layouter, &v2, &one)?;

    Ok(map.succinct_repr())
}

fn compute_distinct_nullifiers<L: Layouter<F>>(
    std_lib: &ZkStdLib,
    layouter: &mut L,
    old_c1: &AssignedNative<F>,
    old_c2: &AssignedNative<F>,
    sender_x: &AssignedNative<F>,
    sender_y: &AssignedNative<F>,
) -> Result<(AssignedNative<F>, AssignedNative<F>), Error> {
    let nf1 = compute_nullifier(std_lib, layouter, old_c1, sender_x, sender_y)?;
    let nf2 = compute_nullifier(std_lib, layouter, old_c2, sender_x, sender_y)?;
    std_lib.assert_not_equal(layouter, &nf1, &nf2)?;
    Ok((nf1, nf2))
}

fn assign_point_xy_from_value<L: Layouter<F>>(
    std_lib: &ZkStdLib,
    layouter: &mut L,
    p_val: Value<JubjubSubgroup>,
) -> Result<AssignedPointXY, Error> {
    let p: AssignedNativePoint<Jubjub> = std_lib.jubjub().assign(layouter, p_val)?;
    let (x, y) = point_to_xy(std_lib, layouter, &p)?;
    Ok(AssignedPointXY { point: p, x, y })
}

fn compute_output_commitments_and_assert_distinct<L: Layouter<F>>(
    std_lib: &ZkStdLib,
    layouter: &mut L,
    out1: &AssignedUtxo,
    out2: &AssignedUtxo,
    pk1: &AssignedPointXY,
    pk2: &AssignedPointXY,
) -> Result<(AssignedNative<F>, AssignedNative<F>), Error> {
    let c1 = compute_commitment_from_parts(std_lib, layouter, out1, &pk1.x, &pk1.y)?;
    let c2 = compute_commitment_from_parts(std_lib, layouter, out2, &pk2.x, &pk2.y)?;
    std_lib.assert_not_equal(layouter, &c1, &c2)?;
    Ok((c1, c2))
}

fn assert_value_conservation<L: Layouter<F>>(
    std_lib: &ZkStdLib,
    layouter: &mut L,
    in1: &AssignedUtxo,
    in2: &AssignedUtxo,
    out1: &AssignedUtxo,
    out2: &AssignedUtxo,
) -> Result<(), Error> {
    check_value_conservation_assigned(std_lib, layouter, in1, in2, out1, out2)
}

fn assign_swap_public_inputs<L: Layouter<F>>(
    std_lib: &ZkStdLib,
    layouter: &mut L,
    instance: Value<Spend2Output2PublicInputs>,
) -> Result<SwapPublicInputsAssigned, Error> {
    let sterms: AssignedNative<F> = std_lib.assign(layouter, instance.map(|i| i.sterms))?;
    let swapcm: AssignedNative<F> = std_lib.assign(layouter, instance.map(|i| i.swapcm))?;
    let vto: AssignedNative<F> = std_lib.assign(layouter, instance.map(|i| i.vto))?;
    Ok(SwapPublicInputsAssigned {
        sterms,
        swapcm,
        vto,
    })
}

fn constrain_public_inputs<L: Layouter<F>>(
    std_lib: &ZkStdLib,
    layouter: &mut L,
    core: &PublicCore,
    swap: &SwapPublicInputsAssigned,
) -> Result<(), Error> {
    // Must match `format_instance` ordering exactly.
    std_lib.constrain_as_public_input(layouter, &core.root)?;
    std_lib.constrain_as_public_input(layouter, &core.pk_bx)?;
    std_lib.constrain_as_public_input(layouter, &core.pk_by)?;
    std_lib.constrain_as_public_input(layouter, &core.new_c1)?;
    std_lib.constrain_as_public_input(layouter, &core.new_c2)?;
    std_lib.constrain_as_public_input(layouter, &core.nf1)?;
    std_lib.constrain_as_public_input(layouter, &core.nf2)?;
    std_lib.constrain_as_public_input(layouter, &swap.sterms)?;
    std_lib.constrain_as_public_input(layouter, &swap.swapcm)?;
    std_lib.constrain_as_public_input(layouter, &swap.vto)?;
    Ok(())
}

// -----------------------------
// Swap / transfer handling
// -----------------------------

fn swap_gates<L: Layouter<F>>(
    std_lib: &ZkStdLib,
    layouter: &mut L,
    sterms: &AssignedNative<F>,
) -> Result<SwapGates, Error> {
    let sterms_is_zero = std_lib.is_zero(layouter, sterms)?;
    let is_swap = std_lib.not(layouter, &sterms_is_zero)?;
    let not_swap = std_lib.not(layouter, &is_swap)?;
    Ok(SwapGates { is_swap, not_swap })
}

/// Robust “transfer mode” defaults:
/// If `sterms == 0` (not a swap), require the opaque swap public inputs to be zero too.
///
/// This prevents “malleable” instances where transfer proofs carry unrelated non-zero swap fields.
fn enforce_transfer_mode_defaults<L: Layouter<F>>(
    std_lib: &ZkStdLib,
    layouter: &mut L,
    gates: &SwapGates,
    swap: &SwapPublicInputsAssigned,
) -> Result<(), Error> {
    assert_zero_when_not_swap(std_lib, layouter, gates, &swap.swapcm)?;
    assert_zero_when_not_swap(std_lib, layouter, gates, &swap.vto)?;
    Ok(())
}

/// Swap-mode self-validation:
/// If `sterms != 0` then require:
///   - vto is a u64 (fits in 64 bits)
///   - swapcm == H2(sterms, vto)  (Poseidon over [sterms, vto])
fn enforce_swap_mode_public_inputs<L: Layouter<F>>(
    std_lib: &ZkStdLib,
    layouter: &mut L,
    gates: &SwapGates,
    swap: &SwapPublicInputsAssigned,
) -> Result<(), Error> {
    // Range-check vto into 64 bits (u64). This is cheap and makes vto well-typed.
    // We don't need the bits; we only need the constraints.
    let _vto_bits =
        std_lib.assigned_to_le_bits(layouter, &swap.vto, Some(VTO_BITS as usize), true)?;

    // swapcm_expected := H2(sterms, vto)
    // (Assumes the rollup leaf uses the same H2 definition.)
    let swapcm_expected = std_lib.poseidon(layouter, &[swap.sterms.clone(), swap.vto.clone()])?;

    // Enforce only in swap mode (sterms != 0).
    assert_eq_when_swap(std_lib, layouter, gates, &swap.swapcm, &swapcm_expected)?;
    Ok(())
}

fn assign_swap_terms<L: Layouter<F>>(
    std_lib: &ZkStdLib,
    layouter: &mut L,
    terms_val: Value<SwapTermsWitness>,
) -> Result<AssignedSwapTerms, Error> {
    let asset_id_a = std_lib.assign(layouter, terms_val.clone().map(|t| t.asset_id_a))?;
    let asset_id_b = std_lib.assign(layouter, terms_val.clone().map(|t| t.asset_id_b))?;

    let pk_a = assign_point_xy_from_value(std_lib, layouter, terms_val.clone().map(|t| t.pk_a))?;
    let pk_b = assign_point_xy_from_value(std_lib, layouter, terms_val.clone().map(|t| t.pk_b))?;

    let amt_a_to_b = std_lib.assign(
        layouter,
        terms_val.clone().map(|t| F::from_u128(t.amt_a_to_b)),
    )?;
    let amt_b_to_a = std_lib.assign(layouter, terms_val.map(|t| F::from_u128(t.amt_b_to_a)))?;

    Ok(AssignedSwapTerms {
        asset_id_a,
        asset_id_b,
        pk_a,
        pk_b,
        amt_a_to_b,
        amt_b_to_a,
    })
}

#[allow(clippy::too_many_arguments)]
fn enforce_swap_mode<L: Layouter<F>>(
    std_lib: &ZkStdLib,
    layouter: &mut L,
    gates: &SwapGates,
    sterms: &AssignedNative<F>,
    asset_id: &AssignedNative<F>,
    sender: &SenderKeys,
    new1: &AssignedUtxo,
    out1_pk: &AssignedPointXY,
    out2_pk: &AssignedPointXY,
    terms: &AssignedSwapTerms,
) -> Result<(), Error> {
    // If swap: pk_a != pk_b (avoid degenerate "swap with self" and ambiguous direction).
    let pk_a_eq_pk_b = points_equal(std_lib, layouter, &terms.pk_a, &terms.pk_b)?;
    let pk_a_ne_pk_b = std_lib.not(layouter, &pk_a_eq_pk_b)?;
    assert_when_swap(std_lib, layouter, gates, pk_a_ne_pk_b)?;

    // If swap: sterms must bind to full terms (BOTH assets + both pk + both amounts).
    let sterms_expected = compute_sterms_expected(std_lib, layouter, terms)?;
    assert_eq_when_swap(std_lib, layouter, gates, sterms, &sterms_expected)?;

    // If swap: sender must be pk_a or pk_b.
    let sender_is_a = points_equal_xy(
        std_lib,
        layouter,
        &sender.pk_sx,
        &sender.pk_sy,
        &terms.pk_a.x,
        &terms.pk_a.y,
    )?;
    let sender_is_b = points_equal_xy(
        std_lib,
        layouter,
        &sender.pk_sx,
        &sender.pk_sy,
        &terms.pk_b.x,
        &terms.pk_b.y,
    )?;
    let sender_ok = std_lib.or(layouter, &[sender_is_a.clone(), sender_is_b.clone()])?;
    assert_when_swap(std_lib, layouter, gates, sender_ok)?;

    // Choose expected counterparty and amount based on which side the sender is.
    // If sender_is_a, counterparty is pk_b and amount is amt_a_to_b; else counterparty is pk_a and amount is amt_b_to_a.
    let exp_pk1x = std_lib.select(layouter, &sender_is_a, &terms.pk_b.x, &terms.pk_a.x)?;
    let exp_pk1y = std_lib.select(layouter, &sender_is_a, &terms.pk_b.y, &terms.pk_a.y)?;
    let exp_amt1 = std_lib.select(layouter, &sender_is_a, &terms.amt_a_to_b, &terms.amt_b_to_a)?;
    // Multi-asset swap: this leg's spent asset id must match the sender's side in the terms.
    let exp_asset_id =
        std_lib.select(layouter, &sender_is_a, &terms.asset_id_a, &terms.asset_id_b)?;
    assert_eq_when_swap(std_lib, layouter, gates, asset_id, &exp_asset_id)?;

    // If swap => out1 recipient + amount match expectation
    assert_eq_when_swap(std_lib, layouter, gates, &out1_pk.x, &exp_pk1x)?;
    assert_eq_when_swap(std_lib, layouter, gates, &out1_pk.y, &exp_pk1y)?;
    assert_eq_when_swap(std_lib, layouter, gates, &new1.amount_f, &exp_amt1)?;

    // If swap => out2 recipient is sender (change)
    assert_eq_when_swap(std_lib, layouter, gates, &out2_pk.x, &sender.pk_sx)?;
    assert_eq_when_swap(std_lib, layouter, gates, &out2_pk.y, &sender.pk_sy)?;

    Ok(())
}

fn compute_sterms_expected<L: Layouter<F>>(
    std_lib: &ZkStdLib,
    layouter: &mut L,
    terms: &AssignedSwapTerms,
) -> Result<AssignedNative<F>, Error> {
    let tag = std_lib.assign_fixed(layouter, F::from(SWAP_TERMS_TAG))?;
    std_lib.poseidon(
        layouter,
        &[
            tag,
            terms.asset_id_a.clone(),
            terms.asset_id_b.clone(),
            terms.pk_a.x.clone(),
            terms.pk_a.y.clone(),
            terms.pk_b.x.clone(),
            terms.pk_b.y.clone(),
            terms.amt_a_to_b.clone(),
            terms.amt_b_to_a.clone(),
        ],
    )
}

// -----------------------------
// Boolean / gating primitives
// -----------------------------

fn is_eq_native<L: Layouter<F>>(
    std_lib: &ZkStdLib,
    layouter: &mut L,
    a: &AssignedNative<F>,
    b: &AssignedNative<F>,
) -> Result<AssignedBit<F>, Error> {
    let d = std_lib.sub(layouter, a, b)?;
    std_lib.is_zero(layouter, &d)
}

fn points_equal<L: Layouter<F>>(
    std_lib: &ZkStdLib,
    layouter: &mut L,
    p: &AssignedPointXY,
    q: &AssignedPointXY,
) -> Result<AssignedBit<F>, Error> {
    points_equal_xy(std_lib, layouter, &p.x, &p.y, &q.x, &q.y)
}

fn points_equal_xy<L: Layouter<F>>(
    std_lib: &ZkStdLib,
    layouter: &mut L,
    px: &AssignedNative<F>,
    py: &AssignedNative<F>,
    qx: &AssignedNative<F>,
    qy: &AssignedNative<F>,
) -> Result<AssignedBit<F>, Error> {
    let x_eq = is_eq_native(std_lib, layouter, px, qx)?;
    let y_eq = is_eq_native(std_lib, layouter, py, qy)?;
    std_lib.and(layouter, &[x_eq, y_eq])
}

fn assert_when_swap<L: Layouter<F>>(
    std_lib: &ZkStdLib,
    layouter: &mut L,
    gates: &SwapGates,
    pred: AssignedBit<F>,
) -> Result<(), Error> {
    // ok := (not_swap) OR pred
    let ok = std_lib.or(layouter, &[gates.not_swap.clone(), pred])?;
    std_lib.assert_true(layouter, &ok)
}

fn assert_eq_when_swap<L: Layouter<F>>(
    std_lib: &ZkStdLib,
    layouter: &mut L,
    gates: &SwapGates,
    a: &AssignedNative<F>,
    b: &AssignedNative<F>,
) -> Result<(), Error> {
    let eq = is_eq_native(std_lib, layouter, a, b)?;
    assert_when_swap(std_lib, layouter, gates, eq)
}

fn assert_zero_when_not_swap<L: Layouter<F>>(
    std_lib: &ZkStdLib,
    layouter: &mut L,
    gates: &SwapGates,
    v: &AssignedNative<F>,
) -> Result<(), Error> {
    // ok := is_swap OR (v == 0)
    let v_is_zero = std_lib.is_zero(layouter, v)?;
    let ok = std_lib.or(layouter, &[gates.is_swap.clone(), v_is_zero])?;
    std_lib.assert_true(layouter, &ok)
}

// -----------------------------
// Low-level primitives (UTXO, hashing, etc.)
// -----------------------------

fn assign_utxo<L: Layouter<F>>(
    std_lib: &ZkStdLib,
    layouter: &mut L,
    utxo_val: &Value<Utxo>,
) -> Result<AssignedUtxo, Error> {
    let id = std_lib.assign(layouter, utxo_val.clone().map(|u| u.asset_id))?;
    let amount_f = std_lib.assign(layouter, utxo_val.clone().map(|u| F::from_u128(u.amount)))?;
    let randomness = std_lib.assign(layouter, utxo_val.clone().map(|u| u.randomness))?;

    // Range-check amount into a BigUint via bit-decomposition.
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

fn point_to_xy<L: Layouter<F>>(
    std_lib: &ZkStdLib,
    layouter: &mut L,
    p: &AssignedNativePoint<Jubjub>,
) -> Result<(AssignedNative<F>, AssignedNative<F>), Error> {
    let xy = std_lib.jubjub().as_public_input(layouter, p)?;
    Ok((xy[0].clone(), xy[1].clone()))
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

// -----------------------------
// Host-side helpers
// -----------------------------

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

pub(crate) fn host_swap_terms(
    asset_id_a: F,
    asset_id_b: F,
    pk_ax: F,
    pk_ay: F,
    pk_bx: F,
    pk_by: F,
    amt_a_to_b: u128,
    amt_b_to_a: u128,
) -> F {
    let tag = F::from(SWAP_TERMS_TAG);
    let a2b = F::from_u128(amt_a_to_b);
    let b2a = F::from_u128(amt_b_to_a);
    <PoseidonChip<F> as HashCPU<F, F>>::hash(&[
        tag, asset_id_a, asset_id_b, pk_ax, pk_ay, pk_bx, pk_by, a2b, b2a,
    ])
}

pub(crate) fn host_swapcm(sterms: F, vto: F) -> F {
    // H2(sterms, vto)
    <PoseidonChip<F> as HashCPU<F, F>>::hash(&[sterms, vto])
}

// -----------------------------
// Simulation-only structs (unchanged)
// -----------------------------

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
    use std::ops::Add;
    use std::ops::Not;
    use std::sync::OnceLock;

    // --- If your project already has a helper for SRS, use it here.
    // In the earlier rollup demo you used trusted_setup::filecoin_srs_agg(K).
    // This test assumes that exists; swap if your crate uses a different SRS provider.
    use midnight_proofs::poly::kzg::params::ParamsKZG;
    use transfer_circuit::Spend2Output2;

    type E = <S as SelfEmulation>::Engine;

    // Poseidon transcript state used by compact_std_lib
    use midnight_circuits::compact_std_lib::{self, MidnightPK};
    use midnight_circuits::hash::poseidon::PoseidonState;

    const K_TEST: u32 = 14;

    #[derive(Clone)]
    struct Env {
        srs: ParamsKZG<E>,
        relation: Spend2Output2,
        vk: midnight_circuits::compact_std_lib::MidnightVK,
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
            // compact_std_lib::downsize_srs_for_relation(&mut srs, &relation);

            let vk = compact_std_lib::setup_vk(&srs, &relation);
            let pk = compact_std_lib::setup_pk(&relation, &vk);

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
        witness: <Spend2Output2 as midnight_circuits::compact_std_lib::Relation>::Witness,
        seed: u64,
    ) -> bool {
        let e = env();
        let mut prover_rng = ChaCha8Rng::seed_from_u64(seed ^ 0xA5A5_A5A5_A5A5_A5A5);

        let proof = match compact_std_lib::prove::<Spend2Output2, PoseidonState<F>>(
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

        compact_std_lib::verify::<Spend2Output2, PoseidonState<F>>(
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
        witness: <Spend2Output2 as midnight_circuits::compact_std_lib::Relation>::Witness,
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

    fn make_valid_transfer_case(
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

        // Transfer mode: enforce swap public inputs are all zero (matches circuit hardening).
        let instance = Spend2Output2PublicInputs {
            root,
            pk_bx,
            pk_by,
            new_c1,
            new_c2,
            nf1,
            nf2,
            sterms: F::ZERO,
            swapcm: F::ZERO,
            vto: F::ZERO,
        };

        let witness = (
            commit_map,
            sk,
            alpha_f,
            old1,
            old2,
            new1,
            new2,
            pk1_out,
            pk2_out,
            SwapTermsWitness::default(),
        );

        (instance, witness)
    }

    fn make_valid_swap_case(
        seed: u64,
    ) -> (
        Spend2Output2PublicInputs,
        <transfer_circuit::Spend2Output2 as Relation>::Witness,
    ) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0xBEEF_F00D);

        // Sender keypair (also pk_a)
        let sk = JubjubScalar::random(&mut rng);
        let pk_sender = JubjubSubgroup::generator() * sk;
        let (pk_sx, pk_sy) = jubjub_fields(&pk_sender);

        // Non-zero alpha convertible to F
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

        // Counterparty key (pk_b), distinct from sender.
        let mut sk_b = JubjubScalar::random(&mut rng);
        while sk_b == sk {
            sk_b = JubjubScalar::random(&mut rng);
        }
        let pk_b = JubjubSubgroup::generator() * sk_b;
        let (pk_bx_t, pk_by_t) = jubjub_fields(&pk_b);

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
        if old2.amount == old1.amount && old2.randomness == old1.randomness {
            old2.randomness = F::random(&mut rng);
        }

        let old_c1 = host_commit(old1.asset_id, old1.amount, pk_sx, pk_sy, old1.randomness);
        let old_c2 = host_commit(old2.asset_id, old2.amount, pk_sx, pk_sy, old2.randomness);

        let mut commit_map = MapMt::<F, PoseidonChip<F>>::new(&F::ZERO);
        commit_map.insert(&old_c1, &F::ONE);
        commit_map.insert(&old_c2, &F::ONE);
        let root = commit_map.succinct_repr();

        let total_in = old1.amount.saturating_add(old2.amount);
        let (amt_a_to_b, change_amt) = split_amount(&mut rng, total_in);

        // Ensure swap amount isn't degenerate too often (optional); allow 0 as valid but try not to.
        let amt_a_to_b = if total_in > 0 && amt_a_to_b == 0 {
            1
        } else {
            amt_a_to_b
        };
        let change_amt = total_in.saturating_sub(amt_a_to_b);

        // Swap outputs must be:
        // - out1 to counterparty (pk_b) with amt_a_to_b
        // - out2 back to sender (pk_a) with remaining change
        let pk1_out = pk_b;
        let pk2_out = pk_sender;

        let (pk1x, pk1y) = jubjub_fields(&pk1_out);
        let (pk2x, pk2y) = jubjub_fields(&pk2_out);

        let new1 = Utxo {
            asset_id,
            amount: amt_a_to_b,
            randomness: F::random(&mut rng),
        };
        let new2 = Utxo {
            asset_id,
            amount: change_amt,
            randomness: F::random(&mut rng),
        };

        let new_c1 = host_commit(new1.asset_id, new1.amount, pk1x, pk1y, new1.randomness);
        let mut new_c2 = host_commit(new2.asset_id, new2.amount, pk2x, pk2y, new2.randomness);
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

        // Compute sterms (must be non-zero to trigger swap mode).
        // Multi-asset swap terms bind BOTH assets; for this single-proof unit test we can pick an arbitrary counter-asset.
        let mut asset_id_b = F::random(&mut rng);
        while asset_id_b == asset_id {
            asset_id_b = F::random(&mut rng);
        }
        let mut sterms = host_swap_terms(
            asset_id, asset_id_b, pk_sx, pk_sy, pk_bx_t, pk_by_t, amt_a_to_b, 0,
        );
        if sterms == F::ZERO {
            // extremely unlikely; perturb by changing the "other direction" amount
            sterms = host_swap_terms(
                asset_id, asset_id_b, pk_sx, pk_sy, pk_bx_t, pk_by_t, amt_a_to_b, 1,
            );
        }

        // vto must be a u64; swapcm must be H2(sterms, vto)
        let vto_u64: u64 = rng.r#gen::<u64>();
        let vto = F::from(vto_u64);
        let swapcm = host_swapcm(sterms, vto);

        let instance = Spend2Output2PublicInputs {
            root,
            pk_bx,
            pk_by,
            new_c1,
            new_c2,
            nf1,
            nf2,
            sterms,
            swapcm,
            vto,
        };

        let terms = SwapTermsWitness {
            asset_id_a: asset_id,
            asset_id_b,
            pk_a: pk_sender,
            pk_b,
            amt_a_to_b,
            amt_b_to_a: 0,
        };

        let witness = (
            commit_map, sk, alpha_f, old1, old2, new1, new2, pk1_out, pk2_out, terms,
        );

        (instance, witness)
    }

    // -----------------------------
    // Unit tests (cheap, pure)
    // -----------------------------

    #[test]
    fn unit_domain_separation_tags_differ() {
        assert_ne!(UTXO_COMMIT_TAG, UTXO_NULLIFY_TAG);
        assert_ne!(UTXO_COMMIT_TAG, SWAP_TERMS_TAG);
        assert_ne!(UTXO_NULLIFY_TAG, SWAP_TERMS_TAG);
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
        fn pbt_valid_transfer_witness_is_accepted(seed in any::<u64>()) {
            let (instance, witness) = make_valid_transfer_case(seed);
            prop_assert!(accepts(&instance, witness, seed));
        }
    }

    #[test]
    fn swap_case_is_accepted() {
        let seed = 424242;
        let (instance, witness) = make_valid_swap_case(seed);
        assert!(accepts(&instance, witness, seed));
    }

    // -----------------------------
    // Negative tests (each targets a safety property)
    // -----------------------------

    #[test]
    fn negative_wrong_root_is_rejected() {
        let seed = 1001;
        let (mut instance, witness) = make_valid_transfer_case(seed);
        instance.root = F::random(&mut ChaCha8Rng::seed_from_u64(9999)); // tamper
        assert!(rejects(&instance, witness, seed));
    }

    #[test]
    fn negative_missing_input_membership_is_rejected() {
        let seed = 1002;
        let (instance, mut witness) = make_valid_transfer_case(seed);

        let empty = MapMt::<F, PoseidonChip<F>>::new(&F::ZERO);
        witness.0 = empty;

        assert!(rejects(&instance, witness, seed));
    }

    #[test]
    fn negative_alpha_zero_is_rejected() {
        let seed = 1003;
        let (instance, mut witness) = make_valid_transfer_case(seed);
        witness.2 = F::ZERO; // alpha_f = 0 violates assert_non_zero
        assert!(rejects(&instance, witness, seed));
    }

    #[test]
    fn negative_asset_id_mismatch_is_rejected() {
        let seed = 1004;
        let (instance, mut witness) = make_valid_transfer_case(seed);

        let mut old2 = witness.4.clone();
        old2.asset_id = F::random(&mut ChaCha8Rng::seed_from_u64(55));
        witness.4 = old2;

        assert!(rejects(&instance, witness, seed));
    }

    #[test]
    fn negative_value_conservation_violation_is_rejected() {
        let seed = 1005;
        let (instance, mut witness) = make_valid_transfer_case(seed);

        let mut new1 = witness.5.clone();
        new1.amount = new1.amount.saturating_add(1);
        witness.5 = new1;

        assert!(rejects(&instance, witness, seed));
    }

    #[test]
    fn negative_duplicate_inputs_nf1_eq_nf2_is_rejected() {
        let seed = 1006;
        let (instance, mut witness) = make_valid_transfer_case(seed);

        witness.4 = witness.3.clone();

        assert!(rejects(&instance, witness, seed));
    }

    #[test]
    fn negative_duplicate_outputs_new_c1_eq_new_c2_is_rejected() {
        let seed = 1007;
        let (instance, mut witness) = make_valid_transfer_case(seed);

        witness.6 = witness.5.clone();
        witness.8 = witness.7;

        assert!(rejects(&instance, witness, seed));
    }

    #[test]
    fn negative_public_input_tamper_pk_blinded_is_rejected() {
        let seed = 1008;
        let (mut instance, witness) = make_valid_transfer_case(seed);

        instance.pk_bx = F::random(&mut ChaCha8Rng::seed_from_u64(123));
        assert!(rejects(&instance, witness, seed));
    }

    #[test]
    fn negative_public_input_tamper_output_commitment_is_rejected() {
        let seed = 1009;
        let (mut instance, witness) = make_valid_transfer_case(seed);

        instance.new_c1 = F::random(&mut ChaCha8Rng::seed_from_u64(321));
        assert!(rejects(&instance, witness, seed));
    }

    #[test]
    fn negative_public_input_tamper_nullifier_is_rejected() {
        let seed = 1010;
        let (mut instance, witness) = make_valid_transfer_case(seed);

        instance.nf2 = F::random(&mut ChaCha8Rng::seed_from_u64(777));
        assert!(rejects(&instance, witness, seed));
    }

    #[test]
    fn negative_transfer_mode_nonzero_swap_fields_is_rejected() {
        // This specifically exercises the new robustness rule:
        // if sterms == 0, swapcm and vto must be zero.
        let seed = 1011;
        let (mut instance, witness) = make_valid_transfer_case(seed);
        instance.swapcm = F::from(7u64);
        assert!(rejects(&instance, witness, seed));
    }

    #[test]
    fn negative_swap_mode_bad_sterms_is_rejected() {
        let seed = 1012;
        let (mut instance, witness) = make_valid_swap_case(seed);
        instance.sterms = F::from(123456u64); // mismatched commitment to terms
        assert!(rejects(&instance, witness, seed));
    }

    #[test]
    fn negative_swap_mode_bad_swapcm_is_rejected() {
        // if sterms != 0 then swapcm must equal H2(sterms, vto)
        let seed = 1013;
        let (mut instance, witness) = make_valid_swap_case(seed);
        instance.swapcm = instance.swapcm.add(&F::ONE);
        assert!(rejects(&instance, witness, seed));
    }
}
