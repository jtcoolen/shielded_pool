//! Circuit context ([`IvcCtx`]) and recursive partial verification.
//!
//! `IvcCtx` bundles all chips required by the IVC circuits and exposes
//! a minimal API that step functions use for gadget access.  The RPV
//! helpers are framework-internal.

use ff::Field;
use group::Group;

use midnight_circuits::{
    ecc::foreign::{ForeignEccChip, ForeignEccConfig, nb_foreign_ecc_chip_columns},
    field::{
        NativeGadget,
        decomposition::{
            chip::{P2RDecompositionChip, P2RDecompositionConfig},
            pow2range::Pow2RangeChip,
        },
        foreign::FieldChip,
        native::{NB_ARITH_COLS, NativeChip, NativeConfig},
    },
    hash::poseidon::{
        NB_POSEIDON_ADVICE_COLS, NB_POSEIDON_FIXED_COLS, PoseidonChip, PoseidonConfig,
    },
    instructions::{
        ArithInstructions, AssertionInstructions, AssignmentInstructions, HashInstructions,
        PublicInputInstructions, map::MapInstructions,
    },
    types::{AssignedNative, ComposableChip},
    verifier::{AssignedAccumulator, AssignedVk, VerifierGadget},
};
use midnight_proofs::{
    circuit::{Layouter, Value},
    plonk::{ConstraintSystem, Error},
};

use super::{Acc, C, CurveChip, F, IdPoint, MapGadget, NG, S};

////////////////////////////////////////////////////////////////////////////////
// Circuit configuration
////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Debug)]
pub struct AggCircuitConfig {
    pub native: NativeConfig,
    pub decomp: P2RDecompositionConfig,
    pub curve: ForeignEccConfig<C>,
    pub poseidon: PoseidonConfig<F>,
}

fn alloc<T>(n: usize, mut f: impl FnMut() -> T) -> Vec<T> {
    (0..n).map(|_| f()).collect()
}

fn first<const N: usize, T: Copy>(xs: &[T], what: &'static str) -> [T; N] {
    xs.get(..N)
        .unwrap_or_else(|| panic!("not enough columns for {what}: need {N}, got {}", xs.len()))
        .try_into()
        .unwrap()
}

pub fn configure_ivc_circuit(meta: &mut ConstraintSystem<F>) -> AggCircuitConfig {
    let nb_advice = nb_foreign_ecc_chip_columns::<F, C, C, NG>().max(NB_POSEIDON_ADVICE_COLS);
    let nb_fixed = (NB_ARITH_COLS + 4).max(NB_POSEIDON_FIXED_COLS);

    let advice = alloc(nb_advice, || meta.advice_column());
    let fixed = alloc(nb_fixed, || meta.fixed_column());
    let instance = [meta.instance_column(), meta.instance_column()];

    let native = NativeChip::configure(
        meta,
        &(
            first::<NB_ARITH_COLS, _>(&advice, "native advice"),
            first::<{ NB_ARITH_COLS + 4 }, _>(&fixed, "native fixed"),
            instance,
        ),
    );

    let decomp = {
        let pow2 = Pow2RangeChip::configure(meta, &advice[1..NB_ARITH_COLS]);
        P2RDecompositionChip::configure(meta, &(native.clone(), pow2))
    };

    type CBase = <C as midnight_circuits::ecc::curves::CircuitCurve>::Base;
    let base = FieldChip::<F, CBase, C, NG>::configure(meta, &advice);
    let curve = ForeignEccChip::<F, C, C, NG, NG>::configure(meta, &base, &advice);

    let poseidon = PoseidonChip::configure(
        meta,
        &(
            first::<NB_POSEIDON_ADVICE_COLS, _>(&advice, "poseidon advice"),
            first::<NB_POSEIDON_FIXED_COLS, _>(&fixed, "poseidon fixed"),
        ),
    );

    AggCircuitConfig {
        native,
        decomp,
        curve,
        poseidon,
    }
}

////////////////////////////////////////////////////////////////////////////////
// IvcCtx — chip bundle exposed to step functions
////////////////////////////////////////////////////////////////////////////////

#[derive(Clone)]
pub struct IvcCtx {
    pub native: NativeChip<F>,
    pub core_decomp: P2RDecompositionChip<F>,
    pub scalar: NG,
    pub curve: CurveChip,
    pub poseidon: PoseidonChip<F>,
    pub verifier: VerifierGadget<S>,
}

impl IvcCtx {
    pub fn new(cfg: &AggCircuitConfig, k_minus_1: usize) -> Self {
        let native = <NativeChip<F> as ComposableChip<F>>::new(&cfg.native, &());
        let core_decomp = P2RDecompositionChip::new(&cfg.decomp, &k_minus_1);
        let scalar = NativeGadget::new(core_decomp.clone(), native.clone());
        let curve = ForeignEccChip::new(&cfg.curve, &scalar, &scalar);
        let poseidon = PoseidonChip::new(&cfg.poseidon, &native);
        let verifier = VerifierGadget::<S>::new(&curve, &scalar, &poseidon);

        Self {
            native,
            core_decomp,
            scalar,
            curve,
            poseidon,
            verifier,
        }
    }

    pub fn load(&self, layouter: &mut impl Layouter<F>) -> Result<(), Error> {
        self.core_decomp.load(layouter)
    }

    // ── Scalar helpers ──────────────────────────────────────────────────

    pub fn one(&self, layouter: &mut impl Layouter<F>) -> Result<AssignedNative<F>, Error> {
        self.scalar.assign_fixed(layouter, F::ONE)
    }

    pub fn zero(&self, layouter: &mut impl Layouter<F>) -> Result<AssignedNative<F>, Error> {
        self.scalar.assign_fixed(layouter, F::ZERO)
    }

    pub fn assign(
        &self,
        layouter: &mut impl Layouter<F>,
        v: Value<F>,
    ) -> Result<AssignedNative<F>, Error> {
        self.scalar.assign(layouter, v)
    }

    #[allow(dead_code)]
    pub fn assign_fixed(
        &self,
        layouter: &mut impl Layouter<F>,
        v: F,
    ) -> Result<AssignedNative<F>, Error> {
        self.scalar.assign_fixed(layouter, v)
    }

    pub fn assert_eq(
        &self,
        layouter: &mut impl Layouter<F>,
        a: &AssignedNative<F>,
        b: &AssignedNative<F>,
    ) -> Result<(), Error> {
        self.scalar.assert_equal(layouter, a, b)
    }

    pub fn add(
        &self,
        layouter: &mut impl Layouter<F>,
        a: &AssignedNative<F>,
        b: &AssignedNative<F>,
    ) -> Result<AssignedNative<F>, Error> {
        self.scalar.add(layouter, a, b)
    }

    // ── Hashing ─────────────────────────────────────────────────────────

    pub fn hash2(
        &self,
        layouter: &mut impl Layouter<F>,
        a: &AssignedNative<F>,
        b: &AssignedNative<F>,
    ) -> Result<AssignedNative<F>, Error> {
        self.poseidon.hash(layouter, &[a.clone(), b.clone()])
    }

    pub fn hash_many(
        &self,
        layouter: &mut impl Layouter<F>,
        items: &[AssignedNative<F>],
    ) -> Result<AssignedNative<F>, Error> {
        self.poseidon.hash(layouter, items)
    }

    // ── Map helpers ─────────────────────────────────────────────────────

    pub fn init_map(
        &self,
        layouter: &mut impl Layouter<F>,
        map: Value<super::Map>,
    ) -> Result<MapGadget, Error> {
        let mut gadget = MapGadget::new(&self.scalar, &self.poseidon);
        gadget.init(layouter, map)?;
        Ok(gadget)
    }

    pub fn expose_native(
        &self,
        layouter: &mut impl Layouter<F>,
        values: impl IntoIterator<Item = AssignedNative<F>>,
    ) -> Result<(), Error> {
        for v in values {
            self.native.constrain_as_public_input(layouter, &v)?;
        }
        Ok(())
    }
}

////////////////////////////////////////////////////////////////////////////////
// Framework-internal: recursive partial verification
////////////////////////////////////////////////////////////////////////////////

pub(crate) struct RpvInput<'a> {
    pub assigned_vk: &'a AssignedVk<S>,
    pub children_are_client_proofs: bool,
    pub fixed_base_names: &'a [String],
    pub child_base_pis: Vec<Vec<AssignedNative<F>>>,
    pub child_proofs: Vec<Value<Vec<u8>>>,
    pub child_pi_accs: Vec<Value<Acc>>,
}

pub(crate) struct RpvOutput {
    pub next_acc: AssignedAccumulator<S>,
}

pub(crate) fn assign_pi_acc(
    ctx: &IvcCtx,
    layouter: &mut impl Layouter<F>,
    fixed_base_names: &[String],
    acc: Value<Acc>,
) -> Result<AssignedAccumulator<S>, Error> {
    let mut a = AssignedAccumulator::assign(
        layouter,
        &ctx.curve,
        &ctx.scalar,
        1,
        1,
        &[],
        fixed_base_names,
        acc,
    )?;
    a.collapse(layouter, &ctx.curve, &ctx.scalar)?;
    Ok(a)
}

fn neutralize_for_client_children(
    ctx: &IvcCtx,
    layouter: &mut impl Layouter<F>,
    is_client: bool,
    acc: &mut AssignedAccumulator<S>,
) -> Result<(), Error> {
    if is_client {
        let neutral = ctx.scalar.assign_fixed(layouter, false)?;
        AssignedAccumulator::scale_by_bit(layouter, &ctx.scalar, &neutral, acc)?;
        acc.collapse(layouter, &ctx.curve, &ctx.scalar)?;
    }
    Ok(())
}

fn prepare_proof_acc(
    ctx: &IvcCtx,
    layouter: &mut impl Layouter<F>,
    vk: &AssignedVk<S>,
    id: IdPoint,
    pi: &[AssignedNative<F>],
    proof: Value<Vec<u8>>,
) -> Result<AssignedAccumulator<S>, Error> {
    let mut a = ctx
        .verifier
        .prepare(layouter, vk, &[("com_instance", id)], &[pi], proof)?;
    a.collapse(layouter, &ctx.curve, &ctx.scalar)?;
    Ok(a)
}

fn accumulate_many(
    ctx: &IvcCtx,
    layouter: &mut impl Layouter<F>,
    parts: &[AssignedAccumulator<S>],
) -> Result<AssignedAccumulator<S>, Error> {
    let mut next = AssignedAccumulator::<S>::accumulate(
        layouter,
        &ctx.verifier,
        &ctx.scalar,
        &ctx.poseidon,
        parts,
    )?;
    next.collapse(layouter, &ctx.curve, &ctx.scalar)?;
    Ok(next)
}

/// Run recursive partial verification: verify two child proofs and fold
/// their accumulators into a single output accumulator.
pub(crate) fn recursive_partial_verify(
    ctx: &IvcCtx,
    layouter: &mut impl Layouter<F>,
    input: RpvInput<'_>,
) -> Result<RpvOutput, Error> {
    let RpvInput {
        assigned_vk,
        children_are_client_proofs,
        fixed_base_names,
        child_base_pis,
        child_proofs,
        child_pi_accs,
    } = input;

    if child_base_pis.len() != child_proofs.len() || child_proofs.len() != child_pi_accs.len() {
        return Err(Error::Synthesis("rpv arity mismatch".to_string()));
    }

    let id: super::IdPoint = ctx.curve.assign_fixed(layouter, C::identity())?;

    let mut to_fold = Vec::with_capacity(child_proofs.len() * 2);

    for ((base_pi, proof), pi_acc) in child_base_pis
        .into_iter()
        .zip(child_proofs.into_iter())
        .zip(child_pi_accs.into_iter())
    {
        let mut child_pi_acc = assign_pi_acc(ctx, layouter, fixed_base_names, pi_acc)?;
        neutralize_for_client_children(
            ctx,
            layouter,
            children_are_client_proofs,
            &mut child_pi_acc,
        )?;

        let mut child_pi = base_pi;
        if !children_are_client_proofs {
            child_pi.extend(ctx.verifier.as_public_input(layouter, &child_pi_acc)?);
        }

        let proof_acc =
            prepare_proof_acc(ctx, layouter, assigned_vk, id.clone(), &child_pi, proof)?;
        to_fold.push(proof_acc);
        to_fold.push(child_pi_acc);
    }

    let next_acc = accumulate_many(ctx, layouter, &to_fold)?;

    Ok(RpvOutput { next_acc })
}

/// Expose `[state_fields... || acc_pi...]` as public inputs.
pub(crate) fn expose_node_outputs(
    ctx: &IvcCtx,
    layouter: &mut impl Layouter<F>,
    state_fields: impl IntoIterator<Item = AssignedNative<F>>,
    acc: &AssignedAccumulator<S>,
) -> Result<(), Error> {
    ctx.expose_native(layouter, state_fields)?;
    let acc_pi = ctx.verifier.as_public_input(layouter, acc)?;
    ctx.expose_native(layouter, acc_pi)?;
    Ok(())
}
