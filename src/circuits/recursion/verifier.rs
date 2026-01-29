// SPDX-License-Identifier: CC0-1.0

use group::Group;
use halo2curves::ff::Field;

use midnight_circuits::{
    ecc::foreign::ForeignEccChip,
    field::{NativeGadget, decomposition::chip::P2RDecompositionChip, native::NativeChip},
    instructions::AssignmentInstructions,
    types::{AssignedForeignPoint, AssignedNative},
    verifier::{AssignedAccumulator, AssignedVk, BlstrsEmulation, SelfEmulation, VerifierGadget},
};

use midnight_proofs::{
    circuit::{Layouter, Value},
    plonk::{ConstraintSystem, Error},
    poly::EvaluationDomain,
};

/// Canonical recursion “environment” used by the extracted modules.
pub type S = BlstrsEmulation;
pub type F = <S as SelfEmulation>::F;
pub type C = <S as SelfEmulation>::C;

/// Scalar-chip used everywhere in the monolith.
pub type ScalarChip = NativeGadget<F, P2RDecompositionChip<F>, NativeChip<F>>;

/// Curve-chip used by the verifier gadget (matches your `ForeignEccChip::new(...)` usage).
pub type CurveChip = ForeignEccChip<F, C, C, ScalarChip, ScalarChip>;

/// Verifier gadget used across aggregation circuits.
pub type VerifierChip = VerifierGadget<S>;

use crate::circuits::recursion::chips::AggChips;

// proof_agg/gadgets/plonk.rs
use super::*;

pub fn assign_vk(
    layouter: &mut impl Layouter<F>,
    chips: &AggChips,
    vk_name: &str,
    vk: &(EvaluationDomain<F>, ConstraintSystem<F>, F),
) -> Result<midnight_circuits::verifier::AssignedVk<S>, Error> {
    layouter.namespace(|| "assign_vk", |mut layouter| {
        let vk_val = chips.native.assign_fixed(&mut layouter, vk.2)?;
        chips.verifier.assign_vk(vk_name, &vk.0, &vk.1, vk_val)
    })
}

pub fn prepare_proof_to_acc(
    layouter: &mut impl Layouter<F>,
    chips: &AggChips,
    assigned_vk: &midnight_circuits::verifier::AssignedVk<S>,
    proof: Value<Vec<u8>>,
    public_inputs: &[AssignedNative<F>],
) -> Result<AssignedAccumulator<S>, Error> {
    layouter.namespace(|| "partial_verify_prepare", |mut layouter| {
        let id_point: AssignedForeignPoint<F, C, C> =
            chips.curve.assign_fixed(&mut layouter, C::identity())?;

        let mut acc = chips.verifier.prepare(
            &mut layouter,
            assigned_vk,
            &[("com_instance", id_point)],
            &[public_inputs],
            proof,
        )?;
        acc.collapse(&mut layouter, &chips.curve, &chips.scalar)?;
        Ok(acc)
    })
}

pub fn assign_and_collapse_acc(
    layouter: &mut impl Layouter<F>,
    chips: &AggChips,
    fixed_base_names: &[String],
    acc: Value<Accumulator<S>>,
) -> Result<AssignedAccumulator<S>, Error> {
    layouter.namespace(|| "assign_acc", |mut layouter| {
        let mut a = AssignedAccumulator::assign(
            &mut layouter,
            &chips.curve,
            &chips.scalar,
            1,
            1,
            &[],
            fixed_base_names,
            acc,
        )?;
        a.collapse(&mut layouter, &chips.curve, &chips.scalar)?;
        Ok(a)
    })
}

pub fn neutralize_acc_if_needed(
    layouter: &mut impl Layouter<F>,
    chips: &AggChips,
    acc: &mut AssignedAccumulator<S>,
    neutral_bit: &AssignedNative<F>,
) -> Result<(), Error> {
    layouter.namespace(|| "neutralize_acc", |mut layouter| {
        AssignedAccumulator::scale_by_bit(&mut layouter, &chips.scalar, neutral_bit, acc)?;
        acc.collapse(&mut layouter, &chips.curve, &chips.scalar)?;
        Ok(())
    })
}

pub fn fold_4_accumulators(
    layouter: &mut impl Layouter<F>,
    chips: &AggChips,
    accs: [AssignedAccumulator<S>; 4],
) -> Result<AssignedAccumulator<S>, Error> {
    layouter.namespace(|| "fold_4_accumulators", |mut layouter| {
        let mut out = AssignedAccumulator::<S>::accumulate(
            &mut layouter,
            &chips.verifier,
            &chips.scalar,
            &chips.poseidon,
            &accs,
        )?;
        out.collapse(&mut layouter, &chips.curve, &chips.scalar)?;
        Ok(out)
    })
}
