// SPDX-License-Identifier: CC0-1.0

use midnight_circuits::{
    types::AssignedNative,
    verifier::{AssignedAccumulator, AssignedVk},
};

use midnight_proofs::{
    circuit::{Layouter, Value},
    plonk::Error,
};

use super::{
    accumulation::AggAccumulator,
    verifier::{
        CurveChip, F, S, ScalarChip, VerifierChip, assign_com_instance_id_point,
        prepare_proof_accumulator,
    },
};
use midnight_circuits::instructions::PublicInputInstructions;

/// Result of verifying two children and accumulating the 4-term list:
///   [proof_acc_L, pi_acc_L, proof_acc_R, pi_acc_R]
pub struct VerifyAndAccumulateResult {
    pub next_acc: AssignedAccumulator<S>,
    pub next_acc_pi: Vec<AssignedNative<F>>,
}

/// Verify two proofs and accumulate `(proof_acc, pi_acc)` pairs.
///
/// Notes:
/// - `left_pi_acc` / `right_pi_acc` should already be assigned (and typically collapsed).
/// - Public-input vectors are slices (already assigned field elements).
pub fn verify_and_accumulate_arity_2(
    layouter: &mut impl Layouter<F>,
    verifier_chip: &VerifierChip,
    curve_chip: &CurveChip,
    scalar_chip: &ScalarChip,
    poseidon_chip: &midnight_circuits::hash::poseidon::PoseidonChip<F>,
    assigned_vk: &AssignedVk<S>,

    left_proof: Value<Vec<u8>>,
    left_public_inputs: &[AssignedNative<F>],
    left_pi_acc: AssignedAccumulator<S>,

    right_proof: Value<Vec<u8>>,
    right_public_inputs: &[AssignedNative<F>],
    right_pi_acc: AssignedAccumulator<S>,
) -> Result<VerifyAndAccumulateResult, Error> {
    // Fixed base point binding for committed instances
    let id_point = assign_com_instance_id_point(layouter, curve_chip)?;

    // Convert proofs into accumulators
    let left_proof_acc = prepare_proof_accumulator(
        layouter,
        verifier_chip,
        curve_chip,
        scalar_chip,
        assigned_vk,
        id_point.clone(),
        left_public_inputs,
        left_proof,
    )?;

    let right_proof_acc = prepare_proof_accumulator(
        layouter,
        verifier_chip,
        curve_chip,
        scalar_chip,
        assigned_vk,
        id_point,
        right_public_inputs,
        right_proof,
    )?;

    // Accumulate the 4 elements
    let mut next_acc = AssignedAccumulator::<S>::accumulate(
        layouter,
        verifier_chip,
        scalar_chip,
        poseidon_chip,
        &[left_proof_acc, left_pi_acc, right_proof_acc, right_pi_acc],
    )?;
    next_acc.collapse(layouter, curve_chip, scalar_chip)?;

    // Turn final accumulator into public-input fields (caller can constrain)
    let next_acc_pi = verifier_chip.as_public_input(layouter, &next_acc)?;

    Ok(VerifyAndAccumulateResult {
        next_acc,
        next_acc_pi,
    })
}
