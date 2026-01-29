// SPDX-License-Identifier: CC0-1.0

use crate::primitives::nullifiers::UTXO_NULLIFY_TAG;
use ff::Field;
use midnight_circuits::compact_std_lib::ZkStdLib;
use midnight_circuits::instructions::AssignmentInstructions;
use midnight_circuits::types::AssignedNative;
use midnight_curves::Fq as F;
use midnight_proofs::{circuit::Layouter, plonk::Error};

pub(crate) fn compute_nullifier<L: Layouter<F>>(
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
