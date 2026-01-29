// SPDX-License-Identifier: CC0-1.0

use ff::PrimeField;
use midnight_circuits::compact_std_lib::ZkStdLib;
use midnight_circuits::instructions::{AssignmentInstructions, DecompositionInstructions};
use midnight_circuits::types::{AssignedBigUint, AssignedNative};
use midnight_curves::Fq as F;
use midnight_proofs::circuit::{Layouter, Value};
use midnight_proofs::plonk::Error;

use crate::Utxo;

const AMOUNT_BITS: u32 = 128;

#[derive(Clone)]
pub(crate) struct AssignedUtxo {
    pub(crate) id: AssignedNative<F>,
    pub(crate) amount_f: AssignedNative<F>,
    pub(crate) amount_big: AssignedBigUint<F>,
    pub(crate) randomness: AssignedNative<F>,
}

pub(crate) fn assign_utxo<L: Layouter<F>>(
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
