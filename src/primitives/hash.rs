// SPDX-License-Identifier: CC0-1.0

use midnight_circuits::hash::poseidon::PoseidonChip;
use midnight_circuits::instructions::hash::HashCPU;
use midnight_curves::Fq as F;

pub(crate) fn host_instance_hash(items: [F; 7]) -> F {
    <PoseidonChip<F> as HashCPU<F, F>>::hash(&items)
}
