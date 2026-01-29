// SPDX-License-Identifier: CC0-1.0

use midnight_circuits::{
    ecc::foreign::{ForeignEccChip, ForeignEccConfig},
    field::{
        NativeChip, NativeConfig, NativeGadget,
        decomposition::chip::{P2RDecompositionChip, P2RDecompositionConfig},
    },
    hash::poseidon::{PoseidonChip, PoseidonConfig},
    types::ComposableChip,
    verifier::{BlstrsEmulation, SelfEmulation, VerifierGadget},
};

use super::*;

pub type S = BlstrsEmulation;
type F = <S as SelfEmulation>::F;
type C = <S as SelfEmulation>::C;
type E = <S as SelfEmulation>::Engine;
type CBase = <C as midnight_circuits::ecc::curves::CircuitCurve>::Base;
type NG = NativeGadget<F, P2RDecompositionChip<F>, NativeChip<F>>;

pub struct AggChips {
    pub native: NativeChip<F>,
    pub decomp: P2RDecompositionChip<F>,
    pub scalar: NG, // NativeGadget<F, P2RDecompositionChip<F>, NativeChip<F>>
    pub curve: ForeignEccChip<F, C, C, NG, NG>,
    pub poseidon: PoseidonChip<F>,
    pub verifier: VerifierGadget<S>,
}

pub fn build_agg_chips<const K: u32>(
    config: &(
        NativeConfig,
        P2RDecompositionConfig,
        ForeignEccConfig<C>,
        PoseidonConfig<F>,
    ),
) -> AggChips {
    let native = <NativeChip<F> as ComposableChip<F>>::new(&config.0, &());
    let decomp = P2RDecompositionChip::new(&config.1, &(K as usize - 1));
    let scalar = NativeGadget::new(decomp.clone(), native.clone());
    let curve = ForeignEccChip::new(&config.2, &scalar, &scalar);
    let poseidon = PoseidonChip::new(&config.3, &native);
    let verifier = VerifierGadget::<S>::new(&curve, &scalar, &poseidon);

    AggChips {
        native,
        decomp,
        scalar,
        curve,
        poseidon,
        verifier,
    }
}
