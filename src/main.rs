use std::time::Instant;

use ff::{Field, PrimeField};
use group::Group;

use midnight_circuits::{
    biguint::AssignedBigUint,
    compact_std_lib::{self, Relation, ZkStdLib, ZkStdLibArch, cost_model},
    ecc::native::AssignedScalarOfNativeCurve,
    hash::poseidon::{PoseidonChip, PoseidonState},
    instructions::{
        AssertionInstructions, AssignmentInstructions, ConversionInstructions,
        DecompositionInstructions, EccInstructions, PublicInputInstructions, ZeroInstructions,
        hash::HashCPU,
        map::{MapCPU, MapInstructions},
    },
    map::cpu::MapMt,
    types::{AssignedNative, AssignedNativePoint, Instantiable},
};

use midnight_curves::{Fq as F, Fr as JubjubScalar, JubjubExtended as Jubjub, JubjubSubgroup};
use midnight_proofs::{
    circuit::{Layouter, Value},
    transcript::Transcript,
};
use midnight_proofs::{
    plonk::{Error, create_proof, keygen_pk, keygen_vk_with_k, prepare},
    transcript::{Hashable, Sampleable, TranscriptHash},
};
use rand::{Rng, SeedableRng, rngs::OsRng};
use rand_chacha::ChaCha8Rng;

use sha3::{Digest, Keccak256};
use std::{io, io::Read};

use ff::FromUniformBytes;
use group::GroupEncoding;

/// Newtype so you can refer to it as `KeccakTranscript` in generics.
#[derive(Clone)]
pub struct KeccakTranscript(Keccak256);

impl TranscriptHash for KeccakTranscript {
    type Input = Vec<u8>;
    type Output = Vec<u8>; // we return 64 bytes for your existing sampling code

    fn init() -> Self {
        // Domain separation (on-chain: start transcript bytes with this literal)
        let mut h = Keccak256::new();
        h.update(b"Domain separator for transcript");
        Self(h)
    }

    fn absorb(&mut self, input: &Self::Input) {
        self.0.update(&[0]);
        self.0.update(input);
    }

    fn squeeze(&mut self) -> Self::Output {
        // Mutate transcript state (so multiple squeezes differ)
        self.0.update(&[1]);

        // EVM-compatible 64 bytes:
        // out = keccak256(preimage || 0x00) || keccak256(preimage || 0x01)
        let mut out = Vec::with_capacity(64);

        let r0 = {
            let mut t = self.0.clone();
            t.update(&[0u8]);
            t.finalize()
        };
        out.extend_from_slice(r0.as_slice());

        let r1 = {
            let mut t = self.0.clone();
            t.update(&[1u8]);
            t.finalize()
        };
        out.extend_from_slice(r1.as_slice());

        debug_assert_eq!(out.len(), 64);
        out
    }
}

// ------------------------------------------------------------
// Fix #1 from your error: G1Projective must be Hashable<KeccakTranscript>
// ------------------------------------------------------------
impl Hashable<KeccakTranscript> for midnight_curves::G1Projective {
    fn to_input(&self) -> Vec<u8> {
        Hashable::<KeccakTranscript>::to_bytes(self)
    }

    fn to_bytes(&self) -> Vec<u8> {
        <Self as GroupEncoding>::to_bytes(self).as_ref().to_vec()
    }

    fn read(buffer: &mut impl Read) -> io::Result<Self> {
        let mut bytes = <Self as GroupEncoding>::Repr::default();
        buffer.read_exact(bytes.as_mut())?;

        Option::from(Self::from_bytes(&bytes))
            .ok_or_else(|| io::Error::other("Invalid BLS12-381 point encoding in proof"))
    }
}

// ------------------------------------------------------------
// Fix #2/#3 from your error: BlsScalar must be Hashable + Sampleable for KeccakTranscript
//
// IMPORTANT: Replace `midnight_curves::Fq` below with your actual `BlsScalar` type
// if it is a distinct alias/type in your crate.
// ------------------------------------------------------------
impl Hashable<KeccakTranscript> for midnight_curves::Fq {
    fn to_input(&self) -> Vec<u8> {
        self.to_repr().to_vec()
    }

    fn to_bytes(&self) -> Vec<u8> {
        self.to_repr().to_vec()
    }

    fn read(buffer: &mut impl Read) -> io::Result<Self> {
        let mut bytes = <Self as PrimeField>::Repr::default();
        buffer.read_exact(bytes.as_mut())?;

        Option::from(Self::from_repr(bytes))
            .ok_or_else(|| io::Error::other("Invalid BLS12-381 scalar encoding in proof"))
    }
}

impl Sampleable<KeccakTranscript> for midnight_curves::Fq {
    fn sample(hash_output: Vec<u8>) -> Self {
        assert!(hash_output.len() <= 64);
        assert!(hash_output.len() >= (midnight_curves::Fq::NUM_BITS as usize / 8) + 12);

        let mut bytes = [0u8; 64];
        bytes[..hash_output.len()].copy_from_slice(&hash_output);

        midnight_curves::Fq::from_uniform_bytes(&bytes)
    }
}

mod proof_agg {
    use core::array;

    use halo2curves::{ff::Field, group::Group};
    use midnight_circuits::hash::poseidon::PoseidonState;
    use midnight_circuits::instructions::map::{MapCPU, MapInstructions};
    use midnight_circuits::types::Instantiable;
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
            AssertionInstructions, AssignmentInstructions, HashInstructions,
            PublicInputInstructions,
        },
        types::{AssignedForeignPoint, AssignedNative, ComposableChip},
        verifier::{
            Accumulator, AssignedAccumulator, BlstrsEmulation, SelfEmulation, VerifierGadget,
        },
    };
    use midnight_curves::Bls12;
    use midnight_proofs::poly::kzg::params::ParamsKZG;
    use midnight_proofs::utils::SerdeFormat;
    use midnight_proofs::{
        circuit::{Layouter, SimpleFloorPlanner, Value},
        plonk::{
            Circuit, ConstraintSystem, Error, ProvingKey, VerifyingKey, create_proof, keygen_pk,
            keygen_vk_with_k,
        },
        poly::{EvaluationDomain, kzg::KZGCommitmentScheme},
        transcript::{CircuitTranscript, Transcript},
    };
    use rand::{
        SeedableRng,
        rngs::{OsRng, StdRng},
    };
    use rayon::prelude::*;
    use std::collections::{BTreeMap, BTreeSet};
    use std::env;
    use std::fs::File;
    use std::io::{BufReader, Write};
    use std::path::Path;
    use std::sync::Arc;
    use std::time::Instant;

    pub type S = BlstrsEmulation;
    type F = <S as SelfEmulation>::F;
    type C = <S as SelfEmulation>::C;
    type E = <S as SelfEmulation>::Engine;
    type CBase = <C as midnight_circuits::ecc::curves::CircuitCurve>::Base;
    type NG = NativeGadget<F, P2RDecompositionChip<F>, NativeChip<F>>;
    type Map = midnight_circuits::map::cpu::MapMt<F, PoseidonChip<F>>;

    #[repr(transparent)]
    #[derive(Clone)]
    struct SendableMap(Map);
    unsafe impl Send for SendableMap {}
    unsafe impl Sync for SendableMap {}
    impl SendableMap {
        fn clone_inner(&self) -> Map {
            self.0.clone()
        }
    }

    pub type AggAccumulator = Accumulator<S>;
    pub fn accumulator_as_public_input(acc: &AggAccumulator) -> Vec<F> {
        AssignedAccumulator::as_public_input(acc)
    }

    const K_LEAF: u32 = 19;
    const K_INTERNAL: u32 = 19;

    pub const AGG_K: u32 = K_INTERNAL;

    type Vk = VerifyingKey<F, KZGCommitmentScheme<E>>;
    type Pk = ProvingKey<F, KZGCommitmentScheme<E>>;

    type LeafAggCircuit = AggCircuit<K_LEAF>;
    type InternalAggCircuit = AggCircuit<K_INTERNAL>;

    // ---- FIX (Issue 1): bind historic-roots-set Merkle map root into agg state
    pub const AGG_STATE_WIDTH: usize = 6;

    #[derive(Clone, Copy, Debug)]
    pub struct AggState {
        pub c_pre: F,
        pub c_post: F,
        pub n_pre: F,
        pub n_post: F,
        pub subroot: F,
        pub roots_set_root: F, // NEW: root of historic commitment-roots set used in membership checks
    }
    impl AggState {
        pub fn to_fields(&self) -> [F; AGG_STATE_WIDTH] {
            [
                self.c_pre,
                self.c_post,
                self.n_pre,
                self.n_post,
                self.subroot,
                self.roots_set_root,
            ]
        }
    }

    macro_rules! ensure {
        ($cond:expr, $($arg:tt)*) => {
            if !$cond {
                return Err(io_other(format!($($arg)*)));
            }
        };
    }
    fn io_other(msg: impl Into<String>) -> std::io::Error {
        std::io::Error::new(std::io::ErrorKind::Other, msg.into())
    }

    // Deterministic mock SRS for testing only MUST NOT be used in production
    pub fn mock_srs_agg(k: u32) -> Result<ParamsKZG<Bls12>, std::io::Error> {
        ensure!(
            k <= 20,
            "No Filecoin SRS available for circuits of size k={}",
            k
        );

        let srs_dir = env::var("SRS_DIR").unwrap_or_else(|_| "./examples/assets".into());
        let srs_path = format!("{srs_dir}/bls_mock_2p{k}");
        let fetching_path = if Path::new(&srs_path).exists() {
            srs_path.clone()
        } else {
            format!("{srs_dir}/bls_mock_2p20")
        };

        // If the (mock) params file we're about to read doesn't exist, create it via unsafe_setup.
        if !Path::new(&fetching_path).exists() {
            std::fs::create_dir_all(&srs_dir)
                .map_err(|e| io_other(format!("Failed to create SRS_DIR '{}': {e}", srs_dir)))?;

            let rng = StdRng::seed_from_u64(0xDEAD_BEEF_u64);

            let params = ParamsKZG::<Bls12>::unsafe_setup(20, rng);

            let mut buf = Vec::new();
            params
                .write_custom(&mut buf, SerdeFormat::RawBytesUnchecked)
                .map_err(|e| io_other(format!("Failed to serialize mock params: {e}")))?;

            let mut file = File::create(&fetching_path).map_err(|e| {
                io_other(format!(
                    "Failed to create mock SRS file '{}': {e}",
                    fetching_path
                ))
            })?;
            file.write_all(&buf).map_err(|e| {
                io_other(format!(
                    "Failed to write mock SRS file '{}': {e}",
                    fetching_path
                ))
            })?;
        }

        let params_fs = File::open(Path::new(&fetching_path)).map_err(|e| {
            io_other(format!(
                "Failed to open SRS file at '{}': {e}. (Did you set SRS_DIR?)",
                fetching_path
            ))
        })?;

        let mut params: ParamsKZG<Bls12> = ParamsKZG::read_custom::<_>(
            &mut BufReader::new(params_fs),
            SerdeFormat::RawBytesUnchecked,
        )
        .map_err(|e| {
            io_other(format!(
                "Failed to read SRS params from '{}': {e}",
                fetching_path
            ))
        })?;

        // If we loaded the MAX_K file, downsize and cache the per-k file
        if fetching_path != srs_path {
            params.downsize(k);

            let mut buf = Vec::new();
            params
                .write_custom(&mut buf, SerdeFormat::RawBytesUnchecked)
                .map_err(|e| io_other(format!("Failed to serialize downsized params: {e}")))?;

            let mut file = File::create(&srs_path).map_err(|e| {
                io_other(format!(
                    "Failed to create mock SRS cache file '{}': {e}",
                    srs_path
                ))
            })?;
            file.write_all(&buf).map_err(|e| {
                io_other(format!(
                    "Failed to write mock SRS cache '{}': {e}",
                    srs_path
                ))
            })?;
        }

        Ok(params)
    }

    pub fn filecoin_srs_agg(k: u32) -> Result<ParamsKZG<Bls12>, std::io::Error> {
        ensure!(
            k <= 20,
            "No Filecoin SRS available for circuits of size k={}",
            k
        );

        let srs_dir = env::var("SRS_DIR").unwrap_or_else(|_| "./examples/assets".into());
        let srs_path = format!("{srs_dir}/bls_filecoin_2p{k}");
        let fetching_path = if Path::new(&srs_path).exists() {
            srs_path.clone()
        } else {
            format!("{srs_dir}/bls_filecoin_2p20")
        };

        let params_fs = File::open(Path::new(&fetching_path)).map_err(|e| {
            io_other(format!(
                "Failed to open SRS file at '{}': {e}. (Did you set SRS_DIR?)",
                fetching_path
            ))
        })?;

        let mut params: ParamsKZG<Bls12> = ParamsKZG::read_custom::<_>(
            &mut BufReader::new(params_fs),
            SerdeFormat::RawBytesUnchecked,
        )
        .map_err(|e| {
            io_other(format!(
                "Failed to read SRS params from '{}': {e}",
                fetching_path
            ))
        })?;

        if fetching_path != srs_path {
            params.downsize(k);

            let mut buf = Vec::new();
            params
                .write_custom(&mut buf, SerdeFormat::RawBytesUnchecked)
                .map_err(|e| io_other(format!("Failed to serialize downsized params: {e}")))?;

            let mut file = File::create(&srs_path).map_err(|e| {
                io_other(format!(
                    "Failed to create SRS cache file '{}': {e}",
                    srs_path
                ))
            })?;
            file.write_all(&buf[..])
                .map_err(|e| io_other(format!("Failed to write SRS cache '{}': {e}", srs_path)))?;
        }

        Ok(params)
    }

    #[derive(Clone, Debug)]
    struct VkData {
        domain: EvaluationDomain<F>,
        cs: ConstraintSystem<F>,
        transcript_repr: F,
    }

    fn configure_agg_circuit(
        meta: &mut ConstraintSystem<F>,
    ) -> (
        NativeConfig,
        P2RDecompositionConfig,
        ForeignEccConfig<C>,
        PoseidonConfig<F>,
        //Sha256Config,
    ) {
        let nb_advice_cols = nb_foreign_ecc_chip_columns::<F, C, C, NG>();
        let nb_fixed_cols = NB_ARITH_COLS + 4;

        let advice_columns: Vec<_> = (0..nb_advice_cols).map(|_| meta.advice_column()).collect();
        let fixed_columns: Vec<_> = (0..nb_fixed_cols).map(|_| meta.fixed_column()).collect();
        let committed_instance_column = meta.instance_column();
        let instance_column = meta.instance_column();

        let native_config = NativeChip::configure(
            meta,
            &(
                advice_columns[..NB_ARITH_COLS].try_into().unwrap(),
                fixed_columns[..NB_ARITH_COLS + 4].try_into().unwrap(),
                [committed_instance_column, instance_column],
            ),
        );

        let core_decomp_config = {
            let pow2_config = Pow2RangeChip::configure(meta, &advice_columns[1..NB_ARITH_COLS]);
            P2RDecompositionChip::configure(meta, &(native_config.clone(), pow2_config))
        };

        let base_config = FieldChip::<F, CBase, C, NG>::configure(meta, &advice_columns);
        let curve_config =
            ForeignEccChip::<F, C, C, NG, NG>::configure(meta, &base_config, &advice_columns);

        let poseidon_config = PoseidonChip::configure(
            meta,
            &(
                advice_columns[..NB_POSEIDON_ADVICE_COLS]
                    .try_into()
                    .unwrap(),
                fixed_columns[..NB_POSEIDON_FIXED_COLS].try_into().unwrap(),
            ),
        );

        /*let sha2_config = Sha256Chip::configure(
            meta,
            &(
                advice_columns[..NB_SHA256_ADVICE_COLS].try_into().unwrap(),
                fixed_columns[..NB_SHA256_FIXED_COLS].try_into().unwrap(),
            ),
        );*/

        (
            native_config,
            core_decomp_config,
            curve_config,
            poseidon_config,
            //sha2_config,
        )
    }

    #[derive(Clone, Debug)]
    pub struct AggPublicInputs {
        pub state: AggState,
        pub pi_acc: AggAccumulator,
    }
    impl AggPublicInputs {
        pub fn to_fields(&self) -> Vec<F> {
            let mut out = Vec::new();
            out.extend_from_slice(&self.state.to_fields());
            out.extend(AssignedAccumulator::as_public_input(&self.pi_acc));
            out
        }
    }

    #[derive(Clone, Debug)]
    pub struct AggCircuit<const K: u32> {
        child_vk: VkData,
        child_vk_name: String,

        left_child_state: [Value<F>; AGG_STATE_WIDTH],
        right_child_state: [Value<F>; AGG_STATE_WIDTH],

        left_items: Value<[F; 7]>,
        right_items: Value<[F; 7]>,

        pre_commitment_map: Value<Map>,
        pre_nullifier_map: Value<Map>,

        // NEW: historic commitment-roots set (used to allow “lagging” tx roots)
        pre_commitment_roots_map: Value<Map>,

        left_proof: Value<Vec<u8>>,
        right_proof: Value<Vec<u8>>,
        left_acc: Value<Accumulator<S>>,
        right_acc: Value<Accumulator<S>>,
        fixed_base_names: Vec<String>,
        is_leaf: bool,
    }

    impl<const K: u32> Circuit<F> for AggCircuit<K> {
        type Config = (
            NativeConfig,
            P2RDecompositionConfig,
            ForeignEccConfig<C>,
            PoseidonConfig<F>,
            //Sha256Config,
        );
        type FloorPlanner = SimpleFloorPlanner;
        type Params = ();

        fn without_witnesses(&self) -> Self {
            Self {
                child_vk: self.child_vk.clone(),
                child_vk_name: self.child_vk_name.clone(),
                left_child_state: array::from_fn(|_| Value::unknown()),
                right_child_state: array::from_fn(|_| Value::unknown()),
                left_items: Value::unknown(),
                right_items: Value::unknown(),
                pre_commitment_map: Value::unknown(),
                pre_nullifier_map: Value::unknown(),
                pre_commitment_roots_map: Value::unknown(),
                left_proof: Value::unknown(),
                right_proof: Value::unknown(),
                left_acc: Value::unknown(),
                right_acc: Value::unknown(),
                fixed_base_names: self.fixed_base_names.clone(),
                is_leaf: self.is_leaf,
            }
        }

        fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
            configure_agg_circuit(meta)
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            let native_chip = <NativeChip<F> as ComposableChip<F>>::new(&config.0, &());
            let core_decomp_chip = P2RDecompositionChip::new(&config.1, &(K as usize - 1));
            let scalar_chip: NativeGadget<
                midnight_curves::Fq,
                P2RDecompositionChip<midnight_curves::Fq>,
                NativeChip<midnight_curves::Fq>,
            > = NativeGadget::new(core_decomp_chip.clone(), native_chip.clone());
            let curve_chip = ForeignEccChip::new(&config.2, &scalar_chip, &scalar_chip);
            let poseidon_chip = PoseidonChip::new(&config.3, &native_chip);
            let verifier_chip = VerifierGadget::new(&curve_chip, &scalar_chip, &poseidon_chip);

            //let _sha2_chip = Sha256Chip::new(&config.4, &scalar_chip);

            let child_vk_val: AssignedNative<F> =
                native_chip.assign_fixed(&mut layouter, self.child_vk.transcript_repr)?;
            let assigned_vk = verifier_chip.assign_vk(
                self.child_vk_name.as_str(),
                &self.child_vk.domain,
                &self.child_vk.cs,
                child_vk_val,
            )?;

            let zero = scalar_chip.assign_fixed(&mut layouter, F::ZERO)?;
            let one = scalar_chip.assign_fixed(&mut layouter, F::ONE)?;

            let (out_state_fields, assigned_left_pi_base, assigned_right_pi_base) = if self.is_leaf
            {
                // Commitment map (rollup “current” set)
                let mut commit_map_gadget = midnight_circuits::map::map_gadget::MapGadget::<
                    F,
                    NG,
                    PoseidonChip<F>,
                >::new(&scalar_chip, &poseidon_chip);
                commit_map_gadget.init(&mut layouter, self.pre_commitment_map.clone())?;
                let c_pre = commit_map_gadget.succinct_repr();

                // Nullifier map (rollup “current” spent set)
                let mut null_map_gadget = midnight_circuits::map::map_gadget::MapGadget::<
                    F,
                    NG,
                    PoseidonChip<F>,
                >::new(&scalar_chip, &poseidon_chip);
                null_map_gadget.init(&mut layouter, self.pre_nullifier_map.clone())?;
                let n_pre = null_map_gadget.succinct_repr();

                // Historic roots set (what allows “lagging” tx roots)
                let mut roots_map_gadget = midnight_circuits::map::map_gadget::MapGadget::<
                    F,
                    NG,
                    PoseidonChip<F>,
                >::new(&scalar_chip, &poseidon_chip);
                roots_map_gadget.init(&mut layouter, self.pre_commitment_roots_map.clone())?;

                // ---- FIX (Issue 1): bind roots-set root into leaf agg public state
                let roots_set_root = roots_map_gadget.succinct_repr();

                let mut l: Vec<AssignedNative<F>> = Vec::with_capacity(7);
                for j in 0..7 {
                    l.push(
                        scalar_chip
                            .assign(&mut layouter, self.left_items.clone().map(|arr| arr[j]))?,
                    );
                }
                let mut r: Vec<AssignedNative<F>> = Vec::with_capacity(7);
                for j in 0..7 {
                    r.push(
                        scalar_chip
                            .assign(&mut layouter, self.right_items.clone().map(|arr| arr[j]))?,
                    );
                }

                // NEW: allow tx roots to be any HISTORIC root (not necessarily the rolling c_pre/c_mid)
                // Enforce: tx_root ∈ roots_set
                let ok_l = roots_map_gadget.get(&mut layouter, &l[0])?;
                scalar_chip.assert_equal(&mut layouter, &ok_l, &one)?;
                let ok_r = roots_map_gadget.get(&mut layouter, &r[0])?;
                scalar_chip.assert_equal(&mut layouter, &ok_r, &one)?;

                // ✅ Single Poseidon hash of all 7 would-be public inputs
                let inst_l = poseidon_chip.hash(&mut layouter, &l[..])?;
                let inst_r = poseidon_chip.hash(&mut layouter, &r[..])?;

                // Apply BOTH tx’s effects to the *current* rollup sets (commitments + nullifiers)
                commit_map_gadget.insert(&mut layouter, &l[3], &one)?;
                commit_map_gadget.insert(&mut layouter, &l[4], &one)?;

                for nf in [l[5].clone(), l[6].clone()] {
                    let old = null_map_gadget.get(&mut layouter, &nf)?;
                    scalar_chip.assert_equal(&mut layouter, &old, &zero)?;
                    null_map_gadget.insert(&mut layouter, &nf, &one)?;
                }

                commit_map_gadget.insert(&mut layouter, &r[3], &one)?;
                commit_map_gadget.insert(&mut layouter, &r[4], &one)?;

                for nf in [r[5].clone(), r[6].clone()] {
                    let old = null_map_gadget.get(&mut layouter, &nf)?;
                    scalar_chip.assert_equal(&mut layouter, &old, &zero)?;
                    null_map_gadget.insert(&mut layouter, &nf, &one)?;
                }

                let c_post = commit_map_gadget.succinct_repr();
                let n_post = null_map_gadget.succinct_repr();
                let subroot =
                    poseidon_chip.hash(&mut layouter, &[inst_l.clone(), inst_r.clone()])?;
                let out_fields = [c_pre, c_post, n_pre, n_post, subroot, roots_set_root];

                (out_fields, vec![inst_l], vec![inst_r])
            } else {
                let mut l_vec: Vec<AssignedNative<F>> = Vec::with_capacity(AGG_STATE_WIDTH);
                let mut r_vec: Vec<AssignedNative<F>> = Vec::with_capacity(AGG_STATE_WIDTH);
                for j in 0..AGG_STATE_WIDTH {
                    l_vec.push(scalar_chip.assign(&mut layouter, self.left_child_state[j])?);
                    r_vec.push(scalar_chip.assign(&mut layouter, self.right_child_state[j])?);
                }
                let l: [AssignedNative<F>; AGG_STATE_WIDTH] = l_vec.try_into().unwrap();
                let r: [AssignedNative<F>; AGG_STATE_WIDTH] = r_vec.try_into().unwrap();

                scalar_chip.assert_equal(&mut layouter, &l[1], &r[0])?;
                scalar_chip.assert_equal(&mut layouter, &l[3], &r[2])?;
                // ---- FIX (Issue 1): roots-set root must be identical across whole agg tree
                scalar_chip.assert_equal(&mut layouter, &l[5], &r[5])?;

                let subroot = poseidon_chip.hash(&mut layouter, &[l[4].clone(), r[4].clone()])?;

                let out_fields = [
                    l[0].clone(),
                    r[1].clone(),
                    l[2].clone(),
                    r[3].clone(),
                    subroot,
                    l[5].clone(), // propagate roots_set_root
                ];

                (out_fields, l.to_vec(), r.to_vec())
            };

            let mut left_acc = AssignedAccumulator::assign(
                &mut layouter,
                &curve_chip,
                &scalar_chip,
                1,
                1,
                &[],
                &self.fixed_base_names,
                self.left_acc.clone(),
            )?;
            let mut right_acc = AssignedAccumulator::assign(
                &mut layouter,
                &curve_chip,
                &scalar_chip,
                1,
                1,
                &[],
                &self.fixed_base_names,
                self.right_acc.clone(),
            )?;

            let (assigned_left_pi, assigned_right_pi) = if self.is_leaf {
                let neutral = scalar_chip.assign_fixed(&mut layouter, false)?;
                AssignedAccumulator::scale_by_bit(
                    &mut layouter,
                    &scalar_chip,
                    &neutral,
                    &mut left_acc,
                )?;
                AssignedAccumulator::scale_by_bit(
                    &mut layouter,
                    &scalar_chip,
                    &neutral,
                    &mut right_acc,
                )?;
                left_acc.collapse(&mut layouter, &curve_chip, &scalar_chip)?;
                right_acc.collapse(&mut layouter, &curve_chip, &scalar_chip)?;
                (assigned_left_pi_base, assigned_right_pi_base)
            } else {
                let mut left_pi = assigned_left_pi_base;
                left_pi.extend(verifier_chip.as_public_input(&mut layouter, &left_acc)?);

                let mut right_pi = assigned_right_pi_base;
                right_pi.extend(verifier_chip.as_public_input(&mut layouter, &right_acc)?);

                (left_pi, right_pi)
            };

            let id_point: AssignedForeignPoint<
                midnight_curves::Fq,
                midnight_curves::G1Projective,
                midnight_curves::G1Projective,
            > = curve_chip.assign_fixed(&mut layouter, C::identity())?;

            let mut left_proof_acc = verifier_chip.prepare(
                &mut layouter,
                &assigned_vk,
                &[("com_instance", id_point.clone())],
                &[&assigned_left_pi],
                self.left_proof.clone(),
            )?;
            left_proof_acc.collapse(&mut layouter, &curve_chip, &scalar_chip)?;

            let mut right_proof_acc = verifier_chip.prepare(
                &mut layouter,
                &assigned_vk,
                &[("com_instance", id_point)],
                &[&assigned_right_pi],
                self.right_proof.clone(),
            )?;
            right_proof_acc.collapse(&mut layouter, &curve_chip, &scalar_chip)?;

            let mut next_acc = AssignedAccumulator::<S>::accumulate(
                &mut layouter,
                &verifier_chip,
                &scalar_chip,
                &poseidon_chip,
                &[left_proof_acc, left_acc, right_proof_acc, right_acc],
            )?;
            next_acc.collapse(&mut layouter, &curve_chip, &scalar_chip)?;

            let next_acc_pi = verifier_chip.as_public_input(&mut layouter, &next_acc)?;
            for f in out_state_fields.iter() {
                native_chip.constrain_as_public_input(&mut layouter, f)?;
            }
            for x in next_acc_pi.iter() {
                native_chip.constrain_as_public_input(&mut layouter, x)?;
            }

            core_decomp_chip.load(&mut layouter)
        }
    }

    #[derive(Clone, Debug)]
    pub struct TreeNode {
        pub state: AggState,
        pub proof: Vec<u8>,
        pub proof_acc: AggAccumulator,
        pub pi_acc: AggAccumulator,
    }

    #[derive(Clone, Debug)]
    pub struct ClientProof {
        pub state: F,
        pub proof: Vec<u8>,
        pub public_items: [F; 7],
    }

    #[derive(Clone)]
    struct LeafPlan {
        i: usize,
        left_items: [F; 7],
        right_items: [F; 7],
        pre_commitment_map: SendableMap,
        pre_nullifier_map: SendableMap,
        pre_roots_map: SendableMap,
        expected_state: AggState,
        left_state: F,
        right_state: F,
        left_proof: Vec<u8>,
        right_proof: Vec<u8>,
    }

    #[derive(Clone, Debug)]
    pub struct AggregationResult {
        pub root_state: AggState,
        pub left_top: TreeNode,
        pub right_top: TreeNode,
        pub child_vk: (EvaluationDomain<F>, ConstraintSystem<F>, F),
        pub child_vk_name: String,
        pub child_level: usize,
        pub fixed_base_names: Vec<String>,
        pub fixed_bases: BTreeMap<String, C>,
    }

    fn fixed_base_names_for(vk_name: &str, cs: &ConstraintSystem<F>) -> Vec<String> {
        let mut names = vec![String::from("com_instance"), String::from("~G")];
        names.extend(midnight_circuits::verifier::fixed_base_names::<S>(
            vk_name,
            cs.num_fixed_columns() + cs.num_selectors(),
            cs.permutation().columns.len(),
        ));
        names
    }

    fn trivial_acc_with_names(names: &[String]) -> Accumulator<S> {
        use midnight_circuits::verifier::Msm;
        let fixed: BTreeMap<String, F> = names.iter().cloned().map(|n| (n, F::ZERO)).collect();
        Accumulator::<S>::new(
            Msm::new(&[C::default()], &[F::ONE], &BTreeMap::new()),
            Msm::new(&[C::default()], &[F::ONE], &fixed),
        )
    }

    fn poseidon_tree_root(leaf_states: &[F]) -> F {
        use midnight_circuits::instructions::hash::HashCPU;
        assert!(!leaf_states.is_empty(), "Need at least one leaf");
        assert!(
            leaf_states.len().is_power_of_two(),
            "Leaves must be power of two"
        );

        let mut level_states = leaf_states.to_vec();
        while level_states.len() > 1 {
            level_states = level_states
                .chunks(2)
                .map(|pair| <PoseidonChip<F> as HashCPU<F, F>>::hash(&[pair[0], pair[1]]))
                .collect();
        }
        level_states[0]
    }

    // ✅ Single Poseidon hash of all 7 would-be public inputs (host-side)
    fn host_instance_hash(items: [F; 7]) -> F {
        use midnight_circuits::instructions::hash::HashCPU;
        <PoseidonChip<F> as HashCPU<F, F>>::hash(&items)
    }

    fn verify_and_extract_acc(
        srs: &ParamsKZG<Bls12>,
        vk: &VerifyingKey<F, KZGCommitmentScheme<E>>,
        fixed_bases: &BTreeMap<String, C>,
        proof: &[u8],
        public_inputs: &[F],
    ) -> Accumulator<S> {
        let mut transcript = CircuitTranscript::<PoseidonState<F>>::init_from_bytes(proof);
        let committed_bases: &[&[C]] = &[&[C::identity()]];
        let instances: &[&[&[F]]] = &[&[public_inputs]];

        let dual_msm = midnight_proofs::plonk::prepare::<
            F,
            KZGCommitmentScheme<E>,
            CircuitTranscript<PoseidonState<F>>,
        >(vk, committed_bases, instances, &mut transcript)
        .expect("Verification (prepare) failed");

        assert!(
            dual_msm.clone().check(&srs.verifier_params()),
            "dual MSM did not check"
        );

        let mut acc: Accumulator<S> = dual_msm.into();
        acc.extract_fixed_bases(fixed_bases);
        acc.collapse();

        assert!(
            acc.check(&srs.s_g2().into(), fixed_bases),
            "Accumulator failed final check against fixed bases"
        );

        acc
    }

    fn agg_vk_name_for_level(level: usize) -> String {
        format!("agg_vk_lvl{level}")
    }

    #[derive(Clone)]
    struct AggLevelKeys {
        level: usize,
        name: String,
        vk: Arc<Vk>,
        pk: Arc<Pk>,
        vk_data: VkData,
        fixed_bases: BTreeMap<String, C>,
    }

    impl AggLevelKeys {
        fn new(level: usize, name: String, vk: Vk, pk: Pk) -> Self {
            let k = if level == 1 { K_LEAF } else { K_INTERNAL };
            let vk_data = VkData {
                domain: EvaluationDomain::new(vk.cs().degree() as u32, k),
                cs: vk.cs().clone(),
                transcript_repr: vk.transcript_repr(),
            };

            let mut fixed_bases = BTreeMap::new();
            fixed_bases.insert(String::from("com_instance"), C::identity());
            fixed_bases.extend(midnight_circuits::verifier::fixed_bases::<S>(
                name.as_str(),
                &vk,
            ));

            Self {
                level,
                name,
                vk: Arc::new(vk),
                pk: Arc::new(pk),
                vk_data,
                fixed_bases,
            }
        }
    }

    #[derive(Clone)]
    struct AggKeyStore {
        levels: Vec<AggLevelKeys>,
    }

    impl AggKeyStore {
        fn new(levels: Vec<AggLevelKeys>) -> Self {
            assert!(!levels.is_empty(), "AggKeyStore cannot be empty");

            let mut seen_names = BTreeSet::new();
            for (i, lvl) in levels.iter().enumerate() {
                let expected_level = i + 1;
                assert!(
                    lvl.level == expected_level,
                    "AggKeyStore level mismatch at index {}: expected {}, got {}",
                    i,
                    expected_level,
                    lvl.level
                );
                assert!(
                    seen_names.insert(lvl.name.clone()),
                    "Duplicate vk_name: '{}'",
                    lvl.name
                );
            }
            Self { levels }
        }

        fn max_level(&self) -> usize {
            self.levels.len()
        }

        fn get(&self, level: usize) -> &AggLevelKeys {
            assert!(
                level >= 1 && level <= self.levels.len(),
                "Agg level out of range"
            );
            &self.levels[level - 1]
        }
    }

    fn keygen_vk_pk<Circ: Circuit<F>>(srs: &ParamsKZG<Bls12>, circuit: &Circ, k: u32) -> (Vk, Pk) {
        let vk = keygen_vk_with_k(srs, circuit, k).expect("keygen_vk_with_k failed");
        let pk = keygen_pk(vk.clone(), circuit).expect("keygen_pk failed");
        (vk, pk)
    }

    /// Cached aggregation keys & supporting data. Compute once (e.g., at program start),
    /// then reuse for every batch.
    #[derive(Clone)]
    pub struct AggSetup {
        // Inputs
        leaf_vk_name: String,
        num_leaves: usize,
        max_agg_level: usize,

        // Derived/cached
        leaf_vk_data: VkData,

        agg_srs_leaf: ParamsKZG<Bls12>,
        agg_srs_internal: ParamsKZG<Bls12>,
        agg_store: AggKeyStore,

        leaf_fixed_bases: BTreeMap<String, C>,
        fixed_base_names: Vec<String>,
        fixed_bases: BTreeMap<String, C>,
        trivial_combined: Accumulator<S>,

        child_vk: (EvaluationDomain<F>, ConstraintSystem<F>, F),
        child_vk_name: String,
        child_level: usize,
    }

    impl AggSetup {
        pub fn child_vk(&self) -> &(EvaluationDomain<F>, ConstraintSystem<F>, F) {
            &self.child_vk
        }
        pub fn child_vk_name(&self) -> &str {
            &self.child_vk_name
        }
        pub fn child_level(&self) -> usize {
            self.child_level
        }
        pub fn fixed_base_names(&self) -> &[String] {
            &self.fixed_base_names
        }
    }

    /// Prepare and cache all aggregation keys once, for a fixed `num_leaves` (batch size).
    pub fn prepare_agg_setup(
        leaf_srs: &ParamsKZG<Bls12>,
        leaf_vk: &VerifyingKey<F, KZGCommitmentScheme<E>>,
        leaf_vk_name: &str,
        leaf_k: u32,
        num_leaves: usize,
    ) -> AggSetup {
        assert!(num_leaves > 0, "Need at least one client proof");
        assert!(
            num_leaves.is_power_of_two(),
            "Client proofs must be power of two"
        );

        let max_level: usize = (num_leaves as u32).trailing_zeros() as usize;
        assert!(
            max_level >= 2,
            "Merged final agg requires at least 4 client proofs"
        );
        let max_agg_level: usize = max_level - 1;

        let leaf_vk_data = VkData {
            domain: EvaluationDomain::new(leaf_vk.cs().degree() as u32, leaf_k),
            cs: leaf_vk.cs().clone(),
            transcript_repr: leaf_vk.transcript_repr(),
        };

        let mut agg_cs = ConstraintSystem::default();
        configure_agg_circuit(&mut agg_cs);

        let agg_srs_leaf = filecoin_srs_agg(K_LEAF).unwrap();
        let agg_srs_internal = filecoin_srs_agg(K_INTERNAL).unwrap();

        assert_eq!(leaf_srs.s_g2(), agg_srs_internal.s_g2(), "s_g2 mismatch");
        assert_eq!(
            agg_srs_leaf.s_g2(),
            agg_srs_internal.s_g2(),
            "s_g2 mismatch"
        );

        let agg_vk_names: Vec<String> = (1..=max_agg_level).map(agg_vk_name_for_level).collect();

        let fixed_base_names: Vec<String> = {
            let mut set = BTreeSet::new();
            let mut out = Vec::new();

            for name in fixed_base_names_for(leaf_vk_name, &leaf_vk_data.cs) {
                if set.insert(name.clone()) {
                    out.push(name);
                }
            }
            for vk_name in agg_vk_names.iter() {
                for name in fixed_base_names_for(vk_name.as_str(), &agg_cs) {
                    if set.insert(name.clone()) {
                        out.push(name);
                    }
                }
            }
            out
        };

        let mut agg_levels: Vec<AggLevelKeys> = Vec::with_capacity(max_agg_level);

        for level in 1..=max_agg_level {
            let (child_vk, child_vk_name, is_leaf) = if level == 1 {
                (leaf_vk_data.clone(), leaf_vk_name.to_string(), true)
            } else {
                let child_level = level - 1;
                let child = agg_levels[child_level - 1].vk_data.clone();
                let child_name = agg_vk_names[child_level - 1].clone();
                (child, child_name, false)
            };

            let name = agg_vk_names[level - 1].clone();
            let start = Instant::now();

            if level == 1 {
                let default_circuit = LeafAggCircuit {
                    child_vk,
                    child_vk_name,
                    left_child_state: array::from_fn(|_| Value::unknown()),
                    right_child_state: array::from_fn(|_| Value::unknown()),
                    left_items: Value::unknown(),
                    right_items: Value::unknown(),
                    pre_commitment_map: Value::unknown(),
                    pre_nullifier_map: Value::unknown(),
                    pre_commitment_roots_map: Value::unknown(),
                    left_proof: Value::unknown(),
                    right_proof: Value::unknown(),
                    left_acc: Value::unknown(),
                    right_acc: Value::unknown(),
                    fixed_base_names: fixed_base_names.clone(),
                    is_leaf,
                };
                let (vk, pk) = keygen_vk_pk(&agg_srs_leaf, &default_circuit, K_LEAF);
                println!("Computed {} vk/pk in {:?}", name, start.elapsed());
                agg_levels.push(AggLevelKeys::new(level, name, vk, pk));
            } else {
                let default_circuit = InternalAggCircuit {
                    child_vk,
                    child_vk_name,
                    left_child_state: array::from_fn(|_| Value::unknown()),
                    right_child_state: array::from_fn(|_| Value::unknown()),
                    left_items: Value::unknown(),
                    right_items: Value::unknown(),
                    pre_commitment_map: Value::unknown(),
                    pre_nullifier_map: Value::unknown(),
                    pre_commitment_roots_map: Value::unknown(),
                    left_proof: Value::unknown(),
                    right_proof: Value::unknown(),
                    left_acc: Value::unknown(),
                    right_acc: Value::unknown(),
                    fixed_base_names: fixed_base_names.clone(),
                    is_leaf,
                };
                let (vk, pk) = keygen_vk_pk(&agg_srs_internal, &default_circuit, K_INTERNAL);
                println!("Computed {} vk/pk in {:?}", name, start.elapsed());
                agg_levels.push(AggLevelKeys::new(level, name, vk, pk));
            }
        }

        let agg_store = AggKeyStore::new(agg_levels);

        let mut leaf_fixed_bases = BTreeMap::new();
        leaf_fixed_bases.insert(String::from("com_instance"), C::identity());
        leaf_fixed_bases.extend(midnight_circuits::verifier::fixed_bases::<S>(
            leaf_vk_name,
            leaf_vk,
        ));

        let mut fixed_bases = BTreeMap::new();
        fixed_bases.extend(leaf_fixed_bases.iter().map(|(k, v)| (k.clone(), *v)));
        for level in 1..=agg_store.max_level() {
            fixed_bases.extend(
                agg_store
                    .get(level)
                    .fixed_bases
                    .iter()
                    .map(|(k, v)| (k.clone(), *v)),
            );
        }

        let trivial_leaf =
            trivial_acc_with_names(&fixed_base_names_for(leaf_vk_name, &leaf_vk_data.cs));
        let mut trivial_all: Vec<Accumulator<S>> = vec![trivial_leaf];
        for level in 1..=agg_store.max_level() {
            let vk_name = agg_store.get(level).name.as_str();
            let cs = agg_store.get(level).vk.cs();
            trivial_all.push(trivial_acc_with_names(&fixed_base_names_for(vk_name, cs)));
        }
        let mut trivial_combined = Accumulator::accumulate(&trivial_all);
        trivial_combined.collapse();

        let child_level = max_agg_level;
        let child_keys = agg_store.get(child_level);
        let child_vk = (
            child_keys.vk_data.domain.clone(),
            child_keys.vk_data.cs.clone(),
            child_keys.vk_data.transcript_repr,
        );

        AggSetup {
            leaf_vk_name: leaf_vk_name.to_string(),
            num_leaves,
            max_agg_level,

            leaf_vk_data,

            agg_srs_leaf,
            agg_srs_internal,
            agg_store: agg_store.clone(),

            leaf_fixed_bases,
            fixed_base_names,
            fixed_bases,
            trivial_combined,

            child_vk,
            child_vk_name: child_keys.name.clone(),
            child_level,
        }
    }

    /// Aggregation using cached keys (`AggSetup`). No vk/pk computation occurs here.
    ///
    /// NOTE: Added `pre_commitment_roots_map` so leaf circuits can accept “lagging” tx roots
    /// (proofs that reference any historic confirmed root).
    pub fn aggregate_client_proofs_cached(
        setup: &AggSetup,
        leaf_srs: &ParamsKZG<Bls12>,
        leaf_vk: &VerifyingKey<F, KZGCommitmentScheme<E>>,
        client_proofs: &[ClientProof],
        pre_commitment_map: Map,
        pre_nullifier_map: Map,
        pre_commitment_roots_map: Map,
    ) -> AggregationResult {
        use midnight_circuits::instructions::hash::HashCPU;

        println!(
            "[agg] start c_pre(map) = {:?}",
            pre_commitment_map.succinct_repr()
        );
        println!(
            "[agg] start n_pre(map) = {:?}",
            pre_nullifier_map.succinct_repr()
        );
        println!(
            "[agg] client_proofs[0].root_before = {:?}",
            client_proofs[0].public_items[0]
        );

        assert_eq!(
            client_proofs.len(),
            setup.num_leaves,
            "client_proofs len must match cached setup (expected {}, got {})",
            setup.num_leaves,
            client_proofs.len()
        );
        assert!(!client_proofs.is_empty(), "Need at least one client proof");
        assert!(
            client_proofs.len().is_power_of_two(),
            "Client proofs must be power of two"
        );

        let num_leaves = client_proofs.len();
        let max_level: usize = (num_leaves as u32).trailing_zeros() as usize;
        assert!(
            max_level >= 2,
            "Merged final agg requires at least 4 client proofs"
        );
        let max_agg_level: usize = max_level - 1;
        assert_eq!(
            max_agg_level, setup.max_agg_level,
            "max_agg_level mismatch with cached setup"
        );

        let agg_srs1 = &setup.agg_srs_leaf;
        let agg_srs2 = &setup.agg_srs_internal;

        let combined_fixed_base_names = setup.fixed_base_names.clone();
        let combined_fixed_bases = setup.fixed_bases.clone();
        let trivial_combined = setup.trivial_combined.clone();

        // ---- FIX (Issue 1): bind a single roots-set root across whole agg tree (host side)
        let batch_roots_set_root = pre_commitment_roots_map.succinct_repr();

        println!("\nCreating {} leaf AGG nodes...", num_leaves / 2);

        let leaf_level = 1usize;
        let leaf_keys = setup.agg_store.get(leaf_level);
        let leaf_agg_vk_name = leaf_keys.name.clone();

        let mut rolling_commit_map = pre_commitment_map.clone();
        let mut rolling_null_map = pre_nullifier_map.clone();

        let mut leaf_plans: Vec<LeafPlan> = Vec::with_capacity(num_leaves / 2);

        for i in 0..num_leaves / 2 {
            let left = &client_proofs[i * 2];
            let right = &client_proofs[i * 2 + 1];

            let c_pre = rolling_commit_map.succinct_repr();
            let n_pre = rolling_null_map.succinct_repr();

            // NEW (host-side sanity): both tx roots must be in the historic roots set
            assert_eq!(
                pre_commitment_roots_map.get(&left.public_items[0]),
                F::ONE,
                "leaf {} left tx root not in historic roots set",
                i
            );
            assert_eq!(
                pre_commitment_roots_map.get(&right.public_items[0]),
                F::ONE,
                "leaf {} right tx root not in historic roots set",
                i
            );

            let pre_commit_map_for_leaf = rolling_commit_map.clone();
            let pre_null_map_for_leaf = rolling_null_map.clone();
            let pre_roots_map_for_leaf = pre_commitment_roots_map.clone();

            // Apply BOTH tx’s effects to the rolling rollup state (independent of tx-root in proofs)
            rolling_commit_map.insert(&left.public_items[3], &F::ONE);
            rolling_commit_map.insert(&left.public_items[4], &F::ONE);

            for nf in [left.public_items[5], left.public_items[6]] {
                let _old = rolling_null_map.get(&nf);
                //assert_eq!(old, F::ZERO, "leaf {} left nf already spent", i);
                rolling_null_map.insert(&nf, &F::ONE);
            }

            rolling_commit_map.insert(&right.public_items[3], &F::ONE);
            rolling_commit_map.insert(&right.public_items[4], &F::ONE);

            for nf in [right.public_items[5], right.public_items[6]] {
                let _old = rolling_null_map.get(&nf);
                //assert_eq!(old, F::ZERO, "leaf {} right nf already spent", i);
                rolling_null_map.insert(&nf, &F::ONE);
            }

            let c_post = rolling_commit_map.succinct_repr();
            let n_post = rolling_null_map.succinct_repr();

            let subroot = <PoseidonChip<F> as HashCPU<F, F>>::hash(&[left.state, right.state]);

            let expected_state = AggState {
                c_pre,
                c_post,
                n_pre,
                n_post,
                subroot,
                roots_set_root: batch_roots_set_root, // ---- FIX (Issue 1)
            };

            leaf_plans.push(LeafPlan {
                i,
                left_items: left.public_items,
                right_items: right.public_items,
                pre_commitment_map: SendableMap(pre_commit_map_for_leaf),
                pre_nullifier_map: SendableMap(pre_null_map_for_leaf),
                pre_roots_map: SendableMap(pre_roots_map_for_leaf),
                expected_state,
                left_state: left.state,
                right_state: right.state,
                left_proof: left.proof.clone(),
                right_proof: right.proof.clone(),
            });
        }

        leaf_plans.par_iter().for_each(|p| {
            let inst_l = host_instance_hash(p.left_items);
            let inst_r = host_instance_hash(p.right_items);
            assert_eq!(
                inst_l, p.left_state,
                "left client instance mismatch (leaf {})",
                p.i
            );
            assert_eq!(
                inst_r, p.right_state,
                "right client instance mismatch (leaf {})",
                p.i
            );

            let subroot = <PoseidonChip<F> as HashCPU<F, F>>::hash(&[inst_l, inst_r]);
            assert_eq!(
                subroot, p.expected_state.subroot,
                "leaf {} planned subroot mismatch",
                p.i
            );
        });

        let leaf_vk_data_cl = setup.leaf_vk_data.clone();
        let leaf_vk_name_string = setup.leaf_vk_name.clone();
        let leaf_pk = leaf_keys.pk.clone();
        let leaf_vk_arc = leaf_keys.vk.clone();

        let leaf_fixed_bases = setup.leaf_fixed_bases.clone();

        let mut current_level: Vec<TreeNode> = leaf_plans
            .par_iter()
            .map(|p| {
                let state = p.expected_state;

                let circuit = LeafAggCircuit {
                    child_vk: leaf_vk_data_cl.clone(),
                    child_vk_name: leaf_vk_name_string.clone(),
                    left_child_state: array::from_fn(|_| Value::unknown()),
                    right_child_state: array::from_fn(|_| Value::unknown()),
                    left_items: Value::known(p.left_items),
                    right_items: Value::known(p.right_items),
                    pre_commitment_map: Value::known(p.pre_commitment_map.clone_inner()),
                    pre_nullifier_map: Value::known(p.pre_nullifier_map.clone_inner()),
                    pre_commitment_roots_map: Value::known(p.pre_roots_map.clone_inner()),
                    left_proof: Value::known(p.left_proof.clone()),
                    right_proof: Value::known(p.right_proof.clone()),
                    left_acc: Value::known(trivial_combined.clone()),
                    right_acc: Value::known(trivial_combined.clone()),
                    fixed_base_names: combined_fixed_base_names.clone(),
                    is_leaf: true,
                };

                let proof_acc_left = verify_and_extract_acc(
                    leaf_srs,
                    leaf_vk,
                    &leaf_fixed_bases,
                    &p.left_proof,
                    &[p.left_state],
                );
                let proof_acc_right = verify_and_extract_acc(
                    leaf_srs,
                    leaf_vk,
                    &leaf_fixed_bases,
                    &p.right_proof,
                    &[p.right_state],
                );

                let mut accumulated_pi = Accumulator::accumulate(&[
                    proof_acc_left,
                    trivial_combined.clone(),
                    proof_acc_right,
                    trivial_combined.clone(),
                ]);
                accumulated_pi.collapse();

                let public_inputs = AggPublicInputs {
                    state,
                    pi_acc: accumulated_pi.clone(),
                };
                let public_inputs_fields = public_inputs.to_fields();

                let start = Instant::now();
                let proof = {
                    let mut transcript = CircuitTranscript::<PoseidonState<F>>::init();
                    create_proof::<
                        F,
                        KZGCommitmentScheme<E>,
                        CircuitTranscript<PoseidonState<F>>,
                        LeafAggCircuit,
                    >(
                        agg_srs1,
                        leaf_pk.as_ref(),
                        &[circuit],
                        1,
                        &[&[&[], &public_inputs_fields]],
                        OsRng,
                        &mut transcript,
                    )
                    .expect("Leaf AGG proof failed");
                    transcript.finalize()
                };

                println!("proof size (bytes): {}", proof.len());
                println!(
                    "Leaf AGG {} ({}) created in {:?}",
                    p.i,
                    leaf_agg_vk_name,
                    start.elapsed()
                );

                assert!(
                    accumulated_pi.check(&agg_srs2.s_g2().into(), &combined_fixed_bases),
                    "Leaf node {}: accumulated PI accumulator did not check",
                    p.i
                );

                let proof_acc = verify_and_extract_acc(
                    agg_srs1,
                    leaf_vk_arc.as_ref(),
                    &leaf_keys.fixed_bases,
                    &proof,
                    &public_inputs_fields,
                );

                TreeNode {
                    state,
                    proof,
                    proof_acc,
                    pi_acc: accumulated_pi,
                }
            })
            .collect();

        let mut child_level: usize = 1;
        while current_level.len() > 2 {
            let parent_level = child_level + 1;
            let parent_keys = setup.agg_store.get(parent_level);
            let parent_vk_name = parent_keys.name.clone();

            println!(
                "\nBuilding AGG level {} ({}) with {} nodes...",
                parent_level,
                parent_vk_name,
                current_level.len() / 2
            );

            let child_keys = setup.agg_store.get(child_level);
            let child_vk_data = child_keys.vk_data.clone();
            let child_vk_name = child_keys.name.clone();

            let parent_pk = parent_keys.pk.clone();
            let parent_vk = parent_keys.vk.clone();

            let next_level: Vec<TreeNode> = current_level
                .par_chunks(2)
                .enumerate()
                .map(|(i, pair)| {
                    let left = &pair[0];
                    let right = &pair[1];

                    assert_eq!(
                        left.state.c_post, right.state.c_pre,
                        "commit boundary mismatch"
                    );
                    assert_eq!(
                        left.state.n_post, right.state.n_pre,
                        "null boundary mismatch"
                    );
                    // ---- FIX (Issue 1): roots-set root must match across children
                    assert_eq!(
                        left.state.roots_set_root, right.state.roots_set_root,
                        "roots_set_root mismatch"
                    );

                    let state = AggState {
                        c_pre: left.state.c_pre,
                        c_post: right.state.c_post,
                        n_pre: left.state.n_pre,
                        n_post: right.state.n_post,
                        subroot:
                            <PoseidonChip<F> as midnight_circuits::instructions::hash::HashCPU<
                                F,
                                F,
                            >>::hash(&[
                                left.state.subroot,
                                right.state.subroot,
                            ]),
                        roots_set_root: left.state.roots_set_root,
                    };

                    let l_fields = left.state.to_fields();
                    let r_fields = right.state.to_fields();

                    let circuit = InternalAggCircuit {
                        child_vk: child_vk_data.clone(),
                        child_vk_name: child_vk_name.clone(),
                        left_child_state: array::from_fn(|j| Value::known(l_fields[j])),
                        right_child_state: array::from_fn(|j| Value::known(r_fields[j])),
                        left_items: Value::unknown(),
                        right_items: Value::unknown(),
                        pre_commitment_map: Value::unknown(),
                        pre_nullifier_map: Value::unknown(),
                        pre_commitment_roots_map: Value::unknown(),
                        left_proof: Value::known(left.proof.clone()),
                        right_proof: Value::known(right.proof.clone()),
                        left_acc: Value::known(left.pi_acc.clone()),
                        right_acc: Value::known(right.pi_acc.clone()),
                        fixed_base_names: combined_fixed_base_names.clone(),
                        is_leaf: false,
                    };

                    let mut accumulated_pi = Accumulator::accumulate(&[
                        left.proof_acc.clone(),
                        left.pi_acc.clone(),
                        right.proof_acc.clone(),
                        right.pi_acc.clone(),
                    ]);
                    accumulated_pi.collapse();

                    let public_inputs = AggPublicInputs {
                        state,
                        pi_acc: accumulated_pi.clone(),
                    };
                    let public_inputs_fields = public_inputs.to_fields();

                    let start = Instant::now();
                    let proof = {
                        let mut transcript = CircuitTranscript::<PoseidonState<F>>::init();
                        create_proof::<
                            F,
                            KZGCommitmentScheme<E>,
                            CircuitTranscript<PoseidonState<F>>,
                            InternalAggCircuit,
                        >(
                            agg_srs2,
                            parent_pk.as_ref(),
                            &[circuit],
                            1,
                            &[&[&[], &public_inputs_fields]],
                            OsRng,
                            &mut transcript,
                        )
                        .expect("Internal AGG proof failed");
                        transcript.finalize()
                    };

                    println!(
                        "Level {} node {} ({}) created in {:?}",
                        parent_level,
                        i,
                        parent_vk_name,
                        start.elapsed()
                    );

                    assert!(
                        accumulated_pi.check(&agg_srs2.s_g2().into(), &combined_fixed_bases),
                        "Level {parent_level} node {i}: accumulated PI accumulator did not check"
                    );

                    let proof_acc = verify_and_extract_acc(
                        agg_srs2,
                        parent_vk.as_ref(),
                        &parent_keys.fixed_bases,
                        &proof,
                        &public_inputs_fields,
                    );

                    TreeNode {
                        state,
                        proof,
                        proof_acc,
                        pi_acc: accumulated_pi,
                    }
                })
                .collect();

            current_level = next_level;
            child_level = parent_level;
        }

        assert_eq!(current_level.len(), 2, "expected to stop at top pair");

        let left_top = current_level[0].clone();
        let right_top = current_level[1].clone();

        assert_eq!(
            left_top.state.c_post, right_top.state.c_pre,
            "top commit boundary mismatch"
        );
        assert_eq!(
            left_top.state.n_post, right_top.state.n_pre,
            "top null boundary mismatch"
        );
        // ---- FIX (Issue 1): roots-set root must match at the top as well
        assert_eq!(
            left_top.state.roots_set_root, right_top.state.roots_set_root,
            "top roots_set_root mismatch"
        );

        let root_subroot = <PoseidonChip<F> as HashCPU<F, F>>::hash(&[
            left_top.state.subroot,
            right_top.state.subroot,
        ]);

        let root_state = AggState {
            c_pre: left_top.state.c_pre,
            c_post: right_top.state.c_post,
            n_pre: left_top.state.n_pre,
            n_post: right_top.state.n_post,
            subroot: root_subroot,
            roots_set_root: left_top.state.roots_set_root,
        };

        let leaf_states: Vec<F> = client_proofs.iter().map(|p| p.state).collect();
        let expected_root = poseidon_tree_root(&leaf_states);
        assert_eq!(
            root_state.subroot, expected_root,
            "Root subroot mismatch with recomputed Poseidon tree root"
        );

        let child_keys = setup.agg_store.get(child_level);
        let child_vk_tuple = (
            child_keys.vk_data.domain.clone(),
            child_keys.vk_data.cs.clone(),
            child_keys.vk_data.transcript_repr,
        );

        AggregationResult {
            root_state,
            left_top,
            right_top,
            child_vk: child_vk_tuple,
            child_vk_name: child_keys.name.clone(),
            child_level,
            fixed_base_names: combined_fixed_base_names,
            fixed_bases: combined_fixed_bases,
        }
    }

    #[derive(Clone, Debug)]
    pub struct FinalAggCircuit {
        pub child_vk: (EvaluationDomain<F>, ConstraintSystem<F>, F),
        pub child_vk_name: String,
        pub child_level: F,

        pub left_proof: Value<Vec<u8>>,
        pub right_proof: Value<Vec<u8>>,
        pub left_pi_acc: Value<AggAccumulator>,
        pub right_pi_acc: Value<AggAccumulator>,
        pub fixed_base_names: Vec<String>,

        pub left_child_state: Value<AggState>,
        pub right_child_state: Value<AggState>,

        pub agg_state: Value<AggState>,

        pub pre_commitment_roots_map: Value<Map>,
        pub post_commitment_roots_root: Value<F>,
    }

    impl Circuit<F> for FinalAggCircuit {
        type Config = (
            NativeConfig,
            P2RDecompositionConfig,
            ForeignEccConfig<C>,
            PoseidonConfig<F>,
            //Sha256Config,
        );
        type FloorPlanner = SimpleFloorPlanner;
        type Params = ();

        fn without_witnesses(&self) -> Self {
            Self {
                child_vk: self.child_vk.clone(),
                child_vk_name: self.child_vk_name.clone(),
                child_level: self.child_level,
                left_proof: Value::unknown(),
                right_proof: Value::unknown(),
                left_pi_acc: Value::unknown(),
                right_pi_acc: Value::unknown(),
                fixed_base_names: self.fixed_base_names.clone(),
                left_child_state: Value::unknown(),
                right_child_state: Value::unknown(),
                agg_state: Value::unknown(),
                pre_commitment_roots_map: Value::unknown(),
                post_commitment_roots_root: Value::unknown(),
            }
        }

        fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
            configure_agg_circuit(meta)
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            let native_chip = <NativeChip<F> as ComposableChip<F>>::new(&config.0, &());
            let core_decomp_chip = P2RDecompositionChip::new(&config.1, &(AGG_K as usize - 1));
            let scalar_chip = NativeGadget::new(core_decomp_chip.clone(), native_chip.clone());
            let curve_chip = ForeignEccChip::new(&config.2, &scalar_chip, &scalar_chip);
            let poseidon_chip = PoseidonChip::new(&config.3, &native_chip);
            let verifier_chip: VerifierGadget<_> =
                VerifierGadget::<S>::new(&curve_chip, &scalar_chip, &poseidon_chip);

            let c_pre =
                scalar_chip.assign(&mut layouter, self.agg_state.clone().map(|s| s.c_pre))?;
            scalar_chip.constrain_as_public_input(&mut layouter, &c_pre)?;
            let c_post =
                scalar_chip.assign(&mut layouter, self.agg_state.clone().map(|s| s.c_post))?;
            scalar_chip.constrain_as_public_input(&mut layouter, &c_post)?;
            let n_pre: AssignedNative<F> =
                scalar_chip.assign(&mut layouter, self.agg_state.clone().map(|s| s.n_pre))?;
            scalar_chip.constrain_as_public_input(&mut layouter, &n_pre)?;
            let n_post: AssignedNative<F> =
                scalar_chip.assign(&mut layouter, self.agg_state.clone().map(|s| s.n_post))?;
            scalar_chip.constrain_as_public_input(&mut layouter, &n_post)?;

            let l_c_pre = scalar_chip.assign(
                &mut layouter,
                self.left_child_state.clone().map(|s| s.c_pre),
            )?;
            let l_c_post: AssignedNative<F> = scalar_chip.assign(
                &mut layouter,
                self.left_child_state.clone().map(|s| s.c_post),
            )?;
            let l_n_pre = scalar_chip.assign(
                &mut layouter,
                self.left_child_state.clone().map(|s| s.n_pre),
            )?;
            let l_n_post: AssignedNative<F> = scalar_chip.assign(
                &mut layouter,
                self.left_child_state.clone().map(|s| s.n_post),
            )?;
            let l_subroot: AssignedNative<F> = scalar_chip.assign(
                &mut layouter,
                self.left_child_state.clone().map(|s| s.subroot),
            )?;
            // ---- FIX (Issue 1): read roots_set_root from child state
            let l_roots_set_root: AssignedNative<F> = scalar_chip.assign(
                &mut layouter,
                self.left_child_state.clone().map(|s| s.roots_set_root),
            )?;

            let r_c_pre = scalar_chip.assign(
                &mut layouter,
                self.right_child_state.clone().map(|s| s.c_pre),
            )?;
            let r_c_post = scalar_chip.assign(
                &mut layouter,
                self.right_child_state.clone().map(|s| s.c_post),
            )?;
            let r_n_pre = scalar_chip.assign(
                &mut layouter,
                self.right_child_state.clone().map(|s| s.n_pre),
            )?;
            let r_n_post = scalar_chip.assign(
                &mut layouter,
                self.right_child_state.clone().map(|s| s.n_post),
            )?;
            let r_subroot: AssignedNative<F> = scalar_chip.assign(
                &mut layouter,
                self.right_child_state.clone().map(|s| s.subroot),
            )?;
            // ---- FIX (Issue 1): read roots_set_root from child state
            let r_roots_set_root: AssignedNative<F> = scalar_chip.assign(
                &mut layouter,
                self.right_child_state.clone().map(|s| s.roots_set_root),
            )?;

            scalar_chip.assert_equal(&mut layouter, &l_c_post, &r_c_pre)?;
            scalar_chip.assert_equal(&mut layouter, &l_n_post, &r_n_pre)?;

            scalar_chip.assert_equal(&mut layouter, &c_pre, &l_c_pre)?;
            scalar_chip.assert_equal(&mut layouter, &c_post, &r_c_post)?;
            scalar_chip.assert_equal(&mut layouter, &n_pre, &l_n_pre)?;
            scalar_chip.assert_equal(&mut layouter, &n_post, &r_n_post)?;

            let subroot =
                poseidon_chip.hash(&mut layouter, &[l_subroot.clone(), r_subroot.clone()])?;
            scalar_chip.constrain_as_public_input(&mut layouter, &subroot)?;

            let one = scalar_chip.assign_fixed(&mut layouter, F::ONE)?;

            let mut roots_map_gadget = midnight_circuits::map::map_gadget::MapGadget::<
                F,
                NG,
                PoseidonChip<F>,
            >::new(&scalar_chip, &poseidon_chip);
            roots_map_gadget.init(&mut layouter, self.pre_commitment_roots_map.clone())?;

            let pre_roots_set_root = roots_map_gadget.succinct_repr();
            scalar_chip.constrain_as_public_input(&mut layouter, &pre_roots_set_root)?;

            // ---- FIX (Issue 1): child agg proofs must be bound to THIS roots set
            scalar_chip.assert_equal(&mut layouter, &l_roots_set_root, &pre_roots_set_root)?;
            scalar_chip.assert_equal(&mut layouter, &r_roots_set_root, &pre_roots_set_root)?;

            let expected_post_roots_set_root =
                scalar_chip.assign(&mut layouter, self.post_commitment_roots_root.clone())?;
            scalar_chip.constrain_as_public_input(&mut layouter, &expected_post_roots_set_root)?;

            let pre_ok = roots_map_gadget.get(&mut layouter, &c_pre)?;
            scalar_chip.assert_equal(&mut layouter, &pre_ok, &one)?;

            // Prevent replay: post root must be new to the set
            let zero = scalar_chip.assign_fixed(&mut layouter, F::ZERO)?;
            let already = roots_map_gadget.get(&mut layouter, &c_post)?;
            scalar_chip.assert_equal(&mut layouter, &already, &zero)?;

            roots_map_gadget.insert(&mut layouter, &c_post, &one)?;
            scalar_chip.assert_equal(
                &mut layouter,
                &roots_map_gadget.succinct_repr(),
                &expected_post_roots_set_root,
            )?;

            let vk_val: AssignedNative<F> =
                native_chip.assign_fixed(&mut layouter, self.child_vk.2)?;
            let assigned_vk = verifier_chip.assign_vk(
                self.child_vk_name.as_str(),
                &self.child_vk.0,
                &self.child_vk.1,
                vk_val,
            )?;

            let mut left_pi_acc = AssignedAccumulator::assign(
                &mut layouter,
                &curve_chip,
                &scalar_chip,
                1,
                1,
                &[],
                &self.fixed_base_names,
                self.left_pi_acc.clone(),
            )?;
            left_pi_acc.collapse(&mut layouter, &curve_chip, &scalar_chip)?;

            let mut right_pi_acc = AssignedAccumulator::assign(
                &mut layouter,
                &curve_chip,
                &scalar_chip,
                1,
                1,
                &[],
                &self.fixed_base_names,
                self.right_pi_acc.clone(),
            )?;
            right_pi_acc.collapse(&mut layouter, &curve_chip, &scalar_chip)?;

            let mut left_pi: Vec<AssignedNative<F>> = Vec::new();
            left_pi.push(l_c_pre.clone());
            left_pi.push(l_c_post.clone());
            left_pi.push(l_n_pre.clone());
            left_pi.push(l_n_post.clone());
            left_pi.push(l_subroot.clone());
            left_pi.push(l_roots_set_root.clone()); // ---- FIX (Issue 1): include in PI
            left_pi.extend(verifier_chip.as_public_input(&mut layouter, &left_pi_acc)?);

            let mut right_pi: Vec<AssignedNative<F>> = Vec::new();
            right_pi.push(r_c_pre.clone());
            right_pi.push(r_c_post.clone());
            right_pi.push(r_n_pre.clone());
            right_pi.push(r_n_post.clone());
            right_pi.push(r_subroot.clone());
            right_pi.push(r_roots_set_root.clone()); // ---- FIX (Issue 1): include in PI
            right_pi.extend(verifier_chip.as_public_input(&mut layouter, &right_pi_acc)?);

            let id_point: AssignedForeignPoint<
                midnight_curves::Fq,
                midnight_curves::G1Projective,
                midnight_curves::G1Projective,
            > = curve_chip.assign_fixed(&mut layouter, C::identity())?;

            let mut left_proof_acc = verifier_chip.prepare(
                &mut layouter,
                &assigned_vk,
                &[("com_instance", id_point.clone())],
                &[&left_pi],
                self.left_proof.clone(),
            )?;
            left_proof_acc.collapse(&mut layouter, &curve_chip, &scalar_chip)?;

            let mut right_proof_acc = verifier_chip.prepare(
                &mut layouter,
                &assigned_vk,
                &[("com_instance", id_point)],
                &[&right_pi],
                self.right_proof.clone(),
            )?;
            right_proof_acc.collapse(&mut layouter, &curve_chip, &scalar_chip)?;

            let mut final_acc = AssignedAccumulator::<S>::accumulate(
                &mut layouter,
                &verifier_chip,
                &scalar_chip,
                &poseidon_chip,
                &[left_proof_acc, left_pi_acc, right_proof_acc, right_pi_acc],
            )?;
            final_acc.collapse(&mut layouter, &curve_chip, &scalar_chip)?;

            let final_acc_pi = verifier_chip.as_public_input(&mut layouter, &final_acc)?;
            for x in final_acc_pi.iter() {
                scalar_chip.constrain_as_public_input(&mut layouter, x)?;
            }

            core_decomp_chip.load(&mut layouter)
        }
    }
}

use proof_agg::{
    AggAccumulator, ClientProof as AggClientProof, FinalAggCircuit, accumulator_as_public_input,
    aggregate_client_proofs_cached, filecoin_srs_agg, prepare_agg_setup,
};

const UTXO_COMMIT_TAG: u64 = 0x0001;
const UTXO_NULLIFY_TAG: u64 = 0x0002;
const AMOUNT_BITS: u32 = 128;
const AMOUNT_GEN_BITS: u32 = 120;
const BATCH_SIZE: usize = 4;

// NEW: probability that a client proof is generated against an older confirmed root
const LAG_TX_PROB: f64 = 0.35;

#[derive(Clone, Debug)]
pub struct Utxo {
    pub asset_id: F,
    pub amount: u128,
    pub randomness: F,
}

#[derive(Clone, Default)]
pub struct Spend2Output2;

impl Relation for Spend2Output2 {
    type Instance = F;

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
        Ok(vec![*instance])
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

        let instance_hash = std_lib.poseidon(
            layouter,
            &[
                root.clone(),
                pk_bx.clone(),
                pk_by.clone(),
                new_c1.clone(),
                new_c2.clone(),
                nf1.clone(),
                nf2.clone(),
            ],
        )?;

        std_lib.constrain_as_public_input(layouter, &instance_hash)?;
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

fn host_commit(id: F, amt_u128: u128, pk_x: F, pk_y: F, rand: F) -> F {
    let tag = F::from(UTXO_COMMIT_TAG);
    let amt_f = F::from_u128(amt_u128);
    let h1 = <PoseidonChip<F> as HashCPU<F, F>>::hash(&[tag, id, amt_f]);
    let h2 = <PoseidonChip<F> as HashCPU<F, F>>::hash(&[pk_x, pk_y, rand]);
    <PoseidonChip<F> as HashCPU<F, F>>::hash(&[h1, h2, F::ZERO])
}

fn host_nullify(commit: F, pk_x: F, pk_y: F) -> F {
    let tag = F::from(UTXO_NULLIFY_TAG);
    let h = <PoseidonChip<F> as HashCPU<F, F>>::hash(&[tag, commit, pk_x]);
    <PoseidonChip<F> as HashCPU<F, F>>::hash(&[h, pk_y, F::ZERO])
}

// ✅ Single Poseidon hash of all 7 would-be public inputs (host-side)
fn host_instance_hash(items: [F; 7]) -> F {
    <PoseidonChip<F> as HashCPU<F, F>>::hash(&items)
}

#[derive(Clone, Debug)]
struct Note {
    utxo: Utxo,
    commit: F,
    spent: bool,
    confirmed_at_root_idx: usize,
}

#[derive(Clone)]
struct Account {
    id: usize,
    sk: JubjubScalar,
    pk_point: JubjubSubgroup,
    pk_x: F,
    pk_y: F,
    wallet: Vec<Note>,
}

fn main() {
    const LEAF_VK_NAME: &str = "spend2output2_vk";

    const K: u32 = 14;
    const NUM_ACCOUNTS: usize = 4;
    const NUM_SEED_DEPOSITS_PER_ACCOUNT: usize = 50;
    const NUM_TRANSFERS: usize = 120;

    let srs = filecoin_srs_agg(K).unwrap();
    let relation = Spend2Output2;
    let vk = compact_std_lib::setup_vk(&srs, &relation);
    let pk = compact_std_lib::setup_pk(&relation, &vk);

    // ✅ Cache AGG keys once (for the fixed batch size).
    let agg_setup = prepare_agg_setup(&srs, vk.vk(), LEAF_VK_NAME, K, BATCH_SIZE);

    // ✅ Cache FINAL aggregation vk/pk once (depends only on cached agg_setup for this batch size).
    let final_agg_srs = proof_agg::filecoin_srs_agg(proof_agg::AGG_K).unwrap();
    let default_final_circuit = FinalAggCircuit {
        child_vk: agg_setup.child_vk().clone(),
        child_vk_name: agg_setup.child_vk_name().to_string(),
        child_level: F::from(agg_setup.child_level() as u64),
        left_proof: Value::unknown(),
        right_proof: Value::unknown(),
        left_pi_acc: Value::unknown(),
        right_pi_acc: Value::unknown(),
        fixed_base_names: agg_setup.fixed_base_names().to_vec(),
        left_child_state: Value::unknown(),
        right_child_state: Value::unknown(),
        agg_state: Value::unknown(),
        pre_commitment_roots_map: Value::unknown(),
        post_commitment_roots_root: Value::unknown(),
    };
    let final_vk = keygen_vk_with_k(&final_agg_srs, &default_final_circuit, proof_agg::AGG_K)
        .expect("final vk gen should not fail");
    let final_pk =
        keygen_pk(final_vk.clone(), &default_final_circuit).expect("final pk gen should not fail");

    let mut rng = ChaCha8Rng::from_entropy();
    let asset_id = F::random(&mut rng);

    let mut commitment_map = MapMt::<F, PoseidonChip<F>>::new(&F::ZERO);
    let mut nullifier_map = MapMt::<F, PoseidonChip<F>>::new(&F::ZERO);

    // Confirmed root history (one per “rollup commit”)
    let mut commitment_root_history: Vec<F> = Vec::new();
    // Confirmed map snapshots (for generating lagging proofs)
    let mut commitment_map_history: Vec<MapMt<F, PoseidonChip<F>>> = Vec::new();

    let mut commitment_roots_set = MapMt::<F, PoseidonChip<F>>::new(&F::ZERO);

    let mut accounts: Vec<Account> = (0..NUM_ACCOUNTS)
        .map(|i| {
            let sk = JubjubScalar::random(&mut OsRng);
            let pk_point = JubjubSubgroup::generator() * sk;
            let fields = AssignedNativePoint::<Jubjub>::as_public_input(&pk_point);
            Account {
                id: i,
                sk,
                pk_point,
                pk_x: fields[0],
                pk_y: fields[1],
                wallet: vec![],
            }
        })
        .collect();

    for acc in &mut accounts {
        for _ in 0..NUM_SEED_DEPOSITS_PER_ACCOUNT {
            let hi: u128 = rng.r#gen::<u128>() >> (128 - AMOUNT_GEN_BITS);
            let amt: u128 = hi;
            let utxo = Utxo {
                asset_id,
                amount: amt,
                randomness: F::random(&mut rng),
            };
            let commit = host_commit(
                utxo.asset_id,
                utxo.amount,
                acc.pk_x,
                acc.pk_y,
                utxo.randomness,
            );

            commitment_map.insert(&commit, &F::ONE);

            acc.wallet.push(Note {
                utxo,
                commit,
                spent: false,
                confirmed_at_root_idx: 0,
            });
        }
    }

    let genesis_root = commitment_map.succinct_repr();
    println!("Initial commitment root: {:?}", genesis_root);

    commitment_root_history.push(genesis_root);
    commitment_map_history.push(commitment_map.clone());
    commitment_roots_set.insert(&genesis_root, &F::ONE);

    let choose_sender = |rng: &mut ChaCha8Rng,
                         accs: &mut [Account],
                         latest_confirmed_root_idx: usize|
     -> Option<usize> {
        let viable: Vec<usize> = accs
            .iter()
            .enumerate()
            .filter(|(_, a)| {
                a.wallet
                    .iter()
                    .filter(|n| !n.spent && n.confirmed_at_root_idx <= latest_confirmed_root_idx)
                    .count()
                    >= 2
            })
            .map(|(i, _)| i)
            .collect();
        if viable.is_empty() {
            None
        } else {
            Some(viable[rng.gen_range(0..viable.len())])
        }
    };

    let mut total_transfers_done = 0usize;
    let mut batch_idx = 0usize;

    'outer: loop {
        if total_transfers_done >= NUM_TRANSFERS {
            break;
        }

        let mut shadow_accounts = accounts.clone();
        let mut shadow_nullifier_map = nullifier_map.clone();
        let mut shadow_commitment_map = commitment_map.clone();

        let pre_nullifier_map_for_batch = shadow_nullifier_map.clone();
        let pre_commitment_map_for_batch = shadow_commitment_map.clone();
        let pre_commitment_roots_map_for_batch = commitment_roots_set.clone();

        let latest_confirmed_root_idx = commitment_root_history.len() - 1;

        let mut client_proofs: Vec<AggClientProof> = Vec::new();

        println!(
            "\n=== Starting batch {} from commitment root {:?} ===",
            batch_idx,
            shadow_commitment_map.succinct_repr()
        );

        let mut batch_failed = false;

        for _ in 0..BATCH_SIZE {
            if total_transfers_done >= NUM_TRANSFERS {
                break;
            }

            let sender_idx = match choose_sender(
                &mut rng,
                &mut shadow_accounts,
                latest_confirmed_root_idx,
            ) {
                Some(i) => i,
                None => {
                    println!(
                        "[batch {}] no account has two spendable confirmed notes; stopping batching.",
                        batch_idx
                    );
                    batch_failed = true;
                    break;
                }
            };

            let (i_old1, i_old2) = {
                let unspent: Vec<usize> = shadow_accounts[sender_idx]
                    .wallet
                    .iter()
                    .enumerate()
                    .filter(|(_, n)| {
                        !n.spent && n.confirmed_at_root_idx <= latest_confirmed_root_idx
                    })
                    .map(|(i, _)| i)
                    .collect();
                let a = unspent[rng.gen_range(0..unspent.len())];
                let mut b = unspent[rng.gen_range(0..unspent.len())];
                while b == a {
                    b = unspent[rng.gen_range(0..unspent.len())];
                }
                (a, b)
            };

            let r1 = rng.gen_range(0..NUM_ACCOUNTS);
            let r2 = rng.gen_range(0..NUM_ACCOUNTS);

            let sender = shadow_accounts[sender_idx].clone();
            let old1 = shadow_accounts[sender_idx].wallet[i_old1].clone();
            let old2 = shadow_accounts[sender_idx].wallet[i_old2].clone();

            // NEW: choose a (possibly lagging) confirmed root index to prove against
            let min_root_idx_for_inputs =
                old1.confirmed_at_root_idx.max(old2.confirmed_at_root_idx);

            let root_idx_for_proof = if min_root_idx_for_inputs < latest_confirmed_root_idx
                && rng.gen_bool(LAG_TX_PROB)
            {
                // force lag: pick strictly older than latest when possible
                rng.gen_range(min_root_idx_for_inputs..=latest_confirmed_root_idx - 1)
            } else {
                latest_confirmed_root_idx
            };

            let historic_commit_map = commitment_map_history[root_idx_for_proof].clone();
            let root_before = commitment_root_history[root_idx_for_proof];
            debug_assert_eq!(historic_commit_map.succinct_repr(), root_before);

            if root_idx_for_proof != latest_confirmed_root_idx {
                println!(
                    "[batch {}, tx {}] 🕒 lagging proof root: idx {} (latest {}), root {:?}",
                    batch_idx,
                    total_transfers_done,
                    root_idx_for_proof,
                    latest_confirmed_root_idx,
                    root_before
                );
            }

            let total: u128 = old1.utxo.amount + old2.utxo.amount;
            let out1_amt: u128 = if total == 0 {
                0
            } else {
                rng.gen_range(0..=total)
            };
            let out2_amt: u128 = total - out1_amt;

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

            let new1_commit = host_commit(
                new1.asset_id,
                new1.amount,
                shadow_accounts[r1].pk_x,
                shadow_accounts[r1].pk_y,
                new1.randomness,
            );
            let new2_commit = host_commit(
                new2.asset_id,
                new2.amount,
                shadow_accounts[r2].pk_x,
                shadow_accounts[r2].pk_y,
                new2.randomness,
            );

            let nf1 = host_nullify(old1.commit, sender.pk_x, sender.pk_y);
            let nf2 = host_nullify(old2.commit, sender.pk_x, sender.pk_y);

            shadow_nullifier_map.insert(&nf1, &F::ONE);
            shadow_nullifier_map.insert(&nf2, &F::ONE);

            let alpha = JubjubScalar::random(&mut OsRng);
            let blind_point = JubjubSubgroup::generator() * alpha;
            let pk_blinded_point = sender.pk_point + blind_point;
            let pkb_fields = AssignedNativePoint::<Jubjub>::as_public_input(&pk_blinded_point);
            let pk_bx = pkb_fields[0];
            let pk_by = pkb_fields[1];

            let public_items = [
                root_before,
                pk_bx,
                pk_by,
                new1_commit,
                new2_commit,
                nf1,
                nf2,
            ];
            let instance: F = host_instance_hash(public_items);

            let witness = (
                historic_commit_map,
                sender.sk,
                F::from_bytes_le(&alpha.to_bytes()).unwrap(),
                old1.utxo.clone(),
                old2.utxo.clone(),
                new1.clone(),
                new2.clone(),
                shadow_accounts[r1].pk_point,
                shadow_accounts[r2].pk_point,
            );

            let now = Instant::now();
            let proof = compact_std_lib::prove::<Spend2Output2, PoseidonState<F>>(
                &srs, &pk, &relation, &instance, witness, OsRng,
            )
            .expect("Proof generation failed");
            println!(
                "[batch {}, tx {}] proof gen: {:?}",
                batch_idx,
                total_transfers_done,
                now.elapsed()
            );

            let stats = cost_model(&Spend2Output2);
            println!("client circuit stats: {:?}", stats);

            client_proofs.push(AggClientProof {
                state: instance,
                proof: proof.clone(),
                public_items,
            });

            // Apply to shadow rollup state (current commitment/nullifier sets)
            shadow_commitment_map.insert(&new1_commit, &F::ONE);
            shadow_commitment_map.insert(&new2_commit, &F::ONE);

            shadow_accounts[sender_idx].wallet[i_old1].spent = true;
            shadow_accounts[sender_idx].wallet[i_old2].spent = true;

            let confirm_at_idx = commitment_root_history.len();
            shadow_accounts[r1].wallet.push(Note {
                utxo: new1,
                commit: new1_commit,
                spent: false,
                confirmed_at_root_idx: confirm_at_idx,
            });
            shadow_accounts[r2].wallet.push(Note {
                utxo: new2,
                commit: new2_commit,
                spent: false,
                confirmed_at_root_idx: confirm_at_idx,
            });

            println!(
                "[batch {}, tx {}] shadow commitment root updated: {:?}",
                batch_idx,
                total_transfers_done,
                shadow_commitment_map.succinct_repr()
            );

            total_transfers_done += 1;
        }

        if batch_failed || client_proofs.is_empty() {
            break 'outer;
        }

        assert_eq!(
            client_proofs.len(),
            BATCH_SIZE,
            "Batch not completely filled"
        );
        assert!(
            client_proofs.len().is_power_of_two(),
            "Batch size must be a power of two"
        );

        let now = Instant::now();
        let agg_result = aggregate_client_proofs_cached(
            &agg_setup,
            &srs,
            vk.vk(),
            &client_proofs,
            pre_commitment_map_for_batch.clone(),
            pre_nullifier_map_for_batch.clone(),
            pre_commitment_roots_map_for_batch.clone(),
        );
        println!(
            "Batch {} aggregated (up to top pair) in {:?}",
            batch_idx,
            now.elapsed()
        );
        println!(
            "Batch {} computed subroot (host): {:?}",
            batch_idx, agg_result.root_state.subroot
        );

        let pre_roots_set_root = pre_commitment_roots_map_for_batch.succinct_repr();

        assert_eq!(
            pre_commitment_roots_map_for_batch.get(&agg_result.root_state.c_post),
            F::ZERO,
            "Replay guard: c_post already present in roots set"
        );
        let mut shadow_commitment_roots_set = pre_commitment_roots_map_for_batch.clone();
        shadow_commitment_roots_set.insert(&agg_result.root_state.c_post, &F::ONE);
        let post_roots_set_root = shadow_commitment_roots_set.succinct_repr();

        {
            use midnight_proofs::poly::kzg::KZGCommitmentScheme;
            use midnight_proofs::transcript::CircuitTranscript;

            let mut final_acc: AggAccumulator = AggAccumulator::accumulate(&[
                agg_result.left_top.proof_acc.clone(),
                agg_result.left_top.pi_acc.clone(),
                agg_result.right_top.proof_acc.clone(),
                agg_result.right_top.pi_acc.clone(),
            ]);
            final_acc.collapse();
            let final_acc_pi = accumulator_as_public_input(&final_acc);

            let final_circuit = FinalAggCircuit {
                child_vk: agg_result.child_vk.clone(),
                child_vk_name: agg_result.child_vk_name.clone(),
                child_level: F::from(agg_result.child_level as u64),
                left_proof: Value::known(agg_result.left_top.proof.clone()),
                right_proof: Value::known(agg_result.right_top.proof.clone()),
                left_pi_acc: Value::known(agg_result.left_top.pi_acc.clone()),
                right_pi_acc: Value::known(agg_result.right_top.pi_acc.clone()),
                fixed_base_names: agg_result.fixed_base_names.clone(),
                left_child_state: Value::known(agg_result.left_top.state),
                right_child_state: Value::known(agg_result.right_top.state),
                agg_state: Value::known(agg_result.root_state),
                pre_commitment_roots_map: Value::known(pre_commitment_roots_map_for_batch.clone()),
                post_commitment_roots_root: Value::known(post_roots_set_root),
            };

            let mut final_public_inputs: Vec<F> = vec![
                agg_result.root_state.c_pre,
                agg_result.root_state.c_post,
                agg_result.root_state.n_pre,
                agg_result.root_state.n_post,
                agg_result.root_state.subroot,
                pre_roots_set_root,
                post_roots_set_root,
            ];
            final_public_inputs.extend(final_acc_pi.clone());

            // ✅ Use cached final_pk/final_vk/final_agg_srs (no per-batch keygen).
            let final_proof_bytes = {
                let mut transcript = CircuitTranscript::<KeccakTranscript>::init();
                create_proof::<
                    F,
                    KZGCommitmentScheme<midnight_curves::Bls12>,
                    CircuitTranscript<KeccakTranscript>,
                    FinalAggCircuit,
                >(
                    &final_agg_srs,
                    &final_pk,
                    &[final_circuit],
                    1,
                    &[&[&[], &final_public_inputs]],
                    OsRng,
                    &mut transcript,
                )
                .expect("Final aggregation proof generation should not fail");
                transcript.finalize()
            };

            println!("final proof size (bytes): {}", final_proof_bytes.len());

            let mut transcript =
                CircuitTranscript::<KeccakTranscript>::init_from_bytes(&final_proof_bytes);
            let committed_bases: &[&[midnight_curves::G1Projective]] =
                &[&[midnight_curves::G1Projective::identity()]];
            let instances: &[&[&[F]]] = &[&[&final_public_inputs]];

            let dual_msm = prepare::<
                F,
                KZGCommitmentScheme<midnight_curves::Bls12>,
                CircuitTranscript<KeccakTranscript>,
            >(&final_vk, committed_bases, instances, &mut transcript)
            .expect("Final aggregation verification preparation failed");

            assert!(
                dual_msm.check(&final_agg_srs.verifier_params()),
                "Final proof must verify"
            );

            assert!(
                final_acc.check(&final_agg_srs.s_g2().into(), &agg_result.fixed_bases),
                "Final aggregation accumulator must verify"
            );

            println!(
                "\n✅ Final aggregation (MERGED) proof for batch {} verified.\n\
                 Batch subroot (host): {:?}\n\
                    Commitment-set transition: {:?} -> {:?}\n\
                    Nullifier-set transition: {:?} -> {:?}\n\
                    Historic-roots-set transition: {:?} -> {:?}\n\
                    Final accumulator PI length: {} field elements",
                batch_idx,
                agg_result.root_state.subroot,
                agg_result.root_state.c_pre,
                agg_result.root_state.c_post,
                agg_result.root_state.n_pre,
                agg_result.root_state.n_post,
                pre_roots_set_root,
                post_roots_set_root,
                final_acc_pi.len()
            );
        }

        // Commit batch to “chain state”
        accounts = shadow_accounts;
        nullifier_map = shadow_nullifier_map;
        commitment_map = shadow_commitment_map;

        commitment_roots_set = shadow_commitment_roots_set;
        commitment_root_history.push(commitment_map.succinct_repr());
        commitment_map_history.push(commitment_map.clone());

        println!(
            "After batch {} committed commitment root: {:?}",
            batch_idx,
            commitment_map.succinct_repr()
        );

        // NEW: Demonstrate replay protection:
        // Attempt to apply the SAME client proofs again on the updated state.
        // This should fail because nullifiers are already present.
        println!("REPLAY attempt:");
        println!("  c_pre  = {:?}", agg_result.root_state.c_pre);
        println!("  c_post = {:?}", agg_result.root_state.c_post);
        println!("  n_pre  = {:?}", agg_result.root_state.n_pre);
        println!("  n_post = {:?}", agg_result.root_state.n_post);

        println!(
            "  roots_set has c_pre?  {:?}",
            commitment_roots_set.get(&agg_result.root_state.c_pre)
        );
        println!(
            "  roots_set has c_post? {:?}",
            commitment_roots_set.get(&agg_result.root_state.c_post)
        );

        println!(
            "  nullifier_map root == n_post? {}",
            nullifier_map.succinct_repr() == agg_result.root_state.n_post
        );
        println!(
            "REPLAY using commitment_map root: {:?}",
            commitment_map.succinct_repr()
        );
        println!(
            "REPLAY using nullifier_map  root: {:?}",
            nullifier_map.succinct_repr()
        );

        let replay_commit_map = commitment_map.clone(); // current POST state
        let replay_null_map = nullifier_map.clone(); // current POST state
        let replay_roots_set_map = commitment_roots_set.clone(); // head AFTER applying batch

        println!(
            "REPLAY using commitment_map root: {:?}",
            replay_commit_map.succinct_repr()
        );
        println!(
            "REPLAY using nullifier_map  root: {:?}",
            replay_null_map.succinct_repr()
        );

        let replay = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = aggregate_client_proofs_cached(
                &agg_setup,
                &srs,
                vk.vk(),
                &client_proofs,
                replay_commit_map.clone(),
                replay_null_map.clone(),
                replay_roots_set_map.clone(),
            );
        }));
        match replay {
            Ok(_) => println!("❌ Replay unexpectedly succeeded (BUG)"),
            Err(_) => println!(
                "✅ Replay correctly rejected (nullifiers already spent / state already advanced)"
            ),
        }

        batch_idx += 1;
    }

    println!(
        "\nFinal commitment root: {:?}",
        commitment_map.succinct_repr()
    );

    for acc in &accounts {
        let bal: u128 = acc
            .wallet
            .iter()
            .filter(|n| !n.spent)
            .fold(0u128, |s, n| s.saturating_add(n.utxo.amount));
        println!(
            "Account {} unspent notes: {}, balance {}",
            acc.id,
            acc.wallet.iter().filter(|n| !n.spent).count(),
            bal
        );
    }
}
