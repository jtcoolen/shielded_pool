use std::time::Instant;

use ff::{Field, PrimeField};
use group::Group;

use midnight_circuits::{
    biguint::AssignedBigUint,
    compact_std_lib::{self, Relation, ZkStdLib, ZkStdLibArch},
    ecc::native::AssignedScalarOfNativeCurve,
    hash::poseidon::{PoseidonChip, PoseidonState, constants::PoseidonField},
    instructions::{
        AssertionInstructions, AssignmentInstructions, ControlFlowInstructions,
        ConversionInstructions, DecompositionInstructions, EccInstructions,
        PublicInputInstructions, hash::HashCPU, map::MapInstructions,
    },
    map::cpu::MapMt,
    types::{AssignedBit, AssignedNative, AssignedNativePoint, Instantiable},
};

use midnight_curves::{Fq as F, Fr as JubjubScalar, JubjubExtended as Jubjub, JubjubSubgroup};
use midnight_proofs::{
    circuit::{Layouter, Value},
    plonk::Error,
    transcript::Transcript,
};
use rand::{Rng, SeedableRng, rngs::OsRng};
use rand_chacha::ChaCha8Rng;

// -----------------------------------------------------------------------------
// Proof aggregator module (SWAPPED to use the recursive aggregation approach
// from <AGG>: explicit per-level VKs, enforced prev_level, and raw PIs
// [state, acc_pi..., level] (no VK-PI prefix selection).
// -----------------------------------------------------------------------------
mod proof_agg {
    use halo2curves::{ff::Field, group::Group};
    use midnight_circuits::hash::poseidon::PoseidonState;
    use midnight_circuits::instructions::map::MapInstructions;
    use midnight_circuits::types::{AssignedForeignPoint, Instantiable};
    use midnight_circuits::{
        ecc::{
            curves::CircuitCurve,
            foreign::{ForeignEccChip, ForeignEccConfig, nb_foreign_ecc_chip_columns},
        },
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
            PublicInputInstructions,
        },
        types::{AssignedNative, ComposableChip},
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
    use rand::rngs::OsRng;
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
    type CBase = <C as CircuitCurve>::Base;
    type NG = NativeGadget<F, P2RDecompositionChip<F>, NativeChip<F>>;
    type Map = midnight_circuits::map::cpu::MapMt<F, PoseidonChip<F>>;

    /// Re-exported accumulator type for host-side code.
    pub type AggAccumulator = Accumulator<S>;

    /// Convert an accumulator to the exact "acc_pi..." field list used as public inputs.
    pub fn accumulator_as_public_input(acc: &AggAccumulator) -> Vec<F> {
        AssignedAccumulator::as_public_input(acc)
    }

    // Leaf aggregation circuit size and internal aggregation circuit size.
    // Mirrors the <AGG> pattern (leaf smaller than internal).
    const K_LEAF: u32 = 19;
    const K_INTERNAL: u32 = 20;

    pub const AGG_K: u32 = K_INTERNAL;

    pub const FINAL_NUM_LEAVES: usize = 8;
    pub const NULLIFIERS_PER_LEAF: usize = 2;
    pub const FINAL_NUM_NULLIFIERS: usize = FINAL_NUM_LEAVES * NULLIFIERS_PER_LEAF;

    type Vk = VerifyingKey<F, KZGCommitmentScheme<E>>;
    type Pk = ProvingKey<F, KZGCommitmentScheme<E>>;

    type LeafAggCircuit = AggCircuit<K_LEAF>;
    type InternalAggCircuit = AggCircuit<K_INTERNAL>;

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
                "Failed to open SRS file at '{}': {e}. \
                 (Did you download/parse the Filecoin SRS and set SRS_DIR?)",
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

        // If we loaded 2^20, downsize and cache at 2^k.
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

        (
            native_config,
            core_decomp_config,
            curve_config,
            poseidon_config,
        )
    }

    /// Typed container for AggCircuit public inputs:
    /// layout = [state, acc_pi..., level]
    #[derive(Clone, Debug)]
    pub struct AggPublicInputs {
        pub state: F,
        pub pi_acc: AggAccumulator,
        pub level: F,
    }

    impl AggPublicInputs {
        pub fn to_fields(&self) -> Vec<F> {
            let mut out = Vec::new();
            out.push(self.state);
            out.extend(AssignedAccumulator::as_public_input(&self.pi_acc));
            out.push(self.level);
            out
        }
    }

    #[derive(Clone, Debug)]
    pub struct AggCircuit<const K: u32> {
        // VK *of the circuit verified at this layer* (children).
        child_vk: VkData,
        child_vk_name: String,

        // Enforced prev_level value (0 for leaf agg, >=1 for internal levels).
        expected_prev_level: F,

        left_state: Value<F>,
        right_state: Value<F>,
        left_proof: Value<Vec<u8>>,
        right_proof: Value<Vec<u8>>,
        left_acc: Value<Accumulator<S>>,
        right_acc: Value<Accumulator<S>>,
        fixed_base_names: Vec<String>,
        prev_level: Value<F>,
        is_leaf: bool,
    }

    impl<const K: u32> Circuit<F> for AggCircuit<K> {
        type Config = (
            NativeConfig,
            P2RDecompositionConfig,
            ForeignEccConfig<C>,
            PoseidonConfig<F>,
        );
        type FloorPlanner = SimpleFloorPlanner;
        type Params = ();

        fn without_witnesses(&self) -> Self {
            Self {
                child_vk: self.child_vk.clone(),
                child_vk_name: self.child_vk_name.clone(),
                expected_prev_level: self.expected_prev_level,
                left_state: Value::unknown(),
                right_state: Value::unknown(),
                left_proof: Value::unknown(),
                right_proof: Value::unknown(),
                left_acc: Value::unknown(),
                right_acc: Value::unknown(),
                fixed_base_names: self.fixed_base_names.clone(),
                prev_level: Value::unknown(),
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
            let scalar_chip = NativeGadget::new(core_decomp_chip.clone(), native_chip.clone());
            let curve_chip = ForeignEccChip::new(&config.2, &scalar_chip, &scalar_chip);
            let poseidon_chip = PoseidonChip::new(&config.3, &native_chip);
            let verifier_chip = VerifierGadget::new(&curve_chip, &scalar_chip, &poseidon_chip);

            // Enforce expected level and compute next level
            let prev_level = scalar_chip.assign(&mut layouter, self.prev_level)?;
            scalar_chip.assert_equal_to_fixed(
                &mut layouter,
                &prev_level,
                self.expected_prev_level,
            )?;
            let next_level = scalar_chip.add_constant(&mut layouter, &prev_level, F::ONE)?;

            // VK transcript representation is a fixed constant in-circuit for this layer
            let child_vk_val: AssignedNative<F> =
                native_chip.assign_fixed(&mut layouter, self.child_vk.transcript_repr)?;

            // Compute next state (Poseidon over children states)
            let left_state: AssignedNative<F> =
                scalar_chip.assign(&mut layouter, self.left_state)?;
            let right_state: AssignedNative<F> =
                scalar_chip.assign(&mut layouter, self.right_state)?;
            let next_state =
                poseidon_chip.hash(&mut layouter, &[left_state.clone(), right_state.clone()])?;

            // Assigned accumulators for children (PI accumulators provided as witnesses)
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

            // VK used by the partial verifier at this layer
            let assigned_vk = verifier_chip.assign_vk(
                self.child_vk_name.as_str(),
                &self.child_vk.domain,
                &self.child_vk.cs,
                child_vk_val,
            )?;

            // Child public inputs expected by the verifier gadget.
            // - leaf agg: leaf circuit has PI = [state]
            // - internal agg: child agg circuits have PI = [state, acc_pi..., level]
            let (assigned_left_pi, assigned_right_pi) = if self.is_leaf {
                // Scale PI accumulators to neutral (bit=0) to keep bases available but not contribute.
                let neutral_scaling_factor = scalar_chip.assign_fixed(&mut layouter, false)?;
                AssignedAccumulator::scale_by_bit(
                    &mut layouter,
                    &scalar_chip,
                    &neutral_scaling_factor,
                    &mut left_acc,
                )?;
                AssignedAccumulator::scale_by_bit(
                    &mut layouter,
                    &scalar_chip,
                    &neutral_scaling_factor,
                    &mut right_acc,
                )?;
                left_acc.collapse(&mut layouter, &curve_chip, &scalar_chip)?;
                right_acc.collapse(&mut layouter, &curve_chip, &scalar_chip)?;

                (vec![left_state.clone()], vec![right_state.clone()])
            } else {
                let mut left_pi = vec![left_state.clone()];
                left_pi.extend(verifier_chip.as_public_input(&mut layouter, &left_acc)?);
                left_pi.push(prev_level.clone());

                let mut right_pi = vec![right_state.clone()];
                right_pi.extend(verifier_chip.as_public_input(&mut layouter, &right_acc)?);
                right_pi.push(prev_level.clone());

                (left_pi, right_pi)
            };

            let id_point: AssignedForeignPoint<
                midnight_curves::Fq,
                midnight_curves::G1Projective,
                midnight_curves::G1Projective,
            > = curve_chip.assign_fixed(&mut layouter, C::identity())?;

            // Process left child
            let mut left_proof_acc = verifier_chip.prepare(
                &mut layouter,
                &assigned_vk,
                &[("com_instance", id_point.clone())],
                &[&assigned_left_pi],
                self.left_proof.clone(),
            )?;
            left_proof_acc.collapse(&mut layouter, &curve_chip, &scalar_chip)?;

            // Process right child
            let mut right_proof_acc = verifier_chip.prepare(
                &mut layouter,
                &assigned_vk,
                &[("com_instance", id_point)],
                &[&assigned_right_pi],
                self.right_proof.clone(),
            )?;
            right_proof_acc.collapse(&mut layouter, &curve_chip, &scalar_chip)?;

            // Accumulate and output
            let mut next_acc = AssignedAccumulator::<S>::accumulate(
                &mut layouter,
                &verifier_chip,
                &scalar_chip,
                &poseidon_chip,
                &[left_proof_acc, left_acc, right_proof_acc, right_acc],
            )?;
            next_acc.collapse(&mut layouter, &curve_chip, &scalar_chip)?;

            // Expose raw public inputs (no hashing): [state, acc_pi..., level]
            let next_acc_pi = verifier_chip.as_public_input(&mut layouter, &next_acc)?;
            native_chip.constrain_as_public_input(&mut layouter, &next_state)?;
            for x in next_acc_pi.iter() {
                native_chip.constrain_as_public_input(&mut layouter, x)?;
            }
            native_chip.constrain_as_public_input(&mut layouter, &next_level)?;

            core_decomp_chip.load(&mut layouter)
        }
    }

    #[derive(Clone, Debug)]
    struct TreeNode {
        state: F,
        proof: Vec<u8>,
        proof_acc: AggAccumulator,
        pi_acc: AggAccumulator,
        public_inputs: AggPublicInputs,
    }

    #[derive(Clone, Debug)]
    pub struct ClientProof {
        /// Public instance = single field element for the client's witness
        pub state: F,
        /// The client's proof (created off-chain by the client)
        pub proof: Vec<u8>,
        /// The 7 would-be public inputs that were Poseidon-hashed to `state`
        /// [root, pk'_x, pk'_y, new_c1, new_c2, nf1, nf2]
        pub public_items: [F; 7],
    }

    #[derive(Clone, Debug)]
    pub struct AggregationResult {
        pub root_state: F,
        pub agg_proof: Vec<u8>,
        pub agg_public_inputs: AggPublicInputs,
        pub agg_proof_acc: AggAccumulator,
        pub agg_vk: VerifyingKey<F, KZGCommitmentScheme<E>>,
        pub agg_vk_name: String,
        pub fixed_base_names: Vec<String>,
        pub fixed_bases: BTreeMap<String, C>,
        pub leaf_states: Vec<F>,
        pub client_pis: Vec<[F; 7]>,
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
            "Number of leaves must be a power of two"
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
                    "Duplicate vk_name in AggKeyStore: '{}'",
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
                "Requested agg level {} out of range (valid: 1..={})",
                level,
                self.levels.len()
            );
            &self.levels[level - 1]
        }
    }

    fn keygen_vk_pk<Circ: Circuit<F>>(srs: &ParamsKZG<Bls12>, circuit: &Circ, k: u32) -> (Vk, Pk) {
        let vk = keygen_vk_with_k(srs, circuit, k).expect("keygen_vk_with_k failed");
        let pk = keygen_pk(vk.clone(), circuit).expect("keygen_pk failed");
        (vk, pk)
    }

    /// Aggregates a list of client proofs into a single AGG proof.
    pub fn aggregate_client_proofs(
        leaf_srs: &ParamsKZG<Bls12>,
        leaf_vk: &VerifyingKey<F, KZGCommitmentScheme<E>>,
        leaf_vk_name: &'static str,
        leaf_k: u32,
        client_proofs: &[ClientProof],
    ) -> AggregationResult {
        assert!(!client_proofs.is_empty(), "Need at least one client proof");
        assert!(
            client_proofs.len().is_power_of_two(),
            "Number of client proofs must be a power of two"
        );

        let num_leaves = client_proofs.len();
        let max_level: usize = (num_leaves as u32).trailing_zeros() as usize;
        assert!(max_level > 0, "max_level computed as 0");

        // Leaf vk data (for any 1-instance circuit)
        let leaf_vk_data = VkData {
            domain: EvaluationDomain::new(leaf_vk.cs().degree() as u32, leaf_k),
            cs: leaf_vk.cs().clone(),
            transcript_repr: leaf_vk.transcript_repr(),
        };

        // Setup aggregation CS and SRS
        let mut agg_cs = ConstraintSystem::default();
        configure_agg_circuit(&mut agg_cs);

        let agg_srs1 = filecoin_srs_agg(K_LEAF).unwrap();
        let agg_srs2 = filecoin_srs_agg(K_INTERNAL).unwrap();

        // Ensure same G2 generator (common in Filecoin SRS downsizing)
        assert_eq!(
            leaf_srs.s_g2(),
            agg_srs2.s_g2(),
            "leaf_srs vs agg_srs2 s_g2 mismatch"
        );
        assert_eq!(
            agg_srs1.s_g2(),
            agg_srs2.s_g2(),
            "agg_srs1 vs agg_srs2 s_g2 mismatch"
        );

        // Precompute all AGG vk names
        let agg_vk_names: Vec<String> = (1..=max_level).map(agg_vk_name_for_level).collect();

        // Build a global fixed base name list (Leaf + every AGG vk_name)
        let combined_fixed_base_names: Vec<String> = {
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

        // Keygen one AGG VK/PK per level
        let mut agg_levels: Vec<AggLevelKeys> = Vec::with_capacity(max_level);

        for level in 1..=max_level {
            let (child_vk, child_vk_name, expected_prev_level, is_leaf) = if level == 1 {
                (
                    leaf_vk_data.clone(),
                    leaf_vk_name.to_string(),
                    F::ZERO,
                    true,
                )
            } else {
                let child_level = level - 1;
                let child = agg_levels
                    .get(child_level - 1)
                    .expect("Missing child level during keygen")
                    .vk_data
                    .clone();
                let child_name = agg_vk_names[child_level - 1].clone();
                (child, child_name, F::from(child_level as u64), false)
            };

            let name = agg_vk_names[level - 1].clone();
            let start = Instant::now();

            if level == 1 {
                let default_circuit = LeafAggCircuit {
                    child_vk,
                    child_vk_name,
                    expected_prev_level,
                    left_state: Value::unknown(),
                    right_state: Value::unknown(),
                    left_proof: Value::unknown(),
                    right_proof: Value::unknown(),
                    left_acc: Value::unknown(),
                    right_acc: Value::unknown(),
                    fixed_base_names: combined_fixed_base_names.clone(),
                    prev_level: Value::unknown(),
                    is_leaf,
                };
                let (vk, pk) = keygen_vk_pk(&agg_srs1, &default_circuit, K_LEAF);
                println!("Computed {} vk/pk in {:?}", name, start.elapsed());
                agg_levels.push(AggLevelKeys::new(level, name, vk, pk));
            } else {
                let default_circuit = InternalAggCircuit {
                    child_vk,
                    child_vk_name,
                    expected_prev_level,
                    left_state: Value::unknown(),
                    right_state: Value::unknown(),
                    left_proof: Value::unknown(),
                    right_proof: Value::unknown(),
                    left_acc: Value::unknown(),
                    right_acc: Value::unknown(),
                    fixed_base_names: combined_fixed_base_names.clone(),
                    prev_level: Value::unknown(),
                    is_leaf,
                };
                let (vk, pk) = keygen_vk_pk(&agg_srs2, &default_circuit, K_INTERNAL);
                println!("Computed {} vk/pk in {:?}", name, start.elapsed());
                agg_levels.push(AggLevelKeys::new(level, name, vk, pk));
            }
        }

        let agg_store = AggKeyStore::new(agg_levels);

        // Build combined fixed bases map
        let mut leaf_fixed_bases = BTreeMap::new();
        leaf_fixed_bases.insert(String::from("com_instance"), C::identity());
        leaf_fixed_bases.extend(midnight_circuits::verifier::fixed_bases::<S>(
            leaf_vk_name,
            leaf_vk,
        ));

        let mut combined_fixed_bases = BTreeMap::new();
        combined_fixed_bases.extend(leaf_fixed_bases.iter().map(|(k, v)| (k.clone(), *v)));
        for level in 1..=agg_store.max_level() {
            combined_fixed_bases.extend(
                agg_store
                    .get(level)
                    .fixed_bases
                    .iter()
                    .map(|(k, v)| (k.clone(), *v)),
            );
        }

        // Build a global trivial accumulator carrying fixed bases for all circuits
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

        // Create leaf aggregation layer (AGG level 1, verifies client proofs)
        println!("\nCreating {} leaf AGG nodes...", num_leaves / 2);

        let leaf_level = 1usize;
        let leaf_keys = agg_store.get(leaf_level);
        let leaf_agg_vk_name = leaf_keys.name.clone();

        let mut current_level: Vec<TreeNode> = (0..num_leaves / 2)
            .map(|i| {
                use midnight_circuits::instructions::hash::HashCPU;

                let left = &client_proofs[i * 2];
                let right = &client_proofs[i * 2 + 1];

                let state = <PoseidonChip<F> as HashCPU<F, F>>::hash(&[left.state, right.state]);

                let circuit = LeafAggCircuit {
                    child_vk: leaf_vk_data.clone(),
                    child_vk_name: leaf_vk_name.to_string(),
                    expected_prev_level: F::ZERO,
                    left_state: Value::known(left.state),
                    right_state: Value::known(right.state),
                    left_proof: Value::known(left.proof.clone()),
                    right_proof: Value::known(right.proof.clone()),
                    left_acc: Value::known(trivial_combined.clone()),
                    right_acc: Value::known(trivial_combined.clone()),
                    fixed_base_names: combined_fixed_base_names.clone(),
                    prev_level: Value::known(F::ZERO),
                    is_leaf: true,
                };

                // Verify client proofs and extract accumulators
                let proof_acc_left = verify_and_extract_acc(
                    leaf_srs,
                    leaf_vk,
                    &leaf_fixed_bases,
                    &left.proof,
                    &[left.state],
                );

                let proof_acc_right = verify_and_extract_acc(
                    leaf_srs,
                    leaf_vk,
                    &leaf_fixed_bases,
                    &right.proof,
                    &[right.state],
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
                    level: F::ONE, // leaf level outputs next_level = 1
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
                        &agg_srs1,
                        leaf_keys.pk.as_ref(),
                        &[circuit],
                        1,
                        &[&[&[], &public_inputs_fields]],
                        OsRng,
                        &mut transcript,
                    )
                    .expect("Leaf AGG proof failed");
                    transcript.finalize()
                };
                println!(
                    "Leaf AGG {} ({}) created in {:?}",
                    i,
                    leaf_agg_vk_name,
                    start.elapsed()
                );

                assert!(
                    accumulated_pi.check(&agg_srs2.s_g2().into(), &combined_fixed_bases),
                    "Leaf node {i}: accumulated PI accumulator did not check against combined fixed bases"
                );

                let proof_acc = verify_and_extract_acc(
                    &agg_srs1,
                    leaf_keys.vk.as_ref(),
                    &leaf_keys.fixed_bases,
                    &proof,
                    &public_inputs_fields,
                );

                TreeNode {
                    state,
                    proof,
                    proof_acc,
                    pi_acc: accumulated_pi,
                    public_inputs,
                }
            })
            .collect();

        // Build internal layers (each verifies previous aggregation layer)
        let mut child_level: usize = 1;
        while current_level.len() > 1 {
            let parent_level = child_level + 1;
            let parent_keys = agg_store.get(parent_level);
            let parent_vk_name = parent_keys.name.clone();

            println!(
                "\nBuilding AGG level {} ({}) with {} nodes...",
                parent_level,
                parent_vk_name,
                current_level.len() / 2
            );

            let child_keys = agg_store.get(child_level);
            let child_vk_data = child_keys.vk_data.clone();
            let child_vk_name = child_keys.name.clone();

            let next_level: Vec<TreeNode> = (0..current_level.len() / 2)
                .map(|i| {
                    use midnight_circuits::instructions::hash::HashCPU;

                    let left = &current_level[i * 2];
                    let right = &current_level[i * 2 + 1];

                    let state = <PoseidonChip<F> as HashCPU<F, F>>::hash(&[left.state, right.state]);

                    let circuit = InternalAggCircuit {
                        child_vk: child_vk_data.clone(),
                        child_vk_name: child_vk_name.clone(),
                        expected_prev_level: F::from(child_level as u64),
                        left_state: Value::known(left.state),
                        right_state: Value::known(right.state),
                        left_proof: Value::known(left.proof.clone()),
                        right_proof: Value::known(right.proof.clone()),
                        left_acc: Value::known(left.pi_acc.clone()),
                        right_acc: Value::known(right.pi_acc.clone()),
                        fixed_base_names: combined_fixed_base_names.clone(),
                        prev_level: Value::known(F::from(child_level as u64)),
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
                        level: F::from(parent_level as u64),
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
                            &agg_srs2,
                            parent_keys.pk.as_ref(),
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
                        "Level {parent_level} node {i}: accumulated PI accumulator did not check against combined fixed bases"
                    );

                    let proof_acc = verify_and_extract_acc(
                        &agg_srs2,
                        parent_keys.vk.as_ref(),
                        &parent_keys.fixed_bases,
                        &proof,
                        &public_inputs_fields,
                    );

                    TreeNode {
                        state,
                        proof,
                        proof_acc,
                        pi_acc: accumulated_pi,
                        public_inputs,
                    }
                })
                .collect();

            current_level = next_level;
            child_level = parent_level;
        }

        // Final root and sanity check
        let root = &current_level[0];

        let leaf_states: Vec<F> = client_proofs.iter().map(|p| p.state).collect();
        let client_pis: Vec<[F; 7]> = client_proofs.iter().map(|p| p.public_items).collect();
        let expected_root = poseidon_tree_root(&leaf_states);
        assert_eq!(
            root.state, expected_root,
            "Root state mismatch with recomputed Poseidon tree root"
        );

        let final_level = max_level;
        let final_keys = agg_store.get(final_level);

        AggregationResult {
            root_state: root.state,
            agg_proof: root.proof.clone(),
            agg_public_inputs: root.public_inputs.clone(),
            agg_proof_acc: root.proof_acc.clone(),
            agg_vk: (*final_keys.vk).clone(),
            agg_vk_name: final_keys.name.clone(),
            fixed_base_names: combined_fixed_base_names,
            fixed_bases: combined_fixed_bases,
            leaf_states,
            client_pis,
        }
    }

    /// Final aggregation circuit:
    ///  - recomputes each client's instance hash and Poseidon Merkle root,
    ///  - ties that root to the AGG proof,
    ///  - performs commitment/nullifier state transition checks,
    ///  - AND: computes a "collapsed accumulator" = accumulate( inner_proof_acc , agg_pi_acc )
    ///        and exposes it as public input(s).
    ///
    /// Public inputs exposed:
    ///   * PI0: pre-commitment succinct_repr root
    ///   * PI1: post-commitment succinct_repr root
    ///   * PI2: pre-nullifier succinct_repr root
    ///   * PI3: post-nullifier succinct_repr root
    ///   * PI4.. : collapsed accumulator (acc_pi...)
    #[derive(Clone, Debug)]
    pub struct FinalAggCircuit {
        /// Inner aggregation vk / proof
        pub agg_vk: (EvaluationDomain<F>, ConstraintSystem<F>, F),
        pub agg_vk_name: String,
        pub agg_proof: Value<Vec<u8>>,

        /// The PI accumulator from the inner AggCircuit (must match the one used in `prepare`)
        pub agg_pi_acc: Value<AggAccumulator>,
        /// The "level" field element expected in the inner public inputs
        pub agg_level: F,
        /// Fixed base names needed to assign `agg_pi_acc` deterministically
        pub fixed_base_names: Vec<String>,

        /// For each client leaf, the 7 would-be public inputs:
        ///   [root, pk'_x, pk'_y, new_c1, new_c2, nf1, nf2]
        pub client_pis: [[F; 7]; FINAL_NUM_LEAVES],

        /// Nullifier-set state transition
        pub pre_nullifier_map: Value<Map>,
        pub post_nullifier_root: Value<F>,

        /// Commitment-set state transition
        pub pre_commitment_map: Value<Map>,
        pub post_commitment_root: Value<F>,
    }

    impl Circuit<F> for FinalAggCircuit {
        type Config = (
            NativeConfig,
            P2RDecompositionConfig,
            ForeignEccConfig<C>,
            PoseidonConfig<F>,
        );
        type FloorPlanner = SimpleFloorPlanner;
        type Params = ();

        fn without_witnesses(&self) -> Self {
            unreachable!()
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

            // ------------------------------------------------------------------
            // 1. Commitment & nullifier map transitions (MapMt)
            // ------------------------------------------------------------------
            let mut commit_map_gadget = midnight_circuits::map::map_gadget::MapGadget::<
                F,
                NG,
                PoseidonChip<F>,
            >::new(&scalar_chip, &poseidon_chip);
            commit_map_gadget.init(&mut layouter, self.pre_commitment_map.clone())?;

            // First 2 public inputs: commitment pre/post roots
            let pre_commit_root = commit_map_gadget.succinct_repr();
            scalar_chip.constrain_as_public_input(&mut layouter, &pre_commit_root)?;

            let expected_post_commit_root =
                scalar_chip.assign(&mut layouter, self.post_commitment_root.clone())?;
            scalar_chip.constrain_as_public_input(&mut layouter, &expected_post_commit_root)?;

            let mut null_map_gadget = midnight_circuits::map::map_gadget::MapGadget::<
                F,
                NG,
                PoseidonChip<F>,
            >::new(&scalar_chip, &poseidon_chip);
            null_map_gadget.init(&mut layouter, self.pre_nullifier_map.clone())?;

            // Next 2 public inputs: nullifier pre/post roots
            let pre_null_root = null_map_gadget.succinct_repr();
            scalar_chip.constrain_as_public_input(&mut layouter, &pre_null_root)?;

            let expected_post_null_root =
                scalar_chip.assign(&mut layouter, self.post_nullifier_root.clone())?;
            scalar_chip.constrain_as_public_input(&mut layouter, &expected_post_null_root)?;

            // Values 0 (unspent) and 1 (spent)
            let zero = scalar_chip.assign(&mut layouter, Value::known(F::ZERO))?;
            let one = scalar_chip.assign(&mut layouter, Value::known(F::ONE))?;

            // ------------------------------------------------------------------
            // 2. Recompute client instance hashes and update maps
            // ------------------------------------------------------------------
            assert!(
                FINAL_NUM_LEAVES.is_power_of_two(),
                "FINAL_NUM_LEAVES must be a power of two"
            );

            let mut leaf_hashes: Vec<AssignedNative<F>> = Vec::with_capacity(FINAL_NUM_LEAVES);

            for i in 0..FINAL_NUM_LEAVES {
                let items = self.client_pis[i];

                // Assign the 7 would-be public inputs
                let mut assigned_items: Vec<AssignedNative<F>> = Vec::with_capacity(7);
                for j in 0..7 {
                    assigned_items.push(scalar_chip.assign(&mut layouter, Value::known(items[j]))?);
                }

                let root_pi = assigned_items[0].clone();
                scalar_chip.assert_equal(
                    &mut layouter,
                    &commit_map_gadget.succinct_repr(),
                    &root_pi,
                )?;

                let pkx = assigned_items[1].clone();
                let pky = assigned_items[2].clone();
                let new_c1 = assigned_items[3].clone();
                let new_c2 = assigned_items[4].clone();
                let nf1 = assigned_items[5].clone();
                let nf2 = assigned_items[6].clone();

                // Recompute instance hash:
                // (root, pk'_x, pk'_y) -> acc1
                // (acc1, new_c1, new_c2) -> acc2
                // (acc2, nf1, nf2) -> instance_hash
                let acc1 =
                    poseidon_chip.hash(&mut layouter, &[root_pi, pkx.clone(), pky.clone()])?;
                let acc2 =
                    poseidon_chip.hash(&mut layouter, &[acc1, new_c1.clone(), new_c2.clone()])?;
                let inst_hash =
                    poseidon_chip.hash(&mut layouter, &[acc2, nf1.clone(), nf2.clone()])?;

                leaf_hashes.push(inst_hash.clone());

                // Commitments: just insert (append-only set)
                commit_map_gadget.insert(&mut layouter, &new_c1, &one)?;
                commit_map_gadget.insert(&mut layouter, &new_c2, &one)?;

                // Nullifiers: old_value must be 0, then set to 1
                for nf in [nf1, nf2] {
                    let old_value = null_map_gadget.get(&mut layouter, &nf)?;
                    scalar_chip.assert_equal(&mut layouter, &old_value, &zero)?;
                    null_map_gadget.insert(&mut layouter, &nf, &one)?;
                }
            }

            // Final map roots must equal expected post-state roots
            scalar_chip.assert_equal(
                &mut layouter,
                &commit_map_gadget.succinct_repr(),
                &expected_post_commit_root,
            )?;
            scalar_chip.assert_equal(
                &mut layouter,
                &null_map_gadget.succinct_repr(),
                &expected_post_null_root,
            )?;

            // ------------------------------------------------------------------
            // 3. Poseidon Merkle root of client instance hashes (batch instances)
            // ------------------------------------------------------------------
            let mut level_vals: Vec<AssignedNative<F>> = leaf_hashes;

            while level_vals.len() > 1 {
                let mut next_level = Vec::with_capacity(level_vals.len() / 2);
                for pair in level_vals.chunks(2) {
                    let h =
                        poseidon_chip.hash(&mut layouter, &[pair[0].clone(), pair[1].clone()])?;
                    next_level.push(h);
                }
                level_vals = next_level;
            }

            let root = level_vals[0].clone();

            // ------------------------------------------------------------------
            // 4. Assign vk for the inner AGG circuit and verify the AGG proof
            //    Ensures agg_pi_acc used here is exactly the same used to build
            //    the PI slice passed to `prepare`.
            // ------------------------------------------------------------------
            let vk_val: AssignedNative<F> =
                native_chip.assign_fixed(&mut layouter, self.agg_vk.2)?;
            let assigned_vk = verifier_chip.assign_vk(
                self.agg_vk_name.as_str(),
                &self.agg_vk.0,
                &self.agg_vk.1,
                vk_val,
            )?;

            // Assign the inner PI accumulator witness
            let mut agg_pi_acc = AssignedAccumulator::assign(
                &mut layouter,
                &curve_chip,
                &scalar_chip,
                1,
                1,
                &[],
                &self.fixed_base_names,
                self.agg_pi_acc.clone(),
            )?;
            agg_pi_acc.collapse(&mut layouter, &curve_chip, &scalar_chip)?;

            // Inner level
            let level = scalar_chip.assign(&mut layouter, Value::known(self.agg_level))?;

            // Build the inner public inputs: [root, acc_pi..., level]
            let mut assigned_pi: Vec<AssignedNative<F>> = Vec::new();
            assigned_pi.push(root.clone());
            assigned_pi.extend(verifier_chip.as_public_input(&mut layouter, &agg_pi_acc)?);
            assigned_pi.push(level);

            // Verify the final AGG proof inside the circuit -> proof accumulator
            let id_point: AssignedForeignPoint<
                midnight_curves::Fq,
                midnight_curves::G1Projective,
                midnight_curves::G1Projective,
            > = curve_chip.assign_fixed(&mut layouter, C::identity())?;

            let mut proof_acc: AssignedAccumulator<_> = verifier_chip.prepare(
                &mut layouter,
                &assigned_vk,
                &[("com_instance", id_point)],
                &[&assigned_pi],
                self.agg_proof.clone(),
            )?;
            proof_acc.collapse(&mut layouter, &curve_chip, &scalar_chip)?;

            // ------------------------------------------------------------------
            // 5. Collapse final accumulator = accumulate(proof_acc, agg_pi_acc)
            //    and expose it as public input(s).
            // ------------------------------------------------------------------
            let mut collapsed = AssignedAccumulator::<S>::accumulate(
                &mut layouter,
                &verifier_chip,
                &scalar_chip,
                &poseidon_chip,
                &[proof_acc, agg_pi_acc],
            )?;
            collapsed.collapse(&mut layouter, &curve_chip, &scalar_chip)?;

            let collapsed_pi = verifier_chip.as_public_input(&mut layouter, &collapsed)?;
            for x in collapsed_pi.iter() {
                scalar_chip.constrain_as_public_input(&mut layouter, x)?;
            }

            core_decomp_chip.load(&mut layouter)
        }
    }

    #[allow(dead_code)]
    pub fn demo_poseidon_aggregation() {
        // Kept for local smoke-testing. This uses a 1-instance relation.
        use midnight_circuits::compact_std_lib::{self, MidnightCircuit};
        use midnight_proofs::poly::kzg::KZGCommitmentScheme;

        #[derive(Clone, Default)]
        struct PoseidonExample;

        impl compact_std_lib::Relation for PoseidonExample {
            type Instance = F;
            type Witness = [F; 3];

            fn format_instance(instance: &Self::Instance) -> Result<Vec<F>, Error> {
                Ok(vec![*instance])
            }

            fn circuit(
                &self,
                std_lib: &compact_std_lib::ZkStdLib,
                layouter: &mut impl Layouter<F>,
                _instance: Value<Self::Instance>,
                witness: Value<Self::Witness>,
            ) -> Result<(), Error> {
                let assigned_message = std_lib.assign_many(layouter, &witness.transpose_array())?;
                let output = std_lib.poseidon(layouter, &assigned_message)?;
                std_lib.constrain_as_public_input(layouter, &output)
            }

            fn used_chips(&self) -> compact_std_lib::ZkStdLibArch {
                compact_std_lib::ZkStdLibArch {
                    jubjub: false,
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
                Ok(PoseidonExample)
            }
        }

        let leaf_k = 6;
        let poseidon_srs = filecoin_srs_agg(leaf_k).unwrap();
        let poseidon_relation = PoseidonExample;
        let poseidon_vk = compact_std_lib::setup_vk(&poseidon_srs, &poseidon_relation);
        let poseidon_pk = compact_std_lib::setup_pk(&poseidon_relation, &poseidon_vk);

        let num_leaves = 8;
        println!("Creating {} POSEIDON leaf proofs...", num_leaves);

        let client_proofs: Vec<ClientProof> = (0..num_leaves)
            .map(|i| {
                use midnight_circuits::instructions::hash::HashCPU;
                use rand::SeedableRng;

                let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(i as u64);
                let witness: [F; 3] = core::array::from_fn(|_| F::random(&mut rng));
                let state = <PoseidonChip<F> as HashCPU<F, F>>::hash(&witness);

                let proof = {
                    let mut transcript = CircuitTranscript::<PoseidonState<F>>::init();
                    create_proof::<
                        F,
                        KZGCommitmentScheme<E>,
                        CircuitTranscript<PoseidonState<F>>,
                        MidnightCircuit<PoseidonExample>,
                    >(
                        &poseidon_srs,
                        &poseidon_pk.pk(),
                        &[MidnightCircuit::new(
                            &poseidon_relation,
                            Value::known(state),
                            Value::known(witness),
                            Some(1),
                        )],
                        1,
                        &[&[&[], &[state]]],
                        OsRng,
                        &mut transcript,
                    )
                    .expect("Poseidon proof failed");
                    transcript.finalize()
                };

                println!("POSEIDON leaf {} created", i);
                ClientProof {
                    state,
                    proof,
                    public_items: [state; 7],
                }
            })
            .collect();

        let agg_res = aggregate_client_proofs(
            &poseidon_srs,
            poseidon_vk.vk(),
            "poseidon_vk",
            leaf_k,
            &client_proofs,
        );

        println!("\n=== AGG Tree Complete (via aggregation function) ===");
        println!("Root state: {:?}", agg_res.root_state);
        println!("Aggregated proof length: {} bytes", agg_res.agg_proof.len());
    }
}

// Re-export pieces we need in the shielded example.
use midnight_circuits::instructions::map::MapCPU;
use proof_agg::{
    AggAccumulator, ClientProof as AggClientProof, FINAL_NUM_LEAVES, FinalAggCircuit,
    accumulator_as_public_input, aggregate_client_proofs,
};

use crate::proof_agg::filecoin_srs_agg;

// -----------------------------------------------------------------------------
// Original shielded Spend2Output2 code, modified so client circuits use the SAME
// MapMt commitment state as the batch transition (no hand-rolled Merkle paths).
// -----------------------------------------------------------------------------

const UTXO_COMMIT_TAG: u64 = 0x0001;
const UTXO_NULLIFY_TAG: u64 = 0x0002;
const AMOUNT_BITS: u32 = 128; // 128-bit integers for amounts
const AMOUNT_GEN_BITS: u32 = 120; // generate up to 120 bits to avoid u128 overflow on sums
const BATCH_SIZE: usize = 8; // must be a power of two and == FINAL_NUM_LEAVES

// UTXO structure
#[derive(Clone, Debug)]
pub struct Utxo {
    pub asset_id: F,
    pub amount: u128, // 128-bit host-side amount
    pub randomness: F,
}

// -------------------- Circuit relation (single public instance = Poseidon hash) --------------------

#[derive(Clone, Default)]
pub struct Spend2Output2;

impl Relation for Spend2Output2 {
    type Instance = F;

    // PATCH: replace the two Merkle paths with a single pre-state commitment map witness.
    // The circuit proves membership of the two consumed commitments via MapGadget::get,
    // and uses commit_map.succinct_repr() as the "root" hashed into the instance.
    type Witness = (
        MapMt<F, PoseidonChip<F>>, // pre-state commitment map
        JubjubScalar,              // sk
        JubjubScalar,              // alpha (blinding factor)
        Utxo,
        Utxo,
        Utxo,
        Utxo,
        (F, F), // (pk_out1_x, pk_out1_y)
        (F, F), // (pk_out2_x, pk_out2_y)
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
        // Extract witness components (Values only; assignments happen once below)
        let commit_map_val = witness.clone().map(|(m, _, _, _, _, _, _, _, _)| m);

        let sk_val = witness.clone().map(|(_, sk, _, _, _, _, _, _, _)| sk);
        let alpha_val = witness.clone().map(|(_, _, alpha, _, _, _, _, _, _)| alpha);

        let old1_val = witness.clone().map(|(_, _, _, o1, _, _, _, _, _)| o1);
        let old2_val = witness.clone().map(|(_, _, _, _, o2, _, _, _, _)| o2);
        let new1_val = witness.clone().map(|(_, _, _, _, _, n1, _, _, _)| n1);
        let new2_val = witness.clone().map(|(_, _, _, _, _, _, n2, _, _)| n2);

        let pk1x_val = witness.clone().map(|(_, _, _, _, _, _, _, k1, _)| k1.0);
        let pk1y_val = witness.clone().map(|(_, _, _, _, _, _, _, k1, _)| k1.1);
        let pk2x_val = witness.clone().map(|(_, _, _, _, _, _, _, _, k2)| k2.0);
        let pk2y_val = witness.clone().map(|(_, _, _, _, _, _, _, _, k2)| k2.1);

        // Assign sender secret once, derive sender pk once
        let sk: AssignedScalarOfNativeCurve<Jubjub> = std_lib.jubjub().assign(layouter, sk_val)?;
        let generator = std_lib
            .jubjub()
            .assign_fixed(layouter, JubjubSubgroup::generator())?;
        let pk_sender = std_lib.jubjub().mul(layouter, &sk, &generator)?;
        let pk_sender_fields = std_lib.jubjub().as_public_input(layouter, &pk_sender)?;
        let (pk_sx, pk_sy) = (pk_sender_fields[0].clone(), pk_sender_fields[1].clone());

        // Blinded key: pk' = pk + [alpha]G
        let alpha: AssignedScalarOfNativeCurve<Jubjub> =
            std_lib.jubjub().assign(layouter, alpha_val)?;
        let blind = std_lib.jubjub().mul(layouter, &alpha, &generator)?;
        let pk_blinded = std_lib.jubjub().add(layouter, &pk_sender, &blind)?;
        let pk_blinded_fields = std_lib.jubjub().as_public_input(layouter, &pk_blinded)?;
        let (pk_bx, pk_by) = (pk_blinded_fields[0].clone(), pk_blinded_fields[1].clone());

        // Assign each UTXO's fields exactly once
        let old1_asg = assign_utxo(std_lib, layouter, &old1_val)?;
        let old2_asg = assign_utxo(std_lib, layouter, &old2_val)?;
        let new1_asg = assign_utxo(std_lib, layouter, &new1_val)?;
        let new2_asg = assign_utxo(std_lib, layouter, &new2_val)?;

        // old commitments (must match UNBLINDED sender pk)
        let old_c1 = compute_commitment_from_parts(std_lib, layouter, &old1_asg, &pk_sx, &pk_sy)?;
        let old_c2 = compute_commitment_from_parts(std_lib, layouter, &old2_asg, &pk_sx, &pk_sy)?;

        // PATCH: Use MapGadget over the provided pre-state commitment map to prove membership
        // and to derive the state root.

        let mut commit_map_gadget: midnight_circuits::map::map_gadget::MapGadget<
            midnight_curves::Fq,
            midnight_circuits::field::NativeGadget<
                midnight_curves::Fq,
                midnight_circuits::field::decomposition::chip::P2RDecompositionChip<
                    midnight_curves::Fq,
                >,
                midnight_circuits::field::NativeChip<midnight_curves::Fq>,
            >,
            PoseidonChip<midnight_curves::Fq>,
        > = std_lib.map_gadget().clone();
        commit_map_gadget.init(layouter, commit_map_val)?;

        let one = std_lib.assign_fixed(layouter, F::ONE)?;

        let v1 = commit_map_gadget.get(layouter, &old_c1)?;
        let v2 = commit_map_gadget.get(layouter, &old_c2)?;
        std_lib.assert_equal(layouter, &v1, &one)?;
        std_lib.assert_equal(layouter, &v2, &one)?;

        let root = commit_map_gadget.succinct_repr();

        // Nullifiers (BOUND TO UNBLINDED sender pk to prevent double-spends)
        let nf1 = compute_nullifier(std_lib, layouter, &old_c1, &pk_sx, &pk_sy)?;
        let nf2 = compute_nullifier(std_lib, layouter, &old_c2, &pk_sx, &pk_sy)?;
        std_lib.assert_not_equal(layouter, &nf1, &nf2)?;

        // New outputs: use provided recipient (pk_out*) coordinates (assigned once)
        let pk1x = std_lib.assign(layouter, pk1x_val)?;
        let pk1y = std_lib.assign(layouter, pk1y_val)?;
        let pk2x = std_lib.assign(layouter, pk2x_val)?;
        let pk2y = std_lib.assign(layouter, pk2y_val)?;

        let new_c1 = compute_commitment_from_parts(std_lib, layouter, &new1_asg, &pk1x, &pk1y)?;
        let new_c2 = compute_commitment_from_parts(std_lib, layouter, &new2_asg, &pk2x, &pk2y)?;
        std_lib.assert_not_equal(layouter, &new_c1, &new_c2)?;

        // Value conservation
        check_value_conservation_assigned(
            std_lib, layouter, &old1_asg, &old2_asg, &new1_asg, &new2_asg,
        )?;

        // ---- Single public input: Poseidon hash using BLINDED pk and Map root ----
        let acc1 = std_lib.poseidon(layouter, &[root.clone(), pk_bx.clone(), pk_by.clone()])?;
        let acc2 = std_lib.poseidon(layouter, &[acc1, new_c1.clone(), new_c2.clone()])?;
        let instance_hash = std_lib.poseidon(layouter, &[acc2, nf1.clone(), nf2.clone()])?;

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

fn host_instance_hash(items: [F; 7]) -> F {
    let acc1 = <PoseidonChip<F> as HashCPU<F, F>>::hash(&[items[0], items[1], items[2]]);
    let acc2 = <PoseidonChip<F> as HashCPU<F, F>>::hash(&[acc1, items[3], items[4]]);
    <PoseidonChip<F> as HashCPU<F, F>>::hash(&[acc2, items[5], items[6]])
}

#[derive(Clone, Debug)]
struct Note {
    utxo: Utxo,
    commit: F,
    spent: bool,
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
    use midnight_circuits::instructions::map::MapCPU;

    // PATCH: distinct leaf VK name (must be consistent anywhere fixed bases/names are derived).
    const LEAF_VK_NAME: &str = "spend2output2_vk";

    const K: u32 = 14;
    const NUM_ACCOUNTS: usize = 4;
    const NUM_SEED_DEPOSITS_PER_ACCOUNT: usize = 3;
    const NUM_TRANSFERS: usize = 120;

    assert_eq!(
        BATCH_SIZE, FINAL_NUM_LEAVES,
        "BATCH_SIZE must equal FINAL_NUM_LEAVES"
    );

    let srs = filecoin_srs_agg(K).unwrap();
    let relation = Spend2Output2;
    let vk = compact_std_lib::setup_vk(&srs, &relation);
    let pk = compact_std_lib::setup_pk(&relation, &vk);

    let mut rng = ChaCha8Rng::from_entropy();
    let asset_id = F::random(&mut rng);

    // PATCH: the commitment state is the MapMt, and its succinct_repr() is the state root.
    let mut commitment_map = MapMt::<F, PoseidonChip<F>>::new(&F::ZERO);
    let mut nullifier_map = MapMt::<F, PoseidonChip<F>>::new(&F::ZERO);

    // Create accounts
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

    // Seed deposits
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
            });
        }
    }
    println!(
        "Initial commitment root: {:?}",
        commitment_map.succinct_repr()
    );

    let choose_sender = |rng: &mut ChaCha8Rng, accs: &mut [Account]| -> Option<usize> {
        let viable: Vec<usize> = accs
            .iter()
            .enumerate()
            .filter(|(_, a)| a.wallet.iter().filter(|n| !n.spent).count() >= 2)
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

            let sender_idx = match choose_sender(&mut rng, &mut shadow_accounts) {
                Some(i) => i,
                None => {
                    println!(
                        "[batch {}] no account has two spendable notes; stopping batching.",
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
                    .filter(|(_, n)| !n.spent)
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

            // PATCH: root_before is the MapMt succinct repr BEFORE inserting the new commitments.
            let root_before = shadow_commitment_map.succinct_repr();

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

            // We can update nullifier state immediately (Spend2Output2 doesn't read it),
            // but commitment updates MUST happen after proving so the map witness is pre-state.
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

            // PATCH: witness includes the *pre-state* commitment map.
            let witness = (
                shadow_commitment_map.clone(),
                sender.sk,
                alpha,
                old1.utxo.clone(),
                old2.utxo.clone(),
                new1.clone(),
                new2.clone(),
                (shadow_accounts[r1].pk_x, shadow_accounts[r1].pk_y),
                (shadow_accounts[r2].pk_x, shadow_accounts[r2].pk_y),
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

            client_proofs.push(AggClientProof {
                state: instance,
                proof: proof.clone(),
                public_items,
            });

            // Now apply the state transition for subsequent txs.
            shadow_commitment_map.insert(&new1_commit, &F::ONE);
            shadow_commitment_map.insert(&new2_commit, &F::ONE);

            shadow_accounts[sender_idx].wallet[i_old1].spent = true;
            shadow_accounts[sender_idx].wallet[i_old2].spent = true;

            shadow_accounts[r1].wallet.push(Note {
                utxo: new1,
                commit: new1_commit,
                spent: false,
            });
            shadow_accounts[r2].wallet.push(Note {
                utxo: new2,
                commit: new2_commit,
                spent: false,
            });

            let root_after = shadow_commitment_map.succinct_repr();

            println!(
                "[batch {}, tx {}] shadow commitment root updated: {:?}",
                batch_idx, total_transfers_done, root_after
            );

            total_transfers_done += 1;
        }

        if batch_failed || client_proofs.is_empty() {
            break 'outer;
        }

        assert_eq!(
            client_proofs.len(),
            BATCH_SIZE,
            "Batch not completely filled; aborting aggregation."
        );
        assert!(
            client_proofs.len().is_power_of_two(),
            "Batch size must be a power of two."
        );

        let now = Instant::now();
        // PATCH: use correct leaf vk name (must match fixed-base naming expectations).
        let agg_result = aggregate_client_proofs(&srs, vk.vk(), LEAF_VK_NAME, K, &client_proofs);
        println!(
            "Batch {} aggregated proof generated and internally verified in {:?}",
            batch_idx,
            now.elapsed()
        );
        println!(
            "Batch {} aggregated state: {:?}, agg proof length: {} bytes",
            batch_idx,
            agg_result.root_state,
            agg_result.agg_proof.len()
        );

        let pre_nullifier_root = pre_nullifier_map_for_batch.succinct_repr();
        let post_nullifier_root = shadow_nullifier_map.succinct_repr();

        let pre_commitment_root = pre_commitment_map_for_batch.succinct_repr();
        let post_commitment_root = shadow_commitment_map.succinct_repr();

        // ---------------------------------------------------------------------
        // Create and verify a FinalAggCircuit proof for *this* aggregation
        // ---------------------------------------------------------------------
        {
            use midnight_proofs::poly::kzg::KZGCommitmentScheme;
            use midnight_proofs::{
                circuit::Value,
                plonk::{create_proof, keygen_pk, keygen_vk_with_k, prepare},
                poly::EvaluationDomain,
                transcript::CircuitTranscript,
            };

            assert_eq!(
                agg_result.client_pis.len(),
                FINAL_NUM_LEAVES,
                "FINAL_NUM_LEAVES must match batch size"
            );

            let mut client_pis_array = [[F::ZERO; 7]; FINAL_NUM_LEAVES];
            for i in 0..FINAL_NUM_LEAVES {
                client_pis_array[i] = agg_result.client_pis[i];
            }

            // Build vk data for the inner AggCircuit
            let agg_domain = EvaluationDomain::<F>::new(
                agg_result.agg_vk.cs().degree() as u32,
                proof_agg::AGG_K,
            );
            let agg_vk_data = (
                agg_domain.clone(),
                agg_result.agg_vk.cs().clone(),
                agg_result.agg_vk.transcript_repr(),
            );

            // Compute the "collapsed accumulator" that FinalAggCircuit will expose:
            // collapsed = accumulate( inner_proof_acc , agg_pi_acc )
            let mut collapsed: AggAccumulator = AggAccumulator::accumulate(&[
                agg_result.agg_proof_acc.clone(),
                agg_result.agg_public_inputs.pi_acc.clone(),
            ]);
            collapsed.collapse();
            let collapsed_pi = accumulator_as_public_input(&collapsed);

            // Shape-only circuit for keygen
            let default_final_circuit = FinalAggCircuit {
                agg_vk: agg_vk_data.clone(),
                agg_vk_name: agg_result.agg_vk_name.clone(),
                agg_proof: Value::unknown(),
                agg_pi_acc: Value::unknown(),
                agg_level: F::ZERO,
                fixed_base_names: agg_result.fixed_base_names.clone(),
                client_pis: [[F::ZERO; 7]; FINAL_NUM_LEAVES],
                pre_nullifier_map: Value::unknown(),
                post_nullifier_root: Value::unknown(),
                pre_commitment_map: Value::unknown(),
                post_commitment_root: Value::unknown(),
            };

            let agg_srs = proof_agg::filecoin_srs_agg(proof_agg::AGG_K).unwrap();
            let final_vk = keygen_vk_with_k(&agg_srs, &default_final_circuit, proof_agg::AGG_K)
                .expect("final vk generation should not fail");
            let final_pk = keygen_pk(final_vk.clone(), &default_final_circuit)
                .expect("final pk generation should not fail");

            // Actual final circuit instance
            let final_circuit = FinalAggCircuit {
                agg_vk: agg_vk_data,
                agg_vk_name: agg_result.agg_vk_name.clone(),
                agg_proof: Value::known(agg_result.agg_proof.clone()),
                agg_pi_acc: Value::known(agg_result.agg_public_inputs.pi_acc.clone()),
                agg_level: agg_result.agg_public_inputs.level,
                fixed_base_names: agg_result.fixed_base_names.clone(),
                client_pis: client_pis_array,
                pre_nullifier_map: Value::known(pre_nullifier_map_for_batch.clone()),
                post_nullifier_root: Value::known(post_nullifier_root),
                pre_commitment_map: Value::known(pre_commitment_map_for_batch.clone()),
                post_commitment_root: Value::known(post_commitment_root),
            };

            // Public inputs:
            // [pre_commit, post_commit, pre_null, post_null, collapsed_acc_pi...]
            let mut final_public_inputs: Vec<F> = vec![
                pre_commitment_root,
                post_commitment_root,
                pre_nullifier_root,
                post_nullifier_root,
            ];
            final_public_inputs.extend(collapsed_pi.clone());

            let final_proof_bytes = {
                let mut transcript = CircuitTranscript::<PoseidonState<F>>::init();
                create_proof::<
                    F,
                    KZGCommitmentScheme<midnight_curves::Bls12>,
                    CircuitTranscript<PoseidonState<F>>,
                    FinalAggCircuit,
                >(
                    &agg_srs,
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

            // Verify the final aggregation proof
            let mut transcript =
                CircuitTranscript::<PoseidonState<F>>::init_from_bytes(&final_proof_bytes);
            let committed_bases: &[&[midnight_curves::G1Projective]] =
                &[&[midnight_curves::G1Projective::identity()]];
            let instances: &[&[&[F]]] = &[&[&final_public_inputs]];

            let dual_msm = prepare::<
                F,
                KZGCommitmentScheme<midnight_curves::Bls12>,
                CircuitTranscript<PoseidonState<F>>,
            >(&final_vk, committed_bases, instances, &mut transcript)
            .expect("Final aggregation verification preparation failed");

            assert!(
                dual_msm.check(&agg_srs.verifier_params()),
                "Final aggregation proof must verify"
            );

            // verify the output accumulator attesting to the truthness of the client proofs
            assert!(
                collapsed.check(&agg_srs.s_g2().into(), &agg_result.fixed_bases),
                "Final aggregation collapsed accumulator must verify"
            );

            println!(
                "\n✅ Final aggregation proof for batch {} verified.\n\
                 Shared state Merkle root (internal to circuit): {:?}\n\
                 Commitment-set transition: {:?} -> {:?}\n\
                 Nullifier-set transition: {:?} -> {:?}\n\
                 Collapsed accumulator PI length: {} field elements",
                batch_idx,
                agg_result.root_state,
                pre_commitment_root,
                post_commitment_root,
                pre_nullifier_root,
                post_nullifier_root,
                collapsed_pi.len()
            );
        }

        accounts = shadow_accounts;
        nullifier_map = shadow_nullifier_map;
        commitment_map = shadow_commitment_map;

        println!(
            "After batch {} committed commitment root: {:?}",
            batch_idx,
            commitment_map.succinct_repr()
        );

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
