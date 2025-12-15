use std::collections::HashSet;
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
        PublicInputInstructions, hash::HashCPU,
    },
    testing_utils::plonk_api::filecoin_srs,
    types::{AssignedBit, AssignedNative, AssignedNativePoint, Instantiable},
};

use midnight_curves::{Fq as F, Fr as JubjubScalar, JubjubExtended as Jubjub, JubjubSubgroup};
use midnight_proofs::{
    circuit::{Layouter, Value},
    plonk::Error,
};
use rand::{Rng, SeedableRng, rngs::OsRng};
use rand_chacha::ChaCha8Rng;

// -----------------------------------------------------------------------------
// Proof aggregator module (adapted to be reusable for any 1-instance circuit)
// -----------------------------------------------------------------------------
mod proof_agg {
    use halo2curves::{ff::Field, group::Group};
    use midnight_circuits::compact_std_lib::MidnightCircuit;
    use midnight_circuits::hash::poseidon::PoseidonState;
    use midnight_circuits::types::{AssignedBit, AssignedForeignPoint, InnerValue};
    use midnight_circuits::{
        compact_std_lib::{self, Relation, ZkStdLib, ZkStdLibArch},
        ecc::{
            curves::CircuitCurve,
            foreign::{ForeignEccChip, ForeignEccConfig, nb_foreign_ecc_chip_columns},
        },
        field::{
            NativeChip, NativeConfig, NativeGadget,
            decomposition::{
                chip::{P2RDecompositionChip, P2RDecompositionConfig},
                pow2range::Pow2RangeChip,
            },
            foreign::FieldChip,
            native::NB_ARITH_COLS,
        },
        hash::poseidon::{
            NB_POSEIDON_ADVICE_COLS, NB_POSEIDON_FIXED_COLS, PoseidonChip, PoseidonConfig,
        },
        instructions::*,
        types::{AssignedNative, ComposableChip, Instantiable},
        verifier::{
            Accumulator, AssignedAccumulator, AssignedVk, BlstrsEmulation, SelfEmulation,
            VerifierGadget,
        },
    };
    use midnight_curves::Bls12;
    use midnight_proofs::poly::kzg::params::ParamsKZG;
    use midnight_proofs::utils::SerdeFormat;
    use midnight_proofs::{
        circuit::{Layouter, SimpleFloorPlanner, Value},
        plonk::{
            Circuit, ConstraintSystem, Error, VerifyingKey, create_proof, keygen_pk,
            keygen_vk_with_k, prepare,
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
    use std::time::Instant;

    pub type S = BlstrsEmulation;
    type F = <S as SelfEmulation>::F;
    type C = <S as SelfEmulation>::C;
    type E = <S as SelfEmulation>::Engine;
    type CBase = <C as CircuitCurve>::Base;
    type NG = NativeGadget<F, P2RDecompositionChip<F>, NativeChip<F>>;

    const K: u32 = 20;
    const POSEIDON_K: u32 = 6;

    pub fn filecoin_srs_agg(k: u32) -> ParamsKZG<Bls12> {
        assert!(k <= 20, "We don't have an SRS for circuits of size {k}");
        let srs_dir = env::var("SRS_DIR").unwrap_or("./examples/assets".into());
        let srs_path = format!("{srs_dir}/bls_filecoin_2p{k:?}");
        let mut fetching_path = srs_path.clone();

        if !Path::new(fetching_path.as_str()).exists() {
            fetching_path = format!("{srs_dir}/bls_filecoin_2p20")
        }

        let params_fs = File::open(Path::new(&fetching_path)).unwrap_or_else(|_| {
            panic!("\nIt seems you have not downloaded and/or parsed the SRS from filecoin.")
        });

        let mut params: ParamsKZG<Bls12> = ParamsKZG::read_custom::<_>(
            &mut BufReader::new(params_fs),
            SerdeFormat::RawBytesUnchecked,
        )
        .expect("Failed to read params");

        if fetching_path != srs_path {
            params.downsize(k);
            let mut buf = Vec::new();
            params
                .write_custom(&mut buf, SerdeFormat::RawBytesUnchecked)
                .expect("Failed to write params");
            let mut file = File::create(srs_path).expect("Failed to create file");
            file.write_all(&buf[..])
                .expect("Failed to write params to file");
        }

        params
    }

    fn binary_select_vk(
        layouter: &mut impl Layouter<F>,
        native_chip: &NativeChip<F>,
        poseidon_vk_prefix: &[AssignedNative<F>],
        agg_vk_prefix: &[AssignedNative<F>],
        bit: &AssignedBit<F>,
    ) -> Result<Vec<AssignedNative<F>>, Error> {
        assert_eq!(poseidon_vk_prefix.len(), agg_vk_prefix.len());
        let mut out = Vec::with_capacity(poseidon_vk_prefix.len());
        for i in 0..poseidon_vk_prefix.len() {
            let sel =
                native_chip.select(layouter, bit, &poseidon_vk_prefix[i], &agg_vk_prefix[i])?;
            out.push(sel);
        }
        Ok(out)
    }

    #[derive(Clone, Default)]
    pub struct PoseidonExample;

    impl Relation for PoseidonExample {
        type Instance = F;
        type Witness = [F; 3];

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
            let assigned_message = std_lib.assign_many(layouter, &witness.transpose_array())?;
            let output = std_lib.poseidon(layouter, &assigned_message)?;
            std_lib.constrain_as_public_input(layouter, &output)
        }

        fn used_chips(&self) -> ZkStdLibArch {
            ZkStdLibArch {
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

    #[derive(Clone, Debug)]
    pub struct AggCircuit {
        agg_vk: (EvaluationDomain<F>, ConstraintSystem<F>, Value<F>),
        agg_vk_name: &'static str,
        poseidon_vk: (EvaluationDomain<F>, ConstraintSystem<F>, Value<F>),
        leaf_agg_vk: (EvaluationDomain<F>, ConstraintSystem<F>, Value<F>),
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

    impl Circuit<F> for AggCircuit {
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
            let core_decomp_chip = P2RDecompositionChip::new(&config.1, &(K as usize - 1));
            let scalar_chip = NativeGadget::new(core_decomp_chip.clone(), native_chip.clone());
            let curve_chip = ForeignEccChip::new(&config.2, &scalar_chip, &scalar_chip);
            let poseidon_chip = PoseidonChip::new(&config.3, &native_chip);
            let verifier_chip = VerifierGadget::new(&curve_chip, &scalar_chip, &poseidon_chip);

            // Assign and compute level information
            let prev_level = scalar_chip.assign(&mut layouter, self.prev_level)?;
            let next_level = scalar_chip.add_constant(&mut layouter, &prev_level, F::ONE)?;
            let is_genesis = scalar_chip.is_equal_to_fixed(&mut layouter, &prev_level, F::ZERO)?;
            let children_are_genesis =
                scalar_chip.is_equal_to_fixed(&mut layouter, &prev_level, F::ONE)?;
            let is_level_2 =
                scalar_chip.is_equal_to_fixed(&mut layouter, &prev_level, F::from(2u64))?;

            // Assign VKs
            let poseidon_vk: AssignedNative<F> =
                native_chip.assign(&mut layouter, self.poseidon_vk.2)?;
            let leaf_agg_vk = native_chip.assign(&mut layouter, self.leaf_agg_vk.2)?;
            let agg_vk = native_chip.assign(&mut layouter, self.agg_vk.2)?;

            // Select correct VK based on level
            let vk_val =
                native_chip.select(&mut layouter, &children_are_genesis, &leaf_agg_vk, &agg_vk)?;
            let vk_val = native_chip.select(&mut layouter, &is_genesis, &poseidon_vk, &vk_val)?;

            let assigned_vk = verifier_chip.assign_vk(
                self.agg_vk_name,
                if self.is_leaf {
                    &self.poseidon_vk.0
                } else {
                    &self.agg_vk.0
                },
                if self.is_leaf {
                    &self.poseidon_vk.1
                } else {
                    &self.agg_vk.1
                },
                vk_val.clone(),
            )?;
            native_chip.constrain_as_public_input(&mut layouter, &vk_val)?;

            // Assign inner VKs for public input selection
            let assigned_vk_poseidon = verifier_chip.assign_vk(
                "poseidon_vk",
                &self.poseidon_vk.0,
                &self.poseidon_vk.1,
                poseidon_vk,
            )?;
            let assigned_vk_agg =
                verifier_chip.assign_vk("agg_vk", &self.agg_vk.0, &self.agg_vk.1, agg_vk)?;
            let assigned_vk_leaf_agg =
                verifier_chip.assign_vk("agg_vk", &self.agg_vk.0, &self.agg_vk.1, leaf_agg_vk)?;

            let poseidon_vk_elts =
                verifier_chip.as_public_input(&mut layouter, &assigned_vk_poseidon)?;
            let agg_vk_elts = verifier_chip.as_public_input(&mut layouter, &assigned_vk_agg)?;
            let leaf_vk_elts =
                verifier_chip.as_public_input(&mut layouter, &assigned_vk_leaf_agg)?;

            // Select inner VK public inputs based on level
            let vk_inner_pi = binary_select_vk(
                &mut layouter,
                &native_chip,
                &poseidon_vk_elts,
                &agg_vk_elts,
                &children_are_genesis,
            )?;
            let vk_inner_pi = binary_select_vk(
                &mut layouter,
                &native_chip,
                &leaf_vk_elts,
                &vk_inner_pi,
                &is_level_2,
            )?;

            // Compute next state
            let left_state: AssignedNative<F> =
                scalar_chip.assign(&mut layouter, self.left_state)?;
            let right_state: AssignedNative<F> =
                scalar_chip.assign(&mut layouter, self.right_state)?;
            let next_state =
                poseidon_chip.hash(&mut layouter, &[left_state.clone(), right_state.clone()])?;
            scalar_chip.constrain_as_public_input(&mut layouter, &next_state)?;

            let id_point: AssignedForeignPoint<
                midnight_curves::Fq,
                midnight_curves::G1Projective,
                midnight_curves::G1Projective,
            > = curve_chip.assign_fixed(&mut layouter, C::identity())?;

            // Process left child
            let left_acc = AssignedAccumulator::assign(
                &mut layouter,
                &curve_chip,
                &scalar_chip,
                1,
                1,
                &[],
                &self.fixed_base_names,
                self.left_acc.clone(),
            )?;

            let assigned_left_pi = if self.is_leaf {
                vec![left_state.clone()]
            } else {
                let mut v = vk_inner_pi.clone();
                v.push(left_state.clone());
                v.extend(verifier_chip.as_public_input(&mut layouter, &left_acc)?);
                v.push(prev_level.clone());
                v
            };

            let mut left_proof_acc = verifier_chip.prepare(
                &mut layouter,
                &assigned_vk,
                &[("com_instance", id_point.clone())],
                &[&assigned_left_pi],
                self.left_proof.clone(),
            )?;
            left_proof_acc.collapse(&mut layouter, &curve_chip, &scalar_chip)?;

            // Process right child
            let right_acc = AssignedAccumulator::assign(
                &mut layouter,
                &curve_chip,
                &scalar_chip,
                1,
                1,
                &[],
                &self.fixed_base_names,
                self.right_acc.clone(),
            )?;

            let assigned_right_pi = if self.is_leaf {
                vec![right_state.clone()]
            } else {
                let mut v = vk_inner_pi;
                v.push(right_state.clone());
                v.extend(verifier_chip.as_public_input(&mut layouter, &right_acc)?);
                v.push(prev_level);
                v
            };

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
            verifier_chip.constrain_as_public_input(&mut layouter, &next_acc)?;
            scalar_chip.constrain_as_public_input(&mut layouter, &next_level)?;

            core_decomp_chip.load(&mut layouter)
        }
    }

    #[derive(Clone, Debug)]
    struct TreeNode {
        state: F,
        proof: Vec<u8>,
        proof_acc: Accumulator<S>,
        pi_acc: Accumulator<S>,
    }

    #[derive(Clone, Debug)]
    pub struct ClientProof {
        /// Public instance = single field element for the client's witness
        pub state: F,
        /// The client's proof (created off-chain by the client)
        pub proof: Vec<u8>,
    }

    fn fixed_base_names_for(
        vk_name: &str,
        cs: &midnight_proofs::plonk::ConstraintSystem<F>,
    ) -> Vec<String> {
        let mut names = vec![String::from("com_instance"), String::from("~G")];
        names.extend(midnight_circuits::verifier::fixed_base_names::<S>(
            vk_name,
            cs.num_fixed_columns() + cs.num_selectors(),
            cs.permutation().columns.len(),
        ));
        names
    }

    fn trivial_acc_with_names(names: &[String]) -> midnight_circuits::verifier::Accumulator<S> {
        use midnight_circuits::verifier::Msm;
        use std::collections::BTreeMap;
        let fixed: BTreeMap<String, F> = names.iter().cloned().map(|n| (n, F::ZERO)).collect();

        midnight_circuits::verifier::Accumulator::<S>::new(
            Msm::new(&[C::default()], &[F::ONE], &BTreeMap::new()),
            Msm::new(&[C::default()], &[F::ONE], &fixed),
        )
    }

    fn poseidon_tree_root(leaf_states: &[F]) -> F {
        use midnight_circuits::instructions::hash::HashCPU;

        assert!(!leaf_states.is_empty(), "Need at least one leaf");
        assert!(
            leaf_states.len().is_power_of_two(),
            "Number of leaves must be a power of two for this simple tree"
        );

        let mut level_states = leaf_states.to_vec();

        while level_states.len() > 1 {
            assert!(
                level_states.len() % 2 == 0,
                "Level size must stay even while building the tree"
            );

            let mut next_level = Vec::with_capacity(level_states.len() / 2);

            for pair in level_states.chunks(2) {
                let parent = <PoseidonChip<F> as HashCPU<F, F>>::hash(&[pair[0], pair[1]]);
                next_level.push(parent);
            }

            level_states = next_level;
        }

        level_states[0]
    }

    fn verify_and_extract_acc(
        srs: &ParamsKZG<Bls12>,
        vk: &midnight_proofs::plonk::VerifyingKey<F, KZGCommitmentScheme<E>>,
        fixed_bases: &BTreeMap<String, C>,
        proof: &[u8],
        plain_public_inputs: &[F],
    ) -> Accumulator<S> {
        let mut transcript = CircuitTranscript::<PoseidonState<F>>::init_from_bytes(proof);
        let committed_bases: &[&[C]] = &[&[C::identity()]];
        let instances: &[&[&[F]]] = &[&[plain_public_inputs]];

        let dual_msm = prepare::<F, KZGCommitmentScheme<E>, CircuitTranscript<PoseidonState<F>>>(
            vk,
            committed_bases,
            instances,
            &mut transcript,
        )
        .expect("Verification failed");

        assert!(dual_msm.clone().check(&srs.verifier_params()));

        let mut acc: Accumulator<S> = dual_msm.into();
        acc.extract_fixed_bases(fixed_bases);
        acc.collapse();

        assert!(
            acc.check(&srs.s_g2().into(), fixed_bases),
            "Accumulator verification failed"
        );

        acc
    }

    /// Aggregates a list of client proofs into a single AGG proof.
    ///
    /// Requirements:
    /// - `client_proofs.len() > 0`
    /// - `client_proofs.len()` is a power of two
    ///
    /// Returns:
    /// - `(root_state, root_agg_proof_bytes)`
    pub fn aggregate_client_proofs(
        leaf_srs: &ParamsKZG<Bls12>,
        leaf_vk: &VerifyingKey<F, KZGCommitmentScheme<E>>,
        leaf_vk_name: &'static str,
        leaf_k: u32,
        client_proofs: &[ClientProof],
    ) -> (F, Vec<u8>) {
        assert!(!client_proofs.is_empty(), "Need at least one client proof");
        assert!(
            client_proofs.len().is_power_of_two(),
            "Number of client proofs must be a power of two"
        );

        //
        // 1. Leaf vk data (for any 1-instance circuit)
        //
        let poseidon_vk_data = (
            EvaluationDomain::new(leaf_vk.cs().degree() as u32, leaf_k),
            leaf_vk.cs().clone(),
            Value::known(leaf_vk.transcript_repr()),
        );

        //
        // 2. Configure aggregation circuit and generate vk/pk
        //
        let mut agg_cs = ConstraintSystem::default();
        configure_agg_circuit(&mut agg_cs);
        let agg_domain = EvaluationDomain::new(agg_cs.degree() as u32, K);

        let combined_fixed_base_names_keygen: Vec<String> = {
            let poseidon_fb = fixed_base_names_for(leaf_vk_name, &poseidon_vk_data.1);
            let leaf_agg_fb = fixed_base_names_for("agg_vk", &agg_cs);
            let agg_fb = fixed_base_names_for("agg_vk", &agg_cs);

            let mut set = BTreeSet::new();
            let mut v = Vec::new();
            for name in poseidon_fb
                .iter()
                .chain(leaf_agg_fb.iter())
                .chain(agg_fb.iter())
            {
                if set.insert(name.clone()) {
                    v.push(name.clone());
                }
            }
            v
        };

        let default_agg_circuit = AggCircuit {
            agg_vk: (agg_domain.clone(), agg_cs.clone(), Value::unknown()),
            agg_vk_name: "agg_vk",
            poseidon_vk: poseidon_vk_data.clone(),
            left_state: Value::unknown(),
            right_state: Value::unknown(),
            left_proof: Value::unknown(),
            right_proof: Value::unknown(),
            left_acc: Value::unknown(),
            right_acc: Value::unknown(),
            fixed_base_names: combined_fixed_base_names_keygen.clone(),
            is_leaf: false,
            prev_level: Value::unknown(),
            leaf_agg_vk: (agg_domain.clone(), agg_cs.clone(), Value::unknown()),
        };

        let agg_srs = filecoin_srs_agg(K);
        let agg_vk = keygen_vk_with_k(&agg_srs, &default_agg_circuit, K).unwrap();
        let agg_pk = keygen_pk(agg_vk.clone(), &default_agg_circuit).unwrap();

        let default_leaf_agg_circuit = AggCircuit {
            agg_vk: (agg_domain.clone(), agg_cs.clone(), Value::unknown()),
            agg_vk_name: leaf_vk_name,
            poseidon_vk: poseidon_vk_data.clone(),
            prev_level: Value::unknown(),
            is_leaf: true,
            leaf_agg_vk: (agg_domain.clone(), agg_cs.clone(), Value::unknown()),
            left_state: Value::unknown(),
            right_state: Value::unknown(),
            left_proof: Value::unknown(),
            right_proof: Value::unknown(),
            left_acc: Value::unknown(),
            right_acc: Value::unknown(),
            fixed_base_names: combined_fixed_base_names_keygen.clone(),
        };

        let leaf_agg_vk: VerifyingKey<F, KZGCommitmentScheme<E>> =
            keygen_vk_with_k(&agg_srs, &default_leaf_agg_circuit, K).unwrap();
        let leaf_agg_pk = keygen_pk(leaf_agg_vk.clone(), &default_leaf_agg_circuit).unwrap();

        //
        // 3. Fixed bases and trivial accumulators
        //
        let mut agg_fixed_bases: BTreeMap<String, C> = BTreeMap::new();
        agg_fixed_bases.insert(String::from("com_instance"), C::identity());
        agg_fixed_bases.extend(midnight_circuits::verifier::fixed_bases::<S>(
            "agg_vk", &agg_vk,
        ));

        let mut leaf_agg_fixed_bases: BTreeMap<String, C> = BTreeMap::new();
        leaf_agg_fixed_bases.insert(String::from("com_instance"), C::identity());
        leaf_agg_fixed_bases.extend(midnight_circuits::verifier::fixed_bases::<S>(
            "agg_vk",
            &leaf_agg_vk,
        ));

        let mut poseidon_fixed_bases: BTreeMap<String, C> = BTreeMap::new();
        poseidon_fixed_bases.insert(String::from("com_instance"), C::identity());
        poseidon_fixed_bases.extend(midnight_circuits::verifier::fixed_bases::<S>(
            leaf_vk_name,
            leaf_vk,
        ));

        let poseidon_fixed_base_names = fixed_base_names_for(leaf_vk_name, &poseidon_vk_data.1);
        let leaf_agg_fixed_base_names = fixed_base_names_for("agg_vk", &leaf_agg_vk.cs());
        let agg_fixed_base_names = fixed_base_names_for("agg_vk", &agg_vk.cs());

        let combined_fixed_base_names: Vec<String> = {
            let mut set = BTreeSet::new();
            let mut v = Vec::new();
            for name in poseidon_fixed_base_names
                .iter()
                .chain(leaf_agg_fixed_base_names.iter())
                .chain(agg_fixed_base_names.iter())
            {
                if set.insert(name.clone()) {
                    v.push(name.clone());
                }
            }
            v
        };

        let trivial_poseidon_pi: Accumulator<S> =
            trivial_acc_with_names(&poseidon_fixed_base_names);
        let trivial_leaf_agg: Accumulator<S> = trivial_acc_with_names(&leaf_agg_fixed_base_names);
        let trivial_agg: Accumulator<S> = trivial_acc_with_names(&agg_fixed_base_names);

        let mut trivial_combined =
            Accumulator::accumulate(&[trivial_poseidon_pi, trivial_leaf_agg, trivial_agg]);
        trivial_combined.collapse();

        //
        // 4. vk data for AggCircuit
        //
        let leaf_agg_vk_data = (
            EvaluationDomain::<F>::new(leaf_agg_vk.cs().degree() as u32, K),
            leaf_agg_vk.cs().clone(),
            Value::known(leaf_agg_vk.transcript_repr()),
        );
        let agg_vk_data = (
            EvaluationDomain::<F>::new(agg_vk.cs().degree() as u32, K),
            agg_vk.cs().clone(),
            Value::known(agg_vk.transcript_repr()),
        );

        //
        // 5. Build first level of AGG tree from client proofs (leaf AGG nodes)
        //
        let num_leaves = client_proofs.len();
        let mut current_level: Vec<TreeNode> = (0..num_leaves / 2)
            .map(|i| {
                let left = &client_proofs[i * 2];
                let right = &client_proofs[i * 2 + 1];

                use midnight_circuits::instructions::hash::HashCPU;
                let state = <PoseidonChip<F> as HashCPU<F, F>>::hash(&[left.state, right.state]);

                let circuit = AggCircuit {
                    agg_vk: agg_vk_data.clone(),
                    agg_vk_name: leaf_vk_name,
                    poseidon_vk: poseidon_vk_data.clone(),
                    left_state: Value::known(left.state),
                    right_state: Value::known(right.state),
                    left_proof: Value::known(left.proof.clone()),
                    right_proof: Value::known(right.proof.clone()),
                    left_acc: Value::known(trivial_combined.clone()),
                    right_acc: Value::known(trivial_combined.clone()),
                    fixed_base_names: combined_fixed_base_names.clone(),
                    is_leaf: true,
                    prev_level: Value::known(F::ZERO),
                    leaf_agg_vk: leaf_agg_vk_data.clone(),
                };

                // Verify client proofs and extract accumulators
                let proof_acc_left = verify_and_extract_acc(
                    leaf_srs,
                    leaf_vk,
                    &poseidon_fixed_bases,
                    &left.proof,
                    &[left.state],
                );

                let proof_acc_right = verify_and_extract_acc(
                    leaf_srs,
                    leaf_vk,
                    &poseidon_fixed_bases,
                    &right.proof,
                    &[right.state],
                );

                let mut accumulated_pi = Accumulator::accumulate(&[
                    proof_acc_left.clone(),
                    trivial_combined.clone(),
                    proof_acc_right.clone(),
                    trivial_combined.clone(),
                ]);
                accumulated_pi.collapse();

                let mut public_inputs = AssignedVk::<S>::as_public_input(leaf_vk);
                public_inputs.extend(AssignedNative::<F>::as_public_input(&state));
                public_inputs.extend(AssignedAccumulator::as_public_input(&accumulated_pi));
                public_inputs.extend(AssignedNative::<F>::as_public_input(&F::ONE));

                let proof = {
                    let mut transcript = CircuitTranscript::<PoseidonState<F>>::init();
                    create_proof::<
                        F,
                        KZGCommitmentScheme<E>,
                        CircuitTranscript<PoseidonState<F>>,
                        AggCircuit,
                    >(
                        &agg_srs,
                        &leaf_agg_pk,
                        &[circuit],
                        1,
                        &[&[&[], &public_inputs]],
                        OsRng,
                        &mut transcript,
                    )
                    .expect("Leaf AGG proof failed");
                    transcript.finalize()
                };

                let proof_acc = verify_and_extract_acc(
                    &agg_srs,
                    &leaf_agg_vk,
                    &leaf_agg_fixed_bases,
                    &proof,
                    &public_inputs,
                );

                TreeNode {
                    state,
                    proof,
                    proof_acc,
                    pi_acc: accumulated_pi.clone(),
                }
            })
            .collect();

        //
        // 6. Build upper levels of aggregation tree
        //
        let agg_srs_ref = &agg_srs;

        let mut level = 0;
        while current_level.len() > 1 {
            level += 1;

            let next_level: Vec<TreeNode> = (0..current_level.len() / 2)
                .map(|pair_idx| {
                    let i = pair_idx * 2;
                    let left = current_level[i].clone();
                    let right = current_level[i + 1].clone();

                    use midnight_circuits::instructions::hash::HashCPU;
                    let state =
                        <PoseidonChip<F> as HashCPU<F, F>>::hash(&[left.state, right.state]);

                    let circuit = AggCircuit {
                        agg_vk: agg_vk_data.clone(),
                        agg_vk_name: "agg_vk",
                        poseidon_vk: poseidon_vk_data.clone(),
                        left_state: Value::known(left.state),
                        right_state: Value::known(right.state),
                        left_proof: Value::known(left.proof.clone()),
                        right_proof: Value::known(right.proof.clone()),
                        left_acc: Value::known(left.pi_acc.clone()),
                        right_acc: Value::known(right.pi_acc.clone()),
                        fixed_base_names: combined_fixed_base_names.clone(),
                        is_leaf: false,
                        prev_level: Value::known(F::from(level)),
                        leaf_agg_vk: leaf_agg_vk_data.clone(),
                    };

                    let mut accumulated_pi = Accumulator::accumulate(&[
                        left.proof_acc.clone(),
                        left.pi_acc.clone(),
                        right.proof_acc.clone(),
                        right.pi_acc.clone(),
                    ]);
                    accumulated_pi.collapse();

                    // VK used for public input (leaf_agg at level 1, agg_vk afterwards)
                    let input_agg_vk = if level == 1 { &leaf_agg_vk } else { &agg_vk };

                    let mut public_inputs = AssignedVk::<S>::as_public_input(&input_agg_vk);
                    public_inputs.extend(AssignedNative::<F>::as_public_input(&state));
                    public_inputs.extend(AssignedAccumulator::as_public_input(&accumulated_pi));
                    public_inputs.extend(AssignedNative::<F>::as_public_input(&F::from(level + 1)));

                    println!("about to produce an internal AGG proof at level {}", level);
                    let start = Instant::now();
                    let proof = {
                        let mut transcript = CircuitTranscript::<PoseidonState<F>>::init();
                        create_proof::<
                            F,
                            KZGCommitmentScheme<E>,
                            CircuitTranscript<PoseidonState<F>>,
                            AggCircuit,
                        >(
                            agg_srs_ref,
                            &agg_pk,
                            &[circuit],
                            1,
                            &[&[&[], &public_inputs]],
                            OsRng,
                            &mut transcript,
                        )
                        .expect("Internal AGG proof failed");
                        transcript.finalize()
                    };
                    println!(
                        "Level {} node {} created in {:?}",
                        level,
                        pair_idx,
                        start.elapsed()
                    );

                    let proof_acc = verify_and_extract_acc(
                        agg_srs_ref,
                        &agg_vk,
                        &agg_fixed_bases,
                        &proof,
                        &public_inputs,
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
        }

        //
        // 7. Final root and sanity check
        //
        let root = &current_level[0];

        // Optional: recompute expected root from client states and assert
        let leaf_states: Vec<F> = client_proofs.iter().map(|p| p.state).collect();
        let expected_root = poseidon_tree_root(&leaf_states);
        assert_eq!(
            root.state, expected_root,
            "Root state mismatch with recomputed Poseidon tree root"
        );

        (root.state, root.proof.clone())
    }

    #[allow(dead_code)]
    pub fn demo_poseidon_aggregation() {
        // For this example, we still generate the Poseidon proofs locally,
        // but in a real deployment they would come from clients.
        let poseidon_srs = filecoin_srs_agg(POSEIDON_K);
        let poseidon_relation = PoseidonExample;
        let poseidon_vk = compact_std_lib::setup_vk(&poseidon_srs, &poseidon_relation);
        let poseidon_pk = compact_std_lib::setup_pk(&poseidon_relation, &poseidon_vk);

        let num_leaves = 8;
        println!("Creating {} POSEIDON leaf proofs...", num_leaves);

        let client_proofs: Vec<ClientProof> = (0..num_leaves)
            .map(|i| {
                use midnight_circuits::instructions::hash::HashCPU;
                use rand::SeedableRng;
                use rand_chacha::ChaCha8Rng;

                let mut rng = ChaCha8Rng::seed_from_u64(i as u64);
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
                ClientProof { state, proof }
            })
            .collect();

        let (root_state, agg_proof) = aggregate_client_proofs(
            &poseidon_srs,
            poseidon_vk.vk(),
            "poseidon_vk",
            POSEIDON_K,
            &client_proofs,
        );

        println!("\n=== AGG Tree Complete (via aggregation function) ===");
        println!("Root state: {:?}", root_state);
        println!("Aggregated proof length: {} bytes", agg_proof.len());

        // Optional sanity: recompute from local states.
        let leaf_states: Vec<F> = client_proofs.iter().map(|p| p.state).collect();
        let expected_root = poseidon_tree_root(&leaf_states);
        println!(
            "Expected root (recomputed from POSEIDON states): {:?}",
            expected_root
        );
        assert_eq!(root_state, expected_root, "Root state mismatch!");
        println!("✓ Root verification successful!");
    }
}

// Re-export pieces we need in the shielded example.
use proof_agg::{ClientProof as AggClientProof, aggregate_client_proofs};

// -----------------------------------------------------------------------------
// Original shielded Spend2Output2 code, modified to batch + aggregate online
// -----------------------------------------------------------------------------

const TREE_HEIGHT: usize = 64;
const UTXO_COMMIT_TAG: u64 = 0x0001;
const UTXO_NULLIFY_TAG: u64 = 0x0002;
const AMOUNT_BITS: u32 = 128; // 128-bit integers for amounts
const AMOUNT_GEN_BITS: u32 = 120; // generate up to 120 bits to avoid u128 overflow on sums
const BATCH_SIZE: usize = 8; // must be a power of two

// Merkle path structure
#[derive(Clone, Debug)]
pub struct MerklePath<Fp: PrimeField> {
    pub leaf: Fp,
    pub siblings: [(Fp, bool); TREE_HEIGHT - 1], // bool: true = sibling is on the RIGHT
}

impl<Fp: PoseidonField> MerklePath<Fp> {
    fn compute_root(&self) -> Fp {
        self.siblings
            .iter()
            .fold(self.leaf, |acc, (sib, is_right)| {
                if *is_right {
                    <PoseidonChip<Fp> as HashCPU<Fp, Fp>>::hash(&[acc, *sib, Fp::ZERO])
                } else {
                    <PoseidonChip<Fp> as HashCPU<Fp, Fp>>::hash(&[*sib, acc, Fp::ZERO])
                }
            })
    }
}

// UTXO structure
#[derive(Clone, Debug)]
pub struct Utxo {
    pub asset_id: F,
    pub amount: u128, // 128-bit host-side amount
    pub randomness: F,
}

// -------------------- Simple append-only treestate --------------------

fn hash_pair(a: F, b: F) -> F {
    <PoseidonChip<F> as HashCPU<F, F>>::hash(&[a, b, F::ZERO])
}

fn zero_roots() -> Vec<F> {
    let mut zs = Vec::with_capacity(TREE_HEIGHT);
    zs.push(F::ZERO);
    for _ in 1..TREE_HEIGHT {
        let prev = *zs.last().unwrap();
        zs.push(hash_pair(prev, prev));
    }
    zs
}

#[derive(Clone, Default)]
struct TreeState {
    leaves: Vec<F>,
    nullifiers: HashSet<F>,
}

impl TreeState {
    fn new() -> Self {
        Self::default()
    }

    fn deposit(&mut self, commit: F) -> usize {
        self.leaves.push(commit);
        self.leaves.len() - 1
    }

    fn apply_transfer(&mut self, nfs: [F; 2], new_commits: [F; 2]) -> (usize, usize) {
        for nf in nfs {
            if !self.nullifiers.insert(nf) {
                panic!("nullifier already seen (double spend)");
            }
        }
        let i1 = self.deposit(new_commits[0]);
        let i2 = self.deposit(new_commits[1]);
        (i1, i2)
    }

    fn root(&self) -> F {
        let zs = zero_roots();
        let n = self.leaves.len();
        if n == 0 {
            return zs[TREE_HEIGHT - 1];
        }
        let m = n.next_power_of_two();
        let base_h = m.trailing_zeros() as usize;

        let mut level: Vec<F> = Vec::with_capacity(m);
        level.extend_from_slice(&self.leaves);
        level.resize(m, zs[0]);

        for _ in 0..base_h {
            let mut next = Vec::with_capacity((level.len() + 1) / 2);
            for i in (0..level.len()).step_by(2) {
                next.push(hash_pair(level[i], level[i + 1]));
            }
            level = next;
        }
        let mut acc = level[0];
        let mut h = base_h;
        while h < TREE_HEIGHT - 1 {
            let sib = zs[h];
            acc = hash_pair(acc, sib);
            h += 1;
        }
        acc
    }

    fn merkle_path(&self, mut index: usize) -> MerklePath<F> {
        assert!(index < self.leaves.len(), "index out of range");
        let original = index;
        let zs = zero_roots();
        let n = self.leaves.len();
        let m = n.next_power_of_two();
        let base_h = m.trailing_zeros() as usize;

        let mut level: Vec<F> = Vec::with_capacity(m);
        level.extend_from_slice(&self.leaves);
        level.resize(m, zs[0]);

        let mut sibs: Vec<(F, bool)> = Vec::with_capacity(TREE_HEIGHT - 1);

        for _ in 0..base_h {
            let is_left = (index & 1) == 0;
            let sib = if is_left {
                level[index + 1]
            } else {
                level[index - 1]
            };
            sibs.push((sib, is_left)); // true => sibling is RIGHT
            let mut next = Vec::with_capacity((level.len() + 1) / 2);
            for i in (0..level.len()).step_by(2) {
                next.push(hash_pair(level[i], level[i + 1]));
            }
            level = next;
            index >>= 1;
        }

        let mut h = base_h;
        while sibs.len() < TREE_HEIGHT - 1 {
            sibs.push((zs[h], true)); // our subtree left, zero-subtree right
            h += 1;
        }

        MerklePath {
            leaf: self.leaves[original],
            siblings: sibs.try_into().unwrap(),
        }
    }
}

// -------------------- Circuit relation (single public instance = Poseidon hash) --------------------

#[derive(Clone, Default)]
pub struct Spend2Output2;

impl Relation for Spend2Output2 {
    // Single public input: Poseidon hash of (root, pk'_x, pk'_y, new_c1, new_c2, nf1, nf2)
    // where pk' = pk + [alpha]G is a per-transaction blinded key
    type Instance = F;

    // Witness unchanged except we add the blinding factor alpha
    // (includes everything needed to recompute values that are no longer public)
    type Witness = (
        MerklePath<F>,
        MerklePath<F>,
        JubjubScalar, // sk
        JubjubScalar, // alpha (blinding factor)
        Utxo,
        Utxo,
        Utxo,
        Utxo,
        (F, F), // (pk_out1_x, pk_out1_y)
        (F, F), // (pk_out2_x, pk_out2_y)
    );

    fn format_instance(instance: &Self::Instance) -> Result<Vec<F>, Error> {
        // Expose only the single hash as the public input
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
        let mp1_val = witness.clone().map(|(mp1, _, _, _, _, _, _, _, _, _)| mp1);
        let mp2_val = witness.clone().map(|(_, mp2, _, _, _, _, _, _, _, _)| mp2);
        let sk_val = witness.clone().map(|(_, _, sk, _, _, _, _, _, _, _)| sk);
        let alpha_val = witness
            .clone()
            .map(|(_, _, _, alpha, _, _, _, _, _, _)| alpha);
        let old1_val = witness.clone().map(|(_, _, _, _, o1, _, _, _, _, _)| o1);
        let old2_val = witness.clone().map(|(_, _, _, _, _, o2, _, _, _, _)| o2);
        let new1_val = witness.clone().map(|(_, _, _, _, _, _, n1, _, _, _)| n1);
        let new2_val = witness.clone().map(|(_, _, _, _, _, _, _, n2, _, _)| n2);
        let pk1x_val = witness.clone().map(|(_, _, _, _, _, _, _, _, k1, _)| k1.0);
        let pk1y_val = witness.clone().map(|(_, _, _, _, _, _, _, _, k1, _)| k1.1);
        let pk2x_val = witness.clone().map(|(_, _, _, _, _, _, _, _, _, k2)| k2.0);
        let pk2y_val = witness.clone().map(|(_, _, _, _, _, _, _, _, _, k2)| k2.1);

        // Assign sender secret once, derive sender pk once
        let sk: AssignedScalarOfNativeCurve<Jubjub> = std_lib.jubjub().assign(layouter, sk_val)?;
        let generator = std_lib
            .jubjub()
            .assign_fixed(layouter, JubjubSubgroup::generator())?;
        let pk_sender = std_lib.jubjub().mul(layouter, &sk, &generator)?;
        let pk_sender_fields = std_lib.jubjub().as_public_input(layouter, &pk_sender)?;
        let (pk_sx, pk_sy) = (pk_sender_fields[0].clone(), pk_sender_fields[1].clone());

        // Blinded key: pk' = pk + [alpha]G  (used publicly; authorization proven inside)
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

        // Verify Merkle proofs and check roots match
        let root1 = compute_merkle_root(std_lib, layouter, mp1_val, old_c1.clone())?;
        let root2 = compute_merkle_root(std_lib, layouter, mp2_val, old_c2.clone())?;
        std_lib.assert_equal(layouter, &root1, &root2)?;

        // Nullifiers (BOUND TO UNBLINDED sender pk to prevent double-spends)
        let nf1 = compute_nullifier(std_lib, layouter, &old_c1, &pk_sx, &pk_sy)?;
        let nf2 = compute_nullifier(std_lib, layouter, &old_c2, &pk_sx, &pk_sy)?;

        // New outputs: use provided recipient (pk_out*) coordinates (assigned once)
        let pk1x = std_lib.assign(layouter, pk1x_val)?;
        let pk1y = std_lib.assign(layouter, pk1y_val)?;
        let pk2x = std_lib.assign(layouter, pk2x_val)?;
        let pk2y = std_lib.assign(layouter, pk2y_val)?;

        let new_c1 = compute_commitment_from_parts(std_lib, layouter, &new1_asg, &pk1x, &pk1y)?;
        let new_c2 = compute_commitment_from_parts(std_lib, layouter, &new2_asg, &pk2x, &pk2y)?;

        // Value conservation (same asset id + 128-bit amounts using BigUint gadget)
        check_value_conservation_assigned(
            std_lib, layouter, &old1_asg, &old2_asg, &new1_asg, &new2_asg,
        )?;

        // ---- Single public input: Poseidon hash using BLINDED pk ----
        // Sponge the seven values using the same 3-arity Poseidon as elsewhere:
        // (root, pk'_x, pk'_y) -> acc1
        // (acc1, new_c1, new_c2) -> acc2
        // (acc2, nf1,  nf2)      -> instance_hash
        let acc1 = std_lib.poseidon(layouter, &[root1.clone(), pk_bx.clone(), pk_by.clone()])?;
        let acc2 = std_lib.poseidon(layouter, &[acc1, new_c1.clone(), new_c2.clone()])?;
        let instance_hash = std_lib.poseidon(layouter, &[acc2, nf1.clone(), nf2.clone()])?;

        // Expose only this hash as the single public input
        std_lib.constrain_as_public_input(layouter, &instance_hash)?;
        // -----------------------------------------------------------------

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
            nr_pow2range_cols: 1, // BigUint gadget uses pow2range; 1 column is fine here
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

// A small helper carrying the once-assigned UTXO components used across the circuit.
#[derive(Clone)]
struct AssignedUtxo {
    id: AssignedNative<F>,
    amount_f: AssignedNative<F>,    // amount as a field (for hashing)
    amount_big: AssignedBigUint<F>, // amount as BigUint (for 128-bit arithmetic)
    randomness: AssignedNative<F>,
}

// Assign UTXO fields exactly once (both field & BigUint representations)
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

// Helpers (amounts are already assigned; we never re-assign the same witness)
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

fn compute_merkle_root<L: Layouter<F>>(
    std_lib: &ZkStdLib,
    layouter: &mut L,
    mp_val: Value<MerklePath<F>>,
    leaf: AssignedNative<F>,
) -> Result<AssignedNative<F>, Error> {
    let siblings: Vec<AssignedNative<F>> = std_lib.assign_many(
        layouter,
        mp_val
            .clone()
            .map(|mp| mp.siblings.iter().map(|x| x.0).collect::<Vec<_>>())
            .transpose_vec(TREE_HEIGHT - 1)
            .as_slice(),
    )?;
    let positions = mp_val
        .map(|mp| {
            mp.siblings
                .iter()
                .map(|x| if x.1 { F::ONE } else { F::ZERO })
                .collect::<Vec<_>>()
        })
        .transpose_vec(TREE_HEIGHT - 1);
    let position_bits: Vec<AssignedBit<F>> = std_lib
        .assign_many(layouter, positions.as_slice())?
        .iter()
        .map(|p| std_lib.convert(layouter, p))
        .collect::<Result<_, _>>()?;
    let zero: AssignedNative<F> = std_lib.assign_fixed(layouter, F::ZERO)?;
    siblings
        .iter()
        .zip(position_bits.iter())
        .try_fold(leaf, |acc, (sib, pos)| {
            let left = std_lib.select(layouter, pos, &acc, sib)?;
            let right = std_lib.select(layouter, pos, sib, &acc)?;
            std_lib.poseidon(layouter, &[left, right, zero.clone()])
        })
}

// 128-bit amount conservation and asset-id equality using already-assigned components.
fn check_value_conservation_assigned<L: Layouter<F>>(
    std_lib: &ZkStdLib,
    layouter: &mut L,
    in1: &AssignedUtxo,
    in2: &AssignedUtxo,
    out1: &AssignedUtxo,
    out2: &AssignedUtxo,
) -> Result<(), Error> {
    // All asset IDs equal (no re-assigning)
    std_lib.assert_equal(layouter, &in1.id, &in2.id)?;
    std_lib.assert_equal(layouter, &in1.id, &out1.id)?;
    std_lib.assert_equal(layouter, &in1.id, &out2.id)?;

    // Amount conservation with 128-bit integers (no re-assigning)
    let big = std_lib.biguint();
    let sum_in = big.add(layouter, &in1.amount_big, &in2.amount_big)?;
    let sum_out = big.add(layouter, &out1.amount_big, &out2.amount_big)?;
    big.assert_equal(layouter, &sum_in, &sum_out)
}

// Host-side helpers
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

// Poseidon sponge (3-arity) over the seven public items (old_c* removed):
// (root, pk_x, pk_y)           -> acc1
// (acc1, new_c1, new_c2)       -> acc2
// (acc2, nf1,  nf2)            -> final hash
fn host_instance_hash(items: [F; 7]) -> F {
    let acc1 = <PoseidonChip<F> as HashCPU<F, F>>::hash(&[items[0], items[1], items[2]]);
    let acc2 = <PoseidonChip<F> as HashCPU<F, F>>::hash(&[acc1, items[3], items[4]]);
    <PoseidonChip<F> as HashCPU<F, F>>::hash(&[acc2, items[5], items[6]])
}

// -------------------- Multiple accounts & randomized transfers --------------------

#[derive(Clone, Debug)]
struct Note {
    idx: usize, // index in treestate
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
    const K: u32 = 14;
    const NUM_ACCOUNTS: usize = 4;
    const NUM_SEED_DEPOSITS_PER_ACCOUNT: usize = 3;
    const NUM_TRANSFERS: usize = 120;

    let srs = filecoin_srs(K);
    let relation = Spend2Output2;
    let vk = compact_std_lib::setup_vk(&srs, &relation);
    let pk = compact_std_lib::setup_pk(&relation, &vk);

    let mut rng = ChaCha8Rng::from_entropy();
    let asset_id = F::random(&mut rng); // single asset across all accounts

    // Global treestate
    let mut tree = TreeState::new();

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

    // Seed deposits: random (<=120-bit) amounts, credited to each account
    for acc in &mut accounts {
        for _ in 0..NUM_SEED_DEPOSITS_PER_ACCOUNT {
            // generate <=120-bit to avoid u128 overflow on sums
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
            let idx = tree.deposit(commit);
            acc.wallet.push(Note {
                idx,
                utxo,
                commit,
                spent: false,
            });
        }
    }
    println!("Initial root: {:?}", tree.root());

    // Helper: choose a sender account with >=2 unspent notes
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

        // Start a new batch from the current committed state
        let mut shadow_tree = tree.clone();
        let mut shadow_accounts = accounts.clone();
        let mut client_proofs: Vec<AggClientProof> = Vec::new();

        println!(
            "\n=== Starting batch {} from root {:?} ===",
            batch_idx,
            shadow_tree.root()
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

            // Pick two distinct unspent notes from sender (on shadow state)
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

            // Choose two (possibly equal) recipients at random
            let r1 = rng.gen_range(0..NUM_ACCOUNTS);
            let r2 = rng.gen_range(0..NUM_ACCOUNTS);

            // Sender & inputs (from shadow state)
            let sender = shadow_accounts[sender_idx].clone();
            let old1 = shadow_accounts[sender_idx].wallet[i_old1].clone();
            let old2 = shadow_accounts[sender_idx].wallet[i_old2].clone();

            // Build membership proofs against current *shadow* root
            let root_before = shadow_tree.root();
            let mp1 = shadow_tree.merkle_path(old1.idx);
            let mp2 = shadow_tree.merkle_path(old2.idx);
            assert_eq!(root_before, mp1.compute_root());
            assert_eq!(root_before, mp2.compute_root());

            // Random split to recipients: out1 in [0..=total]
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

            // Nullifiers (bound to UNBLINDED sender key to maintain uniqueness)
            let nf1 = host_nullify(old1.commit, sender.pk_x, sender.pk_y);
            let nf2 = host_nullify(old2.commit, sender.pk_x, sender.pk_y);

            // Per-transaction blinding factor and blinded key pk' = pk + [alpha]G
            let alpha = JubjubScalar::random(&mut OsRng);
            let blind_point = JubjubSubgroup::generator() * alpha;
            let pk_blinded_point = sender.pk_point + blind_point;
            let pkb_fields = AssignedNativePoint::<Jubjub>::as_public_input(&pk_blinded_point);
            let pk_bx = pkb_fields[0];
            let pk_by = pkb_fields[1];

            // Compute single public instance hash (Poseidon sponge without old commitments) using BLINDED pk
            let instance: F = host_instance_hash([
                root_before,
                pk_bx,
                pk_by,
                new1_commit,
                new2_commit,
                nf1,
                nf2,
            ]);

            // Witness carries alpha and recipient keys for outputs (unchanged)
            let witness = (
                mp1,
                mp2,
                sender.sk,
                alpha, // blinding factor
                old1.utxo.clone(),
                old2.utxo.clone(),
                new1.clone(),
                new2.clone(),
                (shadow_accounts[r1].pk_x, shadow_accounts[r1].pk_y),
                (shadow_accounts[r2].pk_x, shadow_accounts[r2].pk_y),
            );

            // Prove (per-transfer client proof). We do not rely on per-proof
            // verification here; aggregation will re-verify all client proofs.
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

            // Collect client proof for aggregation
            client_proofs.push(AggClientProof {
                state: instance,
                proof: proof.clone(),
            });

            // Apply to shadow tree (pending state)
            let (idx_new1, idx_new2) =
                shadow_tree.apply_transfer([nf1, nf2], [new1_commit, new2_commit]);

            // Mark inputs spent and credit recipients in shadow accounts
            shadow_accounts[sender_idx].wallet[i_old1].spent = true;
            shadow_accounts[sender_idx].wallet[i_old2].spent = true;

            shadow_accounts[r1].wallet.push(Note {
                idx: idx_new1,
                utxo: new1,
                commit: new1_commit,
                spent: false,
            });
            shadow_accounts[r2].wallet.push(Note {
                idx: idx_new2,
                utxo: new2,
                commit: new2_commit,
                spent: false,
            });

            // quick inclusion checks on shadow state
            let root_after = shadow_tree.root();
            let mp_out1 = shadow_tree.merkle_path(idx_new1);
            let mp_out2 = shadow_tree.merkle_path(idx_new2);
            assert_eq!(root_after, mp_out1.compute_root());
            assert_eq!(root_after, mp_out2.compute_root());

            println!(
                "[batch {}, tx {}] shadow root updated: {:?}",
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

        // Aggregate this batch of shielded proofs into a single proof, verifying all
        // client proofs and the aggregation circuit.
        let now = Instant::now();
        let (agg_state, agg_proof) = aggregate_client_proofs(
            &srs,
            vk.vk(),
            "poseidon_vk", // label used only for fixed-base bookkeeping
            K,
            &client_proofs,
        );
        println!(
            "Batch {} aggregated proof generated and internally verified in {:?}",
            batch_idx,
            now.elapsed()
        );
        println!(
            "Batch {} aggregated state: {:?}, agg proof length: {} bytes",
            batch_idx,
            agg_state,
            agg_proof.len()
        );

        // Commit shadow state only after successful aggregation/verification
        tree = shadow_tree;
        accounts = shadow_accounts;

        println!(
            "After batch {} committed root: {:?}",
            batch_idx,
            tree.root()
        );

        batch_idx += 1;
    }

    println!("\nFinal root: {:?}", tree.root());

    // (Optional) show balances per account (sum of unspent amounts)
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
