use core::array;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Instant;

use ff::Field;
use group::Group;
use thiserror::Error;

use midnight_circuits::verifier::{Accumulator, BlstrsEmulation, SelfEmulation};
use midnight_curves::Bls12;
use midnight_proofs::circuit::Value;
use midnight_proofs::plonk::{keygen_pk, keygen_vk_with_k};
use midnight_proofs::poly::kzg::params::ParamsKZG;
use midnight_proofs::{
    plonk::{Circuit, ConstraintSystem, ProvingKey, VerifyingKey},
    poly::{EvaluationDomain, kzg::KZGCommitmentScheme},
};

use crate::rollup_ivc_circuits::VkData;
use crate::{rollup_ivc_circuits, trusted_setup};

// Type aliases
pub type S = BlstrsEmulation;
type F = <S as SelfEmulation>::F;
type C = <S as SelfEmulation>::C;
type E = <S as SelfEmulation>::Engine;
type Vk = VerifyingKey<F, KZGCommitmentScheme<E>>;
type Pk = ProvingKey<F, KZGCommitmentScheme<E>>;

// Constants
const K_LEAF: u32 = 19;
const K_INTERNAL: u32 = 19;
const MIN_NUM_LEAVES: usize = 4;

/// Errors that can occur during aggregation setup preparation.
#[derive(Debug, Error)]
pub enum AggSetupError {
    #[error("invalid num_leaves={num_leaves}: {reason}")]
    InvalidNumLeaves {
        num_leaves: usize,
        reason: &'static str,
    },

    #[error("failed to load aggregation SRS for k={k}")]
    SrsLoadFailed { k: u32 },

    #[error("SRS mismatch: {details}")]
    SrsMismatch { details: &'static str },

    #[error("key generation failed for {component}")]
    KeygenFailed { component: &'static str },
}

/// A single aggregation level's keys and metadata.
///
/// Level numbering is 1-indexed: leaf aggregation is level 1.
#[derive(Clone)]
pub(crate) struct AggLevelKeys {
    pub(crate) level: usize,
    pub(crate) name: String,
    pub(crate) vk: Arc<Vk>,
    pub(crate) pk: Arc<Pk>,
    pub(crate) vk_data: VkData,
    pub(crate) fixed_bases: BTreeMap<String, C>,
}

impl AggLevelKeys {
    fn new(level: usize, name: String, vk: Vk, pk: Pk) -> Self {
        let k = k_for_level(level);
        let vk_data = VkData {
            domain: EvaluationDomain::new(vk.cs().degree() as u32, k),
            cs: vk.cs().clone(),
            transcript_repr: vk.transcript_repr(),
        };
        let fixed_bases = compute_fixed_bases_for_vk(&name, &vk);

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

/// Store for all aggregation level keys with validation.
#[derive(Clone)]
pub(crate) struct AggKeyStore {
    levels: Vec<AggLevelKeys>,
}

impl AggKeyStore {
    fn new(levels: Vec<AggLevelKeys>) -> Self {
        assert!(!levels.is_empty(), "AggKeyStore cannot be empty");
        Self::validate_levels(&levels);
        Self { levels }
    }

    fn validate_levels(levels: &[AggLevelKeys]) {
        let mut seen_names = BTreeSet::new();
        for (i, lvl) in levels.iter().enumerate() {
            let expected_level = i + 1;
            assert_eq!(
                lvl.level, expected_level,
                "AggKeyStore level mismatch at index {i}: expected {expected_level}, got {}",
                lvl.level
            );
            assert!(
                seen_names.insert(lvl.name.clone()),
                "Duplicate vk_name: '{}'",
                lvl.name
            );
        }
    }

    fn max_level(&self) -> usize {
        self.levels.len()
    }

    pub(crate) fn get(&self, level: usize) -> &AggLevelKeys {
        assert!(
            (1..=self.levels.len()).contains(&level),
            "Agg level {level} out of range [1, {}]",
            self.levels.len()
        );
        &self.levels[level - 1]
    }
}

/// Cached aggregation keys and supporting data.
///
/// Compute once at program start, then reuse for every batch.
#[derive(Clone)]
pub struct AggSetup {
    // Configuration
    pub(crate) leaf_vk_name: String,
    pub(crate) num_leaves: usize,
    pub(crate) max_agg_level: usize,

    // Derived data
    pub(crate) leaf_vk_data: VkData,
    pub(crate) agg_srs_leaf: ParamsKZG<Bls12>,
    pub(crate) agg_srs_internal: ParamsKZG<Bls12>,
    pub(crate) agg_store: AggKeyStore,
    pub(crate) leaf_fixed_bases: BTreeMap<String, C>,
    pub(crate) fixed_base_names: Vec<String>,
    pub(crate) fixed_bases: BTreeMap<String, C>,
    pub(crate) trivial_combined: Accumulator<S>,
    pub(crate) child_vk: (EvaluationDomain<F>, ConstraintSystem<F>, F),
    pub(crate) child_vk_name: String,
}

impl AggSetup {
    #[must_use]
    pub fn child_vk(&self) -> VkData {
        VkData {
            domain: self.child_vk.0.clone(),
            cs: self.child_vk.1.clone(),
            transcript_repr: self.child_vk.2,
        }
    }

    #[must_use]
    pub fn child_vk_name(&self) -> &str {
        &self.child_vk_name
    }

    #[must_use]
    pub fn fixed_base_names(&self) -> &[String] {
        &self.fixed_base_names
    }
}

/// Prepare and cache all aggregation keys for a fixed batch size.
///
/// # Panics
/// Panics if setup fails (preserves original API behavior).
#[allow(unused)]
pub fn prepare_agg_setup(
    leaf_srs: &ParamsKZG<Bls12>,
    leaf_vk: &Vk,
    leaf_vk_name: &str,
    leaf_k: u32,
    num_leaves: usize,
) -> AggSetup {
    prepare_agg_setup_impl(leaf_srs, leaf_vk, leaf_vk_name, leaf_k, num_leaves)
        .unwrap_or_else(|e| panic!("prepare_agg_setup failed: {e}"))
}

fn prepare_agg_setup_impl(
    leaf_srs: &ParamsKZG<Bls12>,
    leaf_vk: &Vk,
    leaf_vk_name: &str,
    leaf_k: u32,
    num_leaves: usize,
) -> Result<AggSetup, AggSetupError> {
    // Validate inputs
    validate_num_leaves(num_leaves)?;
    let max_level = compute_max_level(num_leaves);
    validate_min_levels(num_leaves)?;

    // Setup basic data
    let max_agg_level = max_level - 1;
    let leaf_vk_data = create_vk_data(leaf_vk, leaf_k);
    let agg_cs = create_agg_constraint_system();

    // Load and verify SRS
    let (agg_srs_leaf, agg_srs_internal) = load_and_verify_srs(leaf_srs)?;

    // Compute names
    let agg_vk_names = generate_agg_vk_names(max_agg_level);
    let fixed_base_names =
        compute_all_fixed_base_names(leaf_vk_name, &leaf_vk_data.cs, &agg_vk_names, &agg_cs);

    // Build aggregation levels
    let agg_levels = build_agg_levels(
        max_agg_level,
        &leaf_vk_data,
        leaf_vk_name,
        &agg_vk_names,
        &fixed_base_names,
        &agg_srs_leaf,
        &agg_srs_internal,
    )?;
    let agg_store = AggKeyStore::new(agg_levels);

    // Compute fixed bases
    let leaf_fixed_bases = compute_fixed_bases_for_vk(leaf_vk_name, leaf_vk);
    let fixed_bases = merge_all_fixed_bases(&leaf_fixed_bases, &agg_store);

    // Build trivial accumulator
    let trivial_combined = build_trivial_combined(leaf_vk_name, &leaf_vk_data.cs, &agg_store);

    // Derive child VK
    let (child_vk, child_vk_name) = derive_child_vk(&agg_store, max_agg_level);

    Ok(AggSetup {
        leaf_vk_name: leaf_vk_name.to_string(),
        num_leaves,
        max_agg_level,
        leaf_vk_data,
        agg_srs_leaf,
        agg_srs_internal,
        agg_store,
        leaf_fixed_bases,
        fixed_base_names,
        fixed_bases,
        trivial_combined,
        child_vk,
        child_vk_name,
    })
}

// Validation functions

fn validate_num_leaves(num_leaves: usize) -> Result<(), AggSetupError> {
    if num_leaves == 0 {
        return Err(AggSetupError::InvalidNumLeaves {
            num_leaves,
            reason: "need at least one client proof",
        });
    }
    if !num_leaves.is_power_of_two() {
        return Err(AggSetupError::InvalidNumLeaves {
            num_leaves,
            reason: "client proofs must be power of two",
        });
    }
    Ok(())
}

fn validate_min_levels(num_leaves: usize) -> Result<(), AggSetupError> {
    if num_leaves < MIN_NUM_LEAVES {
        return Err(AggSetupError::InvalidNumLeaves {
            num_leaves,
            reason: "merged final agg requires at least 4 client proofs",
        });
    }
    Ok(())
}

// Computation functions

fn compute_max_level(num_leaves: usize) -> usize {
    (num_leaves as u32).trailing_zeros() as usize
}

fn k_for_level(level: usize) -> u32 {
    if level == 1 { K_LEAF } else { K_INTERNAL }
}

fn create_vk_data(vk: &Vk, k: u32) -> VkData {
    VkData {
        domain: EvaluationDomain::new(vk.cs().degree() as u32, k),
        cs: vk.cs().clone(),
        transcript_repr: vk.transcript_repr(),
    }
}

fn create_agg_constraint_system() -> ConstraintSystem<F> {
    let mut cs = ConstraintSystem::default();
    rollup_ivc_circuits::configure_agg_circuit(&mut cs);
    cs
}

// SRS functions

fn load_and_verify_srs(
    leaf_srs: &ParamsKZG<Bls12>,
) -> Result<(ParamsKZG<Bls12>, ParamsKZG<Bls12>), AggSetupError> {
    let agg_srs_leaf = load_srs(K_LEAF)?;
    let agg_srs_internal = load_srs(K_INTERNAL)?;
    verify_srs_compatibility(leaf_srs, &agg_srs_leaf, &agg_srs_internal)?;
    Ok((agg_srs_leaf, agg_srs_internal))
}

fn load_srs(k: u32) -> Result<ParamsKZG<Bls12>, AggSetupError> {
    trusted_setup::filecoin_srs_agg(k).map_err(|_| AggSetupError::SrsLoadFailed { k })
}

fn verify_srs_compatibility(
    leaf_srs: &ParamsKZG<Bls12>,
    agg_srs_leaf: &ParamsKZG<Bls12>,
    agg_srs_internal: &ParamsKZG<Bls12>,
) -> Result<(), AggSetupError> {
    if leaf_srs.s_g2() != agg_srs_internal.s_g2() {
        return Err(AggSetupError::SrsMismatch {
            details: "leaf_srs.s_g2 != agg_srs_internal.s_g2",
        });
    }
    if agg_srs_leaf.s_g2() != agg_srs_internal.s_g2() {
        return Err(AggSetupError::SrsMismatch {
            details: "agg_srs_leaf.s_g2 != agg_srs_internal.s_g2",
        });
    }
    Ok(())
}

// Fixed base functions

fn compute_fixed_base_names_for_vk(vk_name: &str, cs: &ConstraintSystem<F>) -> Vec<String> {
    let mut names = vec!["com_instance".to_string(), "~G".to_string()];
    names.extend(midnight_circuits::verifier::fixed_base_names::<S>(
        vk_name,
        cs.num_fixed_columns() + cs.num_selectors(),
        cs.permutation().columns.len(),
    ));
    names
}

fn compute_fixed_bases_for_vk(vk_name: &str, vk: &Vk) -> BTreeMap<String, C> {
    let mut fixed_bases = BTreeMap::new();
    fixed_bases.insert("com_instance".to_string(), C::identity());
    fixed_bases.extend(midnight_circuits::verifier::fixed_bases::<S>(vk_name, vk));
    fixed_bases
}

fn generate_agg_vk_names(max_agg_level: usize) -> Vec<String> {
    (1..=max_agg_level)
        .map(|level| format!("agg_vk_lvl{level}"))
        .collect()
}

fn compute_all_fixed_base_names(
    leaf_vk_name: &str,
    leaf_cs: &ConstraintSystem<F>,
    agg_vk_names: &[String],
    agg_cs: &ConstraintSystem<F>,
) -> Vec<String> {
    let leaf_names = compute_fixed_base_names_for_vk(leaf_vk_name, leaf_cs);
    let agg_names: Vec<_> = agg_vk_names
        .iter()
        .flat_map(|name| compute_fixed_base_names_for_vk(name, agg_cs))
        .collect();

    deduplicate_preserve_order(leaf_names.into_iter().chain(agg_names))
}

fn deduplicate_preserve_order<I>(iter: I) -> Vec<String>
where
    I: Iterator<Item = String>,
{
    let mut seen = BTreeSet::new();
    iter.filter(|name| seen.insert(name.clone())).collect()
}

fn merge_all_fixed_bases(
    leaf_fixed_bases: &BTreeMap<String, C>,
    agg_store: &AggKeyStore,
) -> BTreeMap<String, C> {
    let mut fixed_bases = leaf_fixed_bases.clone();
    for level in 1..=agg_store.max_level() {
        fixed_bases.extend(agg_store.get(level).fixed_bases.clone());
    }
    fixed_bases
}

// Accumulator functions

fn create_trivial_accumulator(names: &[String]) -> Accumulator<S> {
    use midnight_circuits::verifier::Msm;

    let fixed: BTreeMap<String, F> = names.iter().map(|n| (n.clone(), F::ZERO)).collect();
    Accumulator::<S>::new(
        Msm::new(&[C::default()], &[F::ONE], &BTreeMap::new()),
        Msm::new(&[C::default()], &[F::ONE], &fixed),
    )
}

fn build_trivial_combined(
    leaf_vk_name: &str,
    leaf_cs: &ConstraintSystem<F>,
    agg_store: &AggKeyStore,
) -> Accumulator<S> {
    let leaf_names = compute_fixed_base_names_for_vk(leaf_vk_name, leaf_cs);
    let trivial_leaf = create_trivial_accumulator(&leaf_names);

    let all_trivials: Vec<_> = std::iter::once(trivial_leaf)
        .chain((1..=agg_store.max_level()).map(|level| {
            let lvl = agg_store.get(level);
            let names = compute_fixed_base_names_for_vk(&lvl.name, lvl.vk.cs());
            create_trivial_accumulator(&names)
        }))
        .collect();

    let mut combined = Accumulator::accumulate(&all_trivials);
    combined.collapse();
    combined
}

// Key generation functions

fn keygen_vk_pk<Circ: Circuit<F>>(
    srs: &ParamsKZG<Bls12>,
    circuit: &Circ,
    k: u32,
) -> Result<(Vk, Pk), AggSetupError> {
    let vk = keygen_vk_with_k(srs, circuit, k).map_err(|_| AggSetupError::KeygenFailed {
        component: "keygen_vk_with_k",
    })?;
    let pk = keygen_pk(vk.clone(), circuit).map_err(|_| AggSetupError::KeygenFailed {
        component: "keygen_pk",
    })?;
    Ok((vk, pk))
}

fn build_agg_levels(
    max_agg_level: usize,
    leaf_vk_data: &VkData,
    leaf_vk_name: &str,
    agg_vk_names: &[String],
    fixed_base_names: &[String],
    agg_srs_leaf: &ParamsKZG<Bls12>,
    agg_srs_internal: &ParamsKZG<Bls12>,
) -> Result<Vec<AggLevelKeys>, AggSetupError> {
    let mut levels = Vec::with_capacity(max_agg_level);

    for level in 1..=max_agg_level {
        let (child_vk, child_vk_name) =
            determine_child_for_level(level, leaf_vk_data, leaf_vk_name, agg_vk_names, &levels);

        let name = agg_vk_names[level - 1].clone();

        let start = Instant::now();
        let (vk, pk) = generate_keys_for_level(
            level,
            child_vk,
            child_vk_name,
            fixed_base_names,
            agg_srs_leaf,
            agg_srs_internal,
        )?;
        println!("Computed {name} vk/pk in {:?}", start.elapsed());

        levels.push(AggLevelKeys::new(level, name, vk, pk));
    }

    Ok(levels)
}

fn determine_child_for_level(
    level: usize,
    leaf_vk_data: &VkData,
    leaf_vk_name: &str,
    agg_vk_names: &[String],
    built_levels: &[AggLevelKeys],
) -> (VkData, String) {
    if level == 1 {
        (leaf_vk_data.clone(), leaf_vk_name.to_string())
    } else {
        let child_level = level - 1;
        let child = &built_levels[child_level - 1];
        (child.vk_data.clone(), agg_vk_names[child_level - 1].clone())
    }
}

fn generate_keys_for_level(
    level: usize,
    child_vk: VkData,
    child_vk_name: String,
    fixed_base_names: &[String],
    agg_srs_leaf: &ParamsKZG<Bls12>,
    agg_srs_internal: &ParamsKZG<Bls12>,
) -> Result<(Vk, Pk), AggSetupError> {
    if level == 1 {
        generate_leaf_keys(child_vk, child_vk_name, fixed_base_names, agg_srs_leaf)
    } else {
        generate_internal_keys(child_vk, child_vk_name, fixed_base_names, agg_srs_internal)
    }
}

fn generate_leaf_keys(
    child_vk: VkData,
    child_vk_name: String,
    fixed_base_names: &[String],
    agg_srs_leaf: &ParamsKZG<Bls12>,
) -> Result<(Vk, Pk), AggSetupError> {
    let circuit = rollup_ivc_circuits::LeafAggCircuit {
        child_vk,
        child_vk_name,
        left_items: Value::unknown(),
        right_items: Value::unknown(),
        pre_commitment_map: Value::unknown(),
        pre_nullifier_map: Value::unknown(),
        pre_commitment_roots_set_map: Value::unknown(),
        left_proof: Value::unknown(),
        right_proof: Value::unknown(),
        left_pi_acc: Value::unknown(),
        right_pi_acc: Value::unknown(),
        fixed_base_names: fixed_base_names.to_vec(),
    };
    keygen_vk_pk(agg_srs_leaf, &circuit, K_LEAF)
}

fn generate_internal_keys(
    child_vk: VkData,
    child_vk_name: String,
    fixed_base_names: &[String],
    agg_srs_internal: &ParamsKZG<Bls12>,
) -> Result<(Vk, Pk), AggSetupError> {
    let circuit = rollup_ivc_circuits::InternalAggCircuit {
        child_vk,
        child_vk_name,
        left_child_state: array::from_fn(|_| Value::unknown()),
        right_child_state: array::from_fn(|_| Value::unknown()),
        left_proof: Value::unknown(),
        right_proof: Value::unknown(),
        left_pi_acc: Value::unknown(),
        right_pi_acc: Value::unknown(),
        fixed_base_names: fixed_base_names.to_vec(),
    };
    keygen_vk_pk(agg_srs_internal, &circuit, K_INTERNAL)
}

fn derive_child_vk(
    agg_store: &AggKeyStore,
    max_agg_level: usize,
) -> ((EvaluationDomain<F>, ConstraintSystem<F>, F), String) {
    let child_keys = agg_store.get(max_agg_level);
    let child_vk = (
        child_keys.vk_data.domain.clone(),
        child_keys.vk_data.cs.clone(),
        child_keys.vk_data.transcript_repr,
    );
    (child_vk, child_keys.name.clone())
}
