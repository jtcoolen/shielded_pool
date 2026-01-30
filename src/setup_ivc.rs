use core::array;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;
use std::time::Instant;

use ff::Field;
use group::Group;

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

pub type S = BlstrsEmulation;
type F = <S as SelfEmulation>::F;
type C = <S as SelfEmulation>::C;
type E = <S as SelfEmulation>::Engine;

const K_LEAF: u32 = 19;
const K_INTERNAL: u32 = 19;

type Vk = VerifyingKey<F, KZGCommitmentScheme<E>>;
type Pk = ProvingKey<F, KZGCommitmentScheme<E>>;

/// Errors during aggregation setup preparation.
///
/// Note: `prepare_agg_setup` keeps the original signature and will still panic on error
/// (to preserve existing API/behavior), but the internal implementation is fallible and
/// produces these errors for clearer diagnostics and easier testing.
#[derive(Debug)]
enum AggSetupError {
    InvalidNumLeaves {
        num_leaves: usize,
        reason: &'static str,
    },
    SrsLoadFailed {
        k: u32,
    },
    SrsMismatch {
        which: &'static str,
    },
    KeygenFailed {
        which: &'static str,
    },
}

impl fmt::Display for AggSetupError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AggSetupError::InvalidNumLeaves { num_leaves, reason } => {
                write!(f, "invalid num_leaves={num_leaves}: {reason}")
            }
            AggSetupError::SrsLoadFailed { k } => write!(f, "failed to load agg SRS for k={k}"),
            AggSetupError::SrsMismatch { which } => write!(f, "SRS mismatch: {which}"),
            AggSetupError::KeygenFailed { which } => write!(f, "key generation failed: {which}"),
        }
    }
}

/// Compute the fixed-base *names* used by the verifier gadget for a given verifying-key name.
///
/// The returned list includes:
/// - `com_instance` and `~G` (convention used by the verifier),
/// - plus the circuit-dependent fixed-base names produced by `midnight_circuits`.
fn fixed_base_names_for(vk_name: &str, cs: &ConstraintSystem<F>) -> Vec<String> {
    let mut names = vec![String::from("com_instance"), String::from("~G")];
    names.extend(midnight_circuits::verifier::fixed_base_names::<S>(
        vk_name,
        cs.num_fixed_columns() + cs.num_selectors(),
        cs.permutation().columns.len(),
    ));
    names
}

/// Build a trivial accumulator whose fixed-base scalars are all `0`.
///
/// This is useful for preparing a combined accumulator with the right *shape* (names/keys)
/// without depending on real proof material.
fn trivial_acc_with_names(names: &[String]) -> Accumulator<S> {
    use midnight_circuits::verifier::Msm;

    let fixed: BTreeMap<String, F> = names.iter().cloned().map(|n| (n, F::ZERO)).collect();
    Accumulator::<S>::new(
        Msm::new(&[C::default()], &[F::ONE], &BTreeMap::new()),
        Msm::new(&[C::default()], &[F::ONE], &fixed),
    )
}

fn agg_vk_name_for_level(level: usize) -> String {
    format!("agg_vk_lvl{level}")
}

/// A single aggregation level's keys plus helper data.
///
/// - `level` is 1-indexed: leaf aggregation is level 1.
/// - `name` is the verifier-key name used for fixed-base lookup.
/// - `vk_data` carries circuit metadata used in subsequent levels.
#[derive(Clone)]
pub(crate) struct AggLevelKeys {
    pub(crate) level: usize,
    pub(crate) name: String,
    pub(crate) vk: Arc<Vk>,
    pub(crate) pk: Arc<Pk>,
    pub(crate) vk_data: rollup_ivc_circuits::VkData,
    pub(crate) fixed_bases: BTreeMap<String, C>,
}

impl AggLevelKeys {
    fn new(level: usize, name: String, vk: Vk, pk: Pk) -> Self {
        let k = k_for_level(level);

        let vk_data = rollup_ivc_circuits::VkData {
            domain: EvaluationDomain::new(vk.cs().degree() as u32, k),
            cs: vk.cs().clone(),
            transcript_repr: vk.transcript_repr(),
        };

        let fixed_bases = fixed_bases_for_vk_name(&name, &vk);

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

/// A small store of all aggregation level keys.
///
/// Invariants:
/// - non-empty
/// - levels are contiguous and start at 1
/// - names are unique
#[derive(Clone)]
pub(crate) struct AggKeyStore {
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

    pub(crate) fn get(&self, level: usize) -> &AggLevelKeys {
        assert!(
            level >= 1 && level <= self.levels.len(),
            "Agg level out of range"
        );
        &self.levels[level - 1]
    }
}

/// Keygen helper: build (vk, pk) for a circuit with explicit `k`.
fn keygen_vk_pk<Circ: Circuit<F>>(
    srs: &ParamsKZG<Bls12>,
    circuit: &Circ,
    k: u32,
) -> Result<(Vk, Pk), AggSetupError> {
    let vk = keygen_vk_with_k(srs, circuit, k).map_err(|_| AggSetupError::KeygenFailed {
        which: "keygen_vk_with_k",
    })?;
    let pk = keygen_pk(vk.clone(), circuit)
        .map_err(|_| AggSetupError::KeygenFailed { which: "keygen_pk" })?;
    Ok((vk, pk))
}

/// Cached aggregation keys & supporting data. Compute once (e.g., at program start),
/// then reuse for every batch.
#[derive(Clone)]
pub struct AggSetup {
    // Inputs
    pub(crate) leaf_vk_name: String,
    pub(crate) num_leaves: usize,
    pub(crate) max_agg_level: usize,

    // Derived/cached
    pub(crate) leaf_vk_data: rollup_ivc_circuits::VkData,

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

/// Prepare and cache all aggregation keys once, for a fixed `num_leaves` (batch size).
///
/// This function preserves the original behavior and will panic on invalid inputs / failures.
/// For testability and robustness, the internal implementation is fallible and returns `AggSetupError`.
#[allow(unused)]
pub fn prepare_agg_setup(
    leaf_srs: &ParamsKZG<Bls12>,
    leaf_vk: &VerifyingKey<F, KZGCommitmentScheme<E>>,
    leaf_vk_name: &str,
    leaf_k: u32,
    num_leaves: usize,
) -> AggSetup {
    prepare_agg_setup_impl(leaf_srs, leaf_vk, leaf_vk_name, leaf_k, num_leaves)
        .unwrap_or_else(|e| panic!("prepare_agg_setup failed: {e}"))
}

/* ----------------------------- Pure-ish helpers ----------------------------- */

fn prepare_agg_setup_impl(
    leaf_srs: &ParamsKZG<Bls12>,
    leaf_vk: &VerifyingKey<F, KZGCommitmentScheme<E>>,
    leaf_vk_name: &str,
    leaf_k: u32,
    num_leaves: usize,
) -> Result<AggSetup, AggSetupError> {
    validate_num_leaves(num_leaves)?;

    let max_level = max_level_from_num_leaves(num_leaves);
    ensure_min_levels(max_level)?;

    let max_agg_level = max_level - 1;
    let leaf_vk_data = vk_data_from_leaf_vk(leaf_vk, leaf_k);

    let agg_cs = agg_constraint_system();
    let (agg_srs_leaf, agg_srs_internal) = load_agg_srs_pair(K_LEAF, K_INTERNAL)?;

    verify_srs_compatibility(leaf_srs, &agg_srs_leaf, &agg_srs_internal)?;

    let agg_vk_names = agg_vk_names(max_agg_level);
    let fixed_base_names =
        compute_fixed_base_names(leaf_vk_name, &leaf_vk_data.cs, &agg_vk_names, &agg_cs);

    let agg_levels = build_agg_levels(
        max_agg_level,
        leaf_vk_data.clone(),
        leaf_vk_name,
        &agg_vk_names,
        fixed_base_names.clone(),
        &agg_srs_leaf,
        &agg_srs_internal,
    )?;
    let agg_store = AggKeyStore::new(agg_levels);

    let leaf_fixed_bases = fixed_bases_for_leaf_vk(leaf_vk_name, leaf_vk);
    let fixed_bases = merge_all_fixed_bases(&leaf_fixed_bases, &agg_store);

    let trivial_combined = build_trivial_combined(leaf_vk_name, &leaf_vk_data.cs, &agg_store);

    let (child_vk, child_vk_name) = derive_child_vk(&agg_store, max_agg_level);

    Ok(AggSetup {
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
        child_vk_name,
    })
}

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

fn max_level_from_num_leaves(num_leaves: usize) -> usize {
    // For num_leaves = 2^n, trailing_zeros is n.
    (num_leaves as u32).trailing_zeros() as usize
}

fn ensure_min_levels(max_level: usize) -> Result<(), AggSetupError> {
    // Original behavior required >= 4 client proofs (i.e., max_level >= 2).
    if max_level < 2 {
        return Err(AggSetupError::InvalidNumLeaves {
            num_leaves: 1usize << max_level,
            reason: "merged final agg requires at least 4 client proofs",
        });
    }
    Ok(())
}

fn vk_data_from_leaf_vk(
    leaf_vk: &VerifyingKey<F, KZGCommitmentScheme<E>>,
    leaf_k: u32,
) -> rollup_ivc_circuits::VkData {
    rollup_ivc_circuits::VkData {
        domain: EvaluationDomain::new(leaf_vk.cs().degree() as u32, leaf_k),
        cs: leaf_vk.cs().clone(),
        transcript_repr: leaf_vk.transcript_repr(),
    }
}

fn agg_constraint_system() -> ConstraintSystem<F> {
    let mut cs = ConstraintSystem::default();
    rollup_ivc_circuits::configure_agg_circuit(&mut cs);
    cs
}

fn load_agg_srs_pair(
    k_leaf: u32,
    k_internal: u32,
) -> Result<(ParamsKZG<Bls12>, ParamsKZG<Bls12>), AggSetupError> {
    let agg_srs_leaf = trusted_setup::filecoin_srs_agg(k_leaf)
        .map_err(|_| AggSetupError::SrsLoadFailed { k: k_leaf })?;
    let agg_srs_internal = trusted_setup::filecoin_srs_agg(k_internal)
        .map_err(|_| AggSetupError::SrsLoadFailed { k: k_internal })?;
    Ok((agg_srs_leaf, agg_srs_internal))
}

fn verify_srs_compatibility(
    leaf_srs: &ParamsKZG<Bls12>,
    agg_srs_leaf: &ParamsKZG<Bls12>,
    agg_srs_internal: &ParamsKZG<Bls12>,
) -> Result<(), AggSetupError> {
    if leaf_srs.s_g2() != agg_srs_internal.s_g2() {
        return Err(AggSetupError::SrsMismatch {
            which: "leaf_srs.s_g2 != agg_srs_internal.s_g2",
        });
    }
    if agg_srs_leaf.s_g2() != agg_srs_internal.s_g2() {
        return Err(AggSetupError::SrsMismatch {
            which: "agg_srs_leaf.s_g2 != agg_srs_internal.s_g2",
        });
    }
    Ok(())
}

fn agg_vk_names(max_agg_level: usize) -> Vec<String> {
    (1..=max_agg_level).map(agg_vk_name_for_level).collect()
}

/// Preserve insertion order while deduplicating.
fn unique_preserve_order<I>(iter: I) -> Vec<String>
where
    I: IntoIterator<Item = String>,
{
    let mut seen = BTreeSet::new();
    iter.into_iter()
        .filter(|name| seen.insert(name.clone()))
        .collect()
}

fn compute_fixed_base_names(
    leaf_vk_name: &str,
    leaf_cs: &ConstraintSystem<F>,
    agg_vk_names: &[String],
    agg_cs: &ConstraintSystem<F>,
) -> Vec<String> {
    let leaf_names = fixed_base_names_for(leaf_vk_name, leaf_cs).into_iter();
    let agg_names = agg_vk_names
        .iter()
        .flat_map(|vk_name| fixed_base_names_for(vk_name.as_str(), agg_cs).into_iter());

    unique_preserve_order(leaf_names.chain(agg_names))
}

fn k_for_level(level: usize) -> u32 {
    if level == 1 { K_LEAF } else { K_INTERNAL }
}

fn fixed_bases_for_vk_name(vk_name: &str, vk: &Vk) -> BTreeMap<String, C> {
    let mut fixed_bases = BTreeMap::new();
    fixed_bases.insert(String::from("com_instance"), C::identity());
    fixed_bases.extend(midnight_circuits::verifier::fixed_bases::<S>(vk_name, vk));
    fixed_bases
}

fn fixed_bases_for_leaf_vk(leaf_vk_name: &str, leaf_vk: &Vk) -> BTreeMap<String, C> {
    fixed_bases_for_vk_name(leaf_vk_name, leaf_vk)
}

/// Build keys for all aggregation levels.
///
/// This is written in a functional style using `try_fold`: each new level can depend on
/// the previous level's `vk_data` and name.
fn build_agg_levels(
    max_agg_level: usize,
    leaf_vk_data: rollup_ivc_circuits::VkData,
    leaf_vk_name: &str,
    agg_vk_names: &[String],
    fixed_base_names: Vec<String>,
    agg_srs_leaf: &ParamsKZG<Bls12>,
    agg_srs_internal: &ParamsKZG<Bls12>,
) -> Result<Vec<AggLevelKeys>, AggSetupError> {
    (1..=max_agg_level).try_fold(Vec::with_capacity(max_agg_level), |mut acc, level| {
        let (child_vk, child_vk_name) =
            child_for_level(level, &leaf_vk_data, leaf_vk_name, agg_vk_names, &acc);

        let name = agg_vk_names
            .get(level - 1)
            .cloned()
            .expect("agg_vk_names length matches max_agg_level");

        let start = Instant::now();
        let (vk, pk) = keygen_for_level(
            level,
            child_vk,
            child_vk_name,
            fixed_base_names.clone(),
            agg_srs_leaf,
            agg_srs_internal,
        )?;
        println!("Computed {} vk/pk in {:?}", name, start.elapsed());

        acc.push(AggLevelKeys::new(level, name, vk, pk));
        Ok(acc)
    })
}

/// Determine the child vk-data/name for a given aggregation level:
/// - level 1 aggregates client proofs (leaf vk)
/// - levels >=2 aggregate previous aggregation proofs
fn child_for_level(
    level: usize,
    leaf_vk_data: &rollup_ivc_circuits::VkData,
    leaf_vk_name: &str,
    agg_vk_names: &[String],
    built_levels: &[AggLevelKeys],
) -> (rollup_ivc_circuits::VkData, String) {
    if level == 1 {
        (leaf_vk_data.clone(), leaf_vk_name.to_string())
    } else {
        let child_level = level - 1;
        let child = built_levels[child_level - 1].vk_data.clone();
        let child_name = agg_vk_names[child_level - 1].clone();
        (child, child_name)
    }
}

/// Keygen for either leaf or internal aggregation circuits.
fn keygen_for_level(
    level: usize,
    child_vk: rollup_ivc_circuits::VkData,
    child_vk_name: String,
    fixed_base_names: Vec<String>,
    agg_srs_leaf: &ParamsKZG<Bls12>,
    agg_srs_internal: &ParamsKZG<Bls12>,
) -> Result<(Vk, Pk), AggSetupError> {
    if level == 1 {
        // Leaf aggregation layer: BaseStepCircuit<K_LEAF>
        let default_circuit = rollup_ivc_circuits::LeafAggCircuit {
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

            fixed_base_names,
        };

        keygen_vk_pk(agg_srs_leaf, &default_circuit, K_LEAF)
    } else {
        // Internal aggregation layers: FoldStepCircuit<K_INTERNAL>
        let default_circuit = rollup_ivc_circuits::InternalAggCircuit {
            child_vk,
            child_vk_name,

            left_child_state: array::from_fn(|_| Value::unknown()),
            right_child_state: array::from_fn(|_| Value::unknown()),

            left_proof: Value::unknown(),
            right_proof: Value::unknown(),
            left_pi_acc: Value::unknown(),
            right_pi_acc: Value::unknown(),

            fixed_base_names,
        };

        keygen_vk_pk(agg_srs_internal, &default_circuit, K_INTERNAL)
    }
}

fn merge_all_fixed_bases(
    leaf_fixed_bases: &BTreeMap<String, C>,
    agg_store: &AggKeyStore,
) -> BTreeMap<String, C> {
    let mut fixed_bases = BTreeMap::new();

    fixed_bases.extend(leaf_fixed_bases.iter().map(|(k, v)| (k.clone(), *v)));
    (1..=agg_store.max_level()).for_each(|level| {
        fixed_bases.extend(
            agg_store
                .get(level)
                .fixed_bases
                .iter()
                .map(|(k, v)| (k.clone(), *v)),
        );
    });

    fixed_bases
}

/// Build a collapsed accumulator that covers:
/// - the leaf verifier fixed bases, and
/// - every aggregation verifier key fixed bases.
fn build_trivial_combined(
    leaf_vk_name: &str,
    leaf_cs: &ConstraintSystem<F>,
    agg_store: &AggKeyStore,
) -> Accumulator<S> {
    let trivial_leaf = trivial_acc_with_names(&fixed_base_names_for(leaf_vk_name, leaf_cs));

    let trivials = std::iter::once(trivial_leaf).chain((1..=agg_store.max_level()).map(|level| {
        let lvl = agg_store.get(level);
        trivial_acc_with_names(&fixed_base_names_for(lvl.name.as_str(), lvl.vk.cs()))
    }));

    let mut combined = Accumulator::accumulate(&trivials.collect::<Vec<_>>());
    combined.collapse();
    combined
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
