use std::time::Instant;

use ff::Field;
use group::Group;

use midnight_proofs::circuit::Value;
use midnight_proofs::plonk::{keygen_pk, keygen_vk_with_k};

use core::array;

use midnight_circuits::verifier::{Accumulator, BlstrsEmulation, SelfEmulation};
use midnight_curves::Bls12;
use midnight_proofs::poly::kzg::params::ParamsKZG;
use midnight_proofs::{
    plonk::{Circuit, ConstraintSystem, ProvingKey, VerifyingKey},
    poly::{EvaluationDomain, kzg::KZGCommitmentScheme},
};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use crate::{rollup_ivc, srs};

pub type S = BlstrsEmulation;
type F = <S as SelfEmulation>::F;
type C = <S as SelfEmulation>::C;
type E = <S as SelfEmulation>::Engine;

const K_LEAF: u32 = 19;
const K_INTERNAL: u32 = 19;

type Vk = VerifyingKey<F, KZGCommitmentScheme<E>>;
type Pk = ProvingKey<F, KZGCommitmentScheme<E>>;

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

fn agg_vk_name_for_level(level: usize) -> String {
    format!("agg_vk_lvl{level}")
}

#[derive(Clone)]
pub(crate) struct AggLevelKeys {
    pub(crate) level: usize,
    pub(crate) name: String,
    pub(crate) vk: Arc<Vk>,
    pub(crate) pk: Arc<Pk>,
    pub(crate) vk_data: rollup_ivc::VkData,
    pub(crate) fixed_bases: BTreeMap<String, C>,
}

impl AggLevelKeys {
    fn new(level: usize, name: String, vk: Vk, pk: Pk) -> Self {
        let k = if level == 1 { K_LEAF } else { K_INTERNAL };
        let vk_data = rollup_ivc::VkData {
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
    pub(crate) leaf_vk_name: String,
    pub(crate) num_leaves: usize,
    pub(crate) max_agg_level: usize,

    // Derived/cached
    pub(crate) leaf_vk_data: rollup_ivc::VkData,

    pub(crate) agg_srs_leaf: ParamsKZG<Bls12>,
    pub(crate) agg_srs_internal: ParamsKZG<Bls12>,
    pub(crate) agg_store: AggKeyStore,

    pub(crate) leaf_fixed_bases: BTreeMap<String, C>,
    pub(crate) fixed_base_names: Vec<String>,
    pub(crate) fixed_bases: BTreeMap<String, C>,
    pub(crate) trivial_combined: Accumulator<S>,

    pub(crate) child_vk: (EvaluationDomain<F>, ConstraintSystem<F>, F),
    pub(crate) child_vk_name: String,
    pub(crate) child_level: usize,
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
#[allow(unused)]
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

    let leaf_vk_data = rollup_ivc::VkData {
        domain: EvaluationDomain::new(leaf_vk.cs().degree() as u32, leaf_k),
        cs: leaf_vk.cs().clone(),
        transcript_repr: leaf_vk.transcript_repr(),
    };

    let mut agg_cs = ConstraintSystem::default();
    rollup_ivc::configure_agg_circuit(&mut agg_cs);

    let agg_srs_leaf = srs::filecoin_srs_agg(K_LEAF).unwrap();
    let agg_srs_internal = srs::filecoin_srs_agg(K_INTERNAL).unwrap();

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
            let default_circuit = rollup_ivc::LeafAggCircuit {
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
            let default_circuit = rollup_ivc::InternalAggCircuit {
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
