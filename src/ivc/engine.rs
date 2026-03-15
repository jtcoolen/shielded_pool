//! IVC engine: setup (key generation) and prover (tree construction).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Instant;

use ff::Field;
use group::Group;
use rand::rngs::OsRng;
use rayon::prelude::*;
use thiserror::Error;

use midnight_circuits::{
    hash::poseidon::PoseidonState,
    instructions::hash::HashCPU,
    verifier::{Accumulator, AssignedAccumulator, Msm},
};
use midnight_curves::Bls12;
use midnight_proofs::{
    circuit::Value,
    plonk::{Circuit, ConstraintSystem, ProvingKey, VerifyingKey, create_proof, keygen_pk, keygen_vk_with_k},
    poly::{EvaluationDomain, kzg::{KZGCommitmentScheme, params::ParamsKZG}},
    transcript::CircuitTranscript,
};

use midnight_circuits::hash::poseidon::PoseidonChip;

use super::{
    Acc, C, E, F, FoldStep,
    LeafStep, NodeState, S, TreeNode, TreeResult, VkData,
    circuit::{FrameworkWitness, IvcDeciderCircuit, IvcLeafCircuit, IvcNodeCircuit},
    ctx::configure_ivc_circuit,
};

////////////////////////////////////////////////////////////////////////////////
// Errors
////////////////////////////////////////////////////////////////////////////////

#[derive(Debug, Error)]
pub enum AggregationError {
    #[error("need at least one client proof")]
    Empty,

    #[error("client proofs length must be a power of two (got {0})")]
    NotPowerOfTwo(usize),

    #[error("length mismatch: expected {expected}, got {got}")]
    LenMismatch { expected: usize, got: usize },

    #[error("need at least 4 client proofs for merged final agg (got {0})")]
    NeedAtLeastFour(usize),

    #[error("host-side leaf validation failed: {0}")]
    LeafValidation(String),

    #[error("host-side fold validation failed: {0}")]
    FoldValidation(String),

    #[error("commitment already exists in map")]
    CommitmentAlreadyExists,

    #[error("nullifier already spent")]
    NullifierAlreadySpent,

    #[error("leaf proof creation failed")]
    LeafProofFailed,

    #[error("internal proof creation failed")]
    InternalProofFailed,

    #[error("verification prepare failed")]
    PrepareFailed,

    #[error("dual MSM check failed")]
    DualMsmFailed,

    #[error("accumulator check failed")]
    AccumulatorFailed,

    #[error("PI accumulator check failed: {0}")]
    PiAccFailed(&'static str),

    #[error("Merkle root mismatch")]
    MerkleRootMismatch,

    #[error("setup: {0}")]
    Setup(String),
}

////////////////////////////////////////////////////////////////////////////////
// AggLevelKeys — keys for one level of the aggregation tree
////////////////////////////////////////////////////////////////////////////////

type Vk = VerifyingKey<F, KZGCommitmentScheme<E>>;
type Pk = ProvingKey<F, KZGCommitmentScheme<E>>;

#[derive(Clone)]
pub(crate) struct AggLevelKeys {
    pub level: usize,
    pub name: String,
    pub vk: Arc<Vk>,
    pub pk: Arc<Pk>,
    pub vk_data: VkData,
    pub fixed_bases: BTreeMap<String, C>,
}

impl AggLevelKeys {
    fn new(level: usize, name: String, vk: Vk, pk: Pk, k: u32) -> Self {
        let vk_data = VkData {
            domain: EvaluationDomain::new(vk.cs().degree() as u32, k),
            cs: vk.cs().clone(),
            transcript_repr: vk.transcript_repr(),
        };
        let fixed_bases = compute_fixed_bases_for_vk(&name, &vk);
        Self { level, name, vk: Arc::new(vk), pk: Arc::new(pk), vk_data, fixed_bases }
    }
}

#[derive(Clone)]
pub(crate) struct AggKeyStore {
    levels: Vec<AggLevelKeys>,
}

impl AggKeyStore {
    fn new(levels: Vec<AggLevelKeys>) -> Self {
        assert!(!levels.is_empty());
        Self { levels }
    }

    fn max_level(&self) -> usize {
        self.levels.len()
    }

    pub fn get(&self, level: usize) -> &AggLevelKeys {
        assert!((1..=self.levels.len()).contains(&level), "level {level} out of range");
        &self.levels[level - 1]
    }
}

////////////////////////////////////////////////////////////////////////////////
// IvcSetup — cached keys for all aggregation levels
////////////////////////////////////////////////////////////////////////////////

#[derive(Clone)]
pub struct IvcSetup {
    pub(crate) leaf_vk_name: String,
    pub(crate) num_leaves: usize,
    pub(crate) max_agg_level: usize,
    pub(crate) app_state_width: usize,
    pub(crate) client_pi_width: usize,
    pub(crate) leaf_vk_data: VkData,
    pub(crate) agg_srs_leaf: ParamsKZG<Bls12>,
    pub(crate) agg_srs_internal: ParamsKZG<Bls12>,
    pub(crate) agg_store: AggKeyStore,
    pub(crate) leaf_fixed_bases: BTreeMap<String, C>,
    pub(crate) fixed_base_names: Vec<String>,
    pub fixed_bases: BTreeMap<String, C>,
    pub(crate) trivial_combined: Acc,
}

impl IvcSetup {
    pub fn child_vk(&self) -> VkData {
        let top = self.agg_store.get(self.max_agg_level);
        top.vk_data.clone()
    }

    pub fn child_vk_name(&self) -> &str {
        &self.agg_store.get(self.max_agg_level).name
    }

    pub fn fixed_base_names(&self) -> &[String] {
        &self.fixed_base_names
    }
}

////////////////////////////////////////////////////////////////////////////////
// Setup construction
////////////////////////////////////////////////////////////////////////////////

pub fn prepare_ivc_setup<L, Fo>(
    leaf_step: &L,
    fold_step: &Fo,
    leaf_srs: &ParamsKZG<Bls12>,
    leaf_vk: &Vk,
    leaf_vk_name: &str,
    leaf_k: u32,
    k_leaf_agg: u32,
    k_internal: u32,
    num_leaves: usize,
    app_state_width: usize,
    client_pi_width: usize,
) -> Result<IvcSetup, AggregationError>
where
    L: LeafStep,
    Fo: FoldStep,
{
    if num_leaves == 0 || !num_leaves.is_power_of_two() || num_leaves < 4 {
        return Err(AggregationError::NeedAtLeastFour(num_leaves));
    }

    let max_level = (num_leaves as u32).trailing_zeros() as usize;
    let max_agg_level = max_level - 1;

    let leaf_vk_data = VkData {
        domain: EvaluationDomain::new(leaf_vk.cs().degree() as u32, leaf_k),
        cs: leaf_vk.cs().clone(),
        transcript_repr: leaf_vk.transcript_repr(),
    };

    let agg_cs = {
        let mut cs = ConstraintSystem::default();
        configure_ivc_circuit(&mut cs);
        cs
    };

    let agg_srs_leaf = load_srs(k_leaf_agg)?;
    let agg_srs_internal = load_srs(k_internal)?;

    let agg_vk_names: Vec<String> = (1..=max_agg_level)
        .map(|l| format!("agg_vk_lvl{l}"))
        .collect();

    let fixed_base_names = compute_all_fixed_base_names(
        leaf_vk_name, &leaf_vk_data.cs, &agg_vk_names, &agg_cs,
    );

    let mut levels = Vec::with_capacity(max_agg_level);

    for level in 1..=max_agg_level {
        let (child_vk_data, child_vk_name) = if level == 1 {
            (leaf_vk_data.clone(), leaf_vk_name.to_string())
        } else {
            let prev = &levels[level - 2];
            (prev.vk_data.clone(), prev.name.clone())
        };

        let name = agg_vk_names[level - 1].clone();
        let k = if level == 1 { k_leaf_agg } else { k_internal };
        let srs = if level == 1 { &agg_srs_leaf } else { &agg_srs_internal };

        let start = Instant::now();
        let (vk, pk) = if level == 1 {
            let circuit = IvcLeafCircuit::<L, 19> {
                step: leaf_step.clone(),
                left_client_items: Value::unknown(),
                right_client_items: Value::unknown(),
                witness: Value::unknown(),
                client_pi_width,
                fw: FrameworkWitness {
                    child_vk: child_vk_data,
                    child_vk_name,
                    left_proof: Value::unknown(),
                    right_proof: Value::unknown(),
                    left_pi_acc: Value::unknown(),
                    right_pi_acc: Value::unknown(),
                    fixed_base_names: fixed_base_names.clone(),
                },
            };
            keygen_pair(srs, &circuit, k)?
        } else {
            let full_width = app_state_width + 1;
            let circuit = IvcNodeCircuit::<Fo, 19> {
                step: fold_step.clone(),
                app_state_width,
                left_child_state: vec![Value::unknown(); full_width],
                right_child_state: vec![Value::unknown(); full_width],
                fw: FrameworkWitness {
                    child_vk: child_vk_data,
                    child_vk_name,
                    left_proof: Value::unknown(),
                    right_proof: Value::unknown(),
                    left_pi_acc: Value::unknown(),
                    right_pi_acc: Value::unknown(),
                    fixed_base_names: fixed_base_names.clone(),
                },
            };
            keygen_pair(srs, &circuit, k)?
        };

        println!("Computed {name} vk/pk in {:?}", start.elapsed());
        levels.push(AggLevelKeys::new(level, name, vk, pk, k));
    }

    let agg_store = AggKeyStore::new(levels);
    let leaf_fixed_bases = compute_fixed_bases_for_vk(leaf_vk_name, leaf_vk);
    let fixed_bases = merge_all_fixed_bases(&leaf_fixed_bases, &agg_store);
    let trivial_combined = build_trivial_combined(leaf_vk_name, &leaf_vk_data.cs, &agg_store);

    Ok(IvcSetup {
        leaf_vk_name: leaf_vk_name.to_string(),
        num_leaves,
        max_agg_level,
        app_state_width,
        client_pi_width,
        leaf_vk_data,
        agg_srs_leaf,
        agg_srs_internal,
        agg_store,
        leaf_fixed_bases,
        fixed_base_names,
        fixed_bases,
        trivial_combined,
    })
}

////////////////////////////////////////////////////////////////////////////////
// IvcProver — parallel tree construction
////////////////////////////////////////////////////////////////////////////////

pub struct IvcProver;

/// Plan for one leaf pair, produced by the host-side planning layer.
pub struct LeafPlan<W> {
    pub index: usize,
    pub left: ClientProof,
    pub right: ClientProof,
    pub app_state: Vec<F>,
    pub merkle_digest: F,
    pub witness: W,
}

impl IvcProver {
    /// Prove the full binary tree from `2^d` client proofs.
    ///
    /// Returns the top two tree nodes (ready for the decider).
    /// `host_merge` computes the parent app state from two children
    /// on the host side (must match FoldStep circuit logic).
    pub fn prove_tree<L, Fo>(
        setup: &IvcSetup,
        client_srs: &ParamsKZG<Bls12>,
        client_vk: &Vk,
        leaf_step: &L,
        fold_step: &Fo,
        leaf_plans: Vec<LeafPlan<L::Witness>>,
        host_merge: impl Fn(&[F], &[F]) -> Vec<F> + Send + Sync,
    ) -> Result<TreeResult<Vec<F>>, AggregationError>
    where
        L: LeafStep + 'static,
        Fo: FoldStep + 'static,
    {
        if leaf_plans.len() != setup.num_leaves / 2 {
            return Err(AggregationError::LenMismatch {
                expected: setup.num_leaves / 2,
                got: leaf_plans.len(),
            });
        }

        // 1. Prove leaves in parallel
        let leaf_nodes: Vec<TreeNode<Vec<F>>> = leaf_plans
            .into_par_iter()
            .map(|plan| prove_leaf(setup, client_srs, client_vk, leaf_step, plan))
            .collect::<Result<Vec<_>, _>>()?;

        // 2. Build internal levels up to the top pair
        let (child_level, top_pair) =
            build_to_top_pair(setup, fold_step, &host_merge, 1, leaf_nodes)?;

        let (left, right) = top_pair;

        // 3. Compute the root state
        let root_digest = host_hash_pair(left.merkle_digest, right.merkle_digest);
        let root_app = host_merge(&left.app_state, &right.app_state);
        let root_state = NodeState {
            app_state: root_app,
            merkle_digest: root_digest,
        };

        Ok(TreeResult { left_top: left, right_top: right, root_state })
    }
}

////////////////////////////////////////////////////////////////////////////////
// Internal: leaf proving
////////////////////////////////////////////////////////////////////////////////

fn prove_leaf<L: LeafStep>(
    setup: &IvcSetup,
    client_srs: &ParamsKZG<Bls12>,
    client_vk: &Vk,
    leaf_step: &L,
    plan: LeafPlan<L::Witness>,
) -> Result<TreeNode<Vec<F>>, AggregationError> {
    let leaf_keys = setup.agg_store.get(1);
    let srs = &setup.agg_srs_leaf;

    let circuit = IvcLeafCircuit::<L, 19> {
        step: leaf_step.clone(),
        left_client_items: Value::known(plan.left.public_inputs.clone()),
        right_client_items: Value::known(plan.right.public_inputs.clone()),
        witness: Value::known(plan.witness),
        client_pi_width: setup.client_pi_width,
        fw: FrameworkWitness {
            child_vk: setup.leaf_vk_data.clone(),
            child_vk_name: setup.leaf_vk_name.clone(),
            left_proof: Value::known(plan.left.proof.clone()),
            right_proof: Value::known(plan.right.proof.clone()),
            left_pi_acc: Value::known(setup.trivial_combined.clone()),
            right_pi_acc: Value::known(setup.trivial_combined.clone()),
            fixed_base_names: setup.fixed_base_names.clone(),
        },
    };

    // Verify client proofs and extract accumulators
    let (acc_l, acc_r) = rayon::join(
        || verify_and_extract(client_srs, client_vk, &setup.leaf_fixed_bases, &plan.left.proof, &plan.left.public_inputs),
        || verify_and_extract(client_srs, client_vk, &setup.leaf_fixed_bases, &plan.right.proof, &plan.right.public_inputs),
    );
    let acc_l = acc_l?;
    let acc_r = acc_r?;

    let pi_acc = collapse(Accumulator::accumulate(&[
        acc_l, setup.trivial_combined.clone(),
        acc_r, setup.trivial_combined.clone(),
    ]));

    let mut full_state = plan.app_state.clone();
    full_state.push(plan.merkle_digest);
    let pi_fields: Vec<F> = full_state
        .iter()
        .copied()
        .chain(AssignedAccumulator::as_public_input(&pi_acc))
        .collect();

    let start = Instant::now();
    let proof = create_agg_proof(srs, leaf_keys.pk.as_ref(), circuit, &pi_fields)?;
    println!("Leaf AGG {} in {:?}", plan.index, start.elapsed());

    let proof_acc = verify_and_extract(
        srs, leaf_keys.vk.as_ref(), &leaf_keys.fixed_bases, &proof, &pi_fields,
    )?;

    Ok(TreeNode {
        app_state: plan.app_state,
        merkle_digest: plan.merkle_digest,
        proof,
        proof_acc,
        pi_acc,
    })
}

////////////////////////////////////////////////////////////////////////////////
// Internal: node proving
////////////////////////////////////////////////////////////////////////////////

fn prove_node<Fo: FoldStep>(
    setup: &IvcSetup,
    fold_step: &Fo,
    host_merge: &(impl Fn(&[F], &[F]) -> Vec<F> + Send + Sync),
    parent_level: usize,
    child_level: usize,
    left: &TreeNode<Vec<F>>,
    right: &TreeNode<Vec<F>>,
) -> Result<TreeNode<Vec<F>>, AggregationError> {
    let child_keys = setup.agg_store.get(child_level);
    let parent_keys = setup.agg_store.get(parent_level);
    let srs = &setup.agg_srs_internal;

    let app_state = host_merge(&left.app_state, &right.app_state);
    let digest = host_hash_pair(left.merkle_digest, right.merkle_digest);

    let mut left_full: Vec<F> = left.app_state.clone();
    left_full.push(left.merkle_digest);
    let mut right_full: Vec<F> = right.app_state.clone();
    right_full.push(right.merkle_digest);

    let circuit = IvcNodeCircuit::<Fo, 19> {
        step: fold_step.clone(),
        app_state_width: setup.app_state_width,
        left_child_state: left_full.iter().map(|f| Value::known(*f)).collect(),
        right_child_state: right_full.iter().map(|f| Value::known(*f)).collect(),
        fw: FrameworkWitness {
            child_vk: child_keys.vk_data.clone(),
            child_vk_name: child_keys.name.clone(),
            left_proof: Value::known(left.proof.clone()),
            right_proof: Value::known(right.proof.clone()),
            left_pi_acc: Value::known(left.pi_acc.clone()),
            right_pi_acc: Value::known(right.pi_acc.clone()),
            fixed_base_names: setup.fixed_base_names.clone(),
        },
    };

    let pi_acc = collapse(Accumulator::accumulate(&[
        left.proof_acc.clone(), left.pi_acc.clone(),
        right.proof_acc.clone(), right.pi_acc.clone(),
    ]));

    let mut full_state = app_state.clone();
    full_state.push(digest);
    let pi_fields: Vec<F> = full_state
        .iter()
        .copied()
        .chain(AssignedAccumulator::as_public_input(&pi_acc))
        .collect();

    let start = Instant::now();
    let proof = create_agg_proof(srs, parent_keys.pk.as_ref(), circuit, &pi_fields)?;
    println!("Internal level {} in {:?}", parent_level, start.elapsed());

    let proof_acc = verify_and_extract(
        srs, parent_keys.vk.as_ref(), &parent_keys.fixed_bases, &proof, &pi_fields,
    )?;

    Ok(TreeNode { app_state, merkle_digest: digest, proof, proof_acc, pi_acc })
}

////////////////////////////////////////////////////////////////////////////////
// Internal: tree construction
////////////////////////////////////////////////////////////////////////////////

fn build_to_top_pair<Fo: FoldStep>(
    setup: &IvcSetup,
    fold_step: &Fo,
    host_merge: &(impl Fn(&[F], &[F]) -> Vec<F> + Send + Sync),
    child_level: usize,
    mut nodes: Vec<TreeNode<Vec<F>>>,
) -> Result<(usize, (TreeNode<Vec<F>>, TreeNode<Vec<F>>)), AggregationError> {
    let mut level = child_level;
    loop {
        match nodes.len() {
            0 => return Err(AggregationError::Empty),
            1 => return Err(AggregationError::NeedAtLeastFour(2)),
            2 => {
                let right = nodes.pop().unwrap();
                let left = nodes.pop().unwrap();
                return Ok((level, (left, right)));
            }
            _ => {
                let parent_level = level + 1;
                let pairs: Vec<_> = nodes.chunks_exact(2).collect();
                nodes = pairs
                    .par_iter()
                    .map(|pair| prove_node(setup, fold_step, host_merge, parent_level, level, &pair[0], &pair[1]))
                    .collect::<Result<Vec<_>, _>>()?;
                level = parent_level;
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Crypto helpers
////////////////////////////////////////////////////////////////////////////////

fn host_hash_pair(a: F, b: F) -> F {
    <PoseidonChip<F> as HashCPU<F, F>>::hash(&[a, b])
}

pub fn host_instance_hash(items: &[F]) -> F {
    <PoseidonChip<F> as HashCPU<F, F>>::hash(items)
}

fn collapse(mut acc: Acc) -> Acc {
    acc.collapse();
    acc
}

fn verify_and_extract(
    srs: &ParamsKZG<Bls12>,
    vk: &Vk,
    fixed_bases: &BTreeMap<String, C>,
    proof: &[u8],
    public_inputs: &[F],
) -> Result<Acc, AggregationError> {
    let mut transcript = CircuitTranscript::<PoseidonState<F>>::init_from_bytes(proof);
    let committed: &[&[C]] = &[&[C::identity()]];
    let instances: &[&[&[F]]] = &[&[public_inputs]];

    let dual_msm = midnight_proofs::plonk::prepare::<
        F, KZGCommitmentScheme<E>, CircuitTranscript<PoseidonState<F>>,
    >(vk, committed, instances, &mut transcript)
        .map_err(|_| AggregationError::PrepareFailed)?;

    if !dual_msm.clone().check(&srs.verifier_params()) {
        return Err(AggregationError::DualMsmFailed);
    }

    let mut acc: Acc = dual_msm.into();
    acc.extract_fixed_bases(fixed_bases);
    acc.collapse();

    if !acc.check(&srs.s_g2().into(), fixed_bases) {
        return Err(AggregationError::AccumulatorFailed);
    }

    Ok(acc)
}

fn create_agg_proof<Circ: Circuit<F>>(
    srs: &ParamsKZG<Bls12>,
    pk: &Pk,
    circuit: Circ,
    public_inputs: &[F],
) -> Result<Vec<u8>, AggregationError> {
    let mut transcript = CircuitTranscript::<PoseidonState<F>>::init();
    create_proof::<F, KZGCommitmentScheme<E>, CircuitTranscript<PoseidonState<F>>, Circ>(
        srs, pk, &[circuit], 1, &[&[&[], public_inputs]], OsRng, &mut transcript,
    )
    .map_err(|_| AggregationError::InternalProofFailed)?;
    Ok(transcript.finalize())
}

fn keygen_pair<Circ: Circuit<F>>(
    srs: &ParamsKZG<Bls12>,
    circuit: &Circ,
    k: u32,
) -> Result<(Vk, Pk), AggregationError> {
    let vk = keygen_vk_with_k(srs, circuit, k)
        .map_err(|e| AggregationError::Setup(format!("vk: {e}")))?;
    let pk = keygen_pk(vk.clone(), circuit)
        .map_err(|e| AggregationError::Setup(format!("pk: {e}")))?;
    Ok((vk, pk))
}

fn load_srs(k: u32) -> Result<ParamsKZG<Bls12>, AggregationError> {
    crate::trusted_setup::filecoin_srs_agg(k)
        .map_err(|e| AggregationError::Setup(format!("SRS k={k}: {e}")))
}

////////////////////////////////////////////////////////////////////////////////
// Fixed base helpers
////////////////////////////////////////////////////////////////////////////////

fn compute_fixed_base_names_for_vk(name: &str, cs: &ConstraintSystem<F>) -> Vec<String> {
    let mut names = vec!["com_instance".to_string(), "~G".to_string()];
    names.extend(midnight_circuits::verifier::fixed_base_names::<S>(
        name,
        cs.num_fixed_columns() + cs.num_selectors(),
        cs.permutation().columns.len(),
    ));
    names
}

fn compute_fixed_bases_for_vk(name: &str, vk: &Vk) -> BTreeMap<String, C> {
    let mut fb = BTreeMap::new();
    fb.insert("com_instance".to_string(), C::identity());
    fb.extend(midnight_circuits::verifier::fixed_bases::<S>(name, vk));
    fb
}

fn compute_all_fixed_base_names(
    leaf_vk_name: &str,
    leaf_cs: &ConstraintSystem<F>,
    agg_vk_names: &[String],
    agg_cs: &ConstraintSystem<F>,
) -> Vec<String> {
    let mut seen = BTreeSet::new();
    let leaf_names = compute_fixed_base_names_for_vk(leaf_vk_name, leaf_cs);
    let agg_names: Vec<_> = agg_vk_names
        .iter()
        .flat_map(|n| compute_fixed_base_names_for_vk(n, agg_cs))
        .collect();
    leaf_names
        .into_iter()
        .chain(agg_names)
        .filter(|n| seen.insert(n.clone()))
        .collect()
}

fn merge_all_fixed_bases(
    leaf_fb: &BTreeMap<String, C>,
    store: &AggKeyStore,
) -> BTreeMap<String, C> {
    let mut fb = leaf_fb.clone();
    for l in 1..=store.max_level() {
        fb.extend(store.get(l).fixed_bases.clone());
    }
    fb
}

fn build_trivial_combined(
    leaf_vk_name: &str,
    leaf_cs: &ConstraintSystem<F>,
    store: &AggKeyStore,
) -> Acc {
    fn trivial(names: &[String]) -> Acc {
        let fixed: BTreeMap<String, F> = names.iter().map(|n| (n.clone(), F::ZERO)).collect();
        Accumulator::<S>::new(
            Msm::new(&[C::default()], &[F::ONE], &BTreeMap::new()),
            Msm::new(&[C::default()], &[F::ONE], &fixed),
        )
    }

    let leaf_names = compute_fixed_base_names_for_vk(leaf_vk_name, leaf_cs);
    let all: Vec<_> = std::iter::once(trivial(&leaf_names))
        .chain((1..=store.max_level()).map(|l| {
            let lvl = store.get(l);
            trivial(&compute_fixed_base_names_for_vk(&lvl.name, lvl.vk.cs()))
        }))
        .collect();

    let mut combined = Accumulator::accumulate(&all);
    combined.collapse();
    combined
}
