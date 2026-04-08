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
    types::Instantiable,
    verifier::{Accumulator, AssignedAccumulator, Msm},
};
use midnight_curves::Bls12;
use midnight_proofs::{
    circuit::Value,
    dev::MockProver,
    plonk::{
        Circuit, ConstraintSystem, ProvingKey, VerifyingKey, create_proof, keygen_pk,
        keygen_vk_with_k,
    },
    poly::{
        EvaluationDomain,
        kzg::{KZGCommitmentScheme, params::ParamsKZG},
    },
    transcript::{CircuitTranscript, Transcript},
};

use midnight_circuits::hash::poseidon::PoseidonChip;

use super::{
    Acc, C, ClientProof, E, F, FoldStep, LeafStep, NodeState, S, TreeNode, TreeResult, VkData,
    circuit::{FrameworkWitness, IvcLeafCircuit, IvcNodeCircuit},
    ctx::configure_ivc_circuit,
};

const LEAF_CLIENT_ARITY: usize = 4;

fn neutralize_acc_for_client_children(acc: &Acc) -> Acc {
    fn neutralized_msm(msm: Msm<S>) -> Msm<S> {
        let bases = msm.bases();
        let scalars = vec![F::ZERO; bases.len()];
        let fixed: BTreeMap<String, F> = msm
            .fixed_base_scalars()
            .keys()
            .cloned()
            .map(|k| (k, F::ZERO))
            .collect();
        Msm::new(&bases, &scalars, &fixed)
    }

    let mut out = Accumulator::new(neutralized_msm(acc.lhs()), neutralized_msm(acc.rhs()));
    out.collapse();
    out
}

////////////////////////////////////////////////////////////////////////////////
// Errors
////////////////////////////////////////////////////////////////////////////////

#[derive(Debug, Error)]
#[allow(dead_code)]
pub enum AggregationError {
    #[error("need at least one client proof")]
    Empty,

    #[error("number of leaf aggregation nodes must be of the form 2*4^k (got {0})")]
    NotPowerOfTwo(usize),

    #[error("client proofs length must be divisible by leaf arity {arity} (got {got})")]
    InvalidLeafArity { arity: usize, got: usize },

    #[error("length mismatch: expected {expected}, got {got}")]
    LenMismatch { expected: usize, got: usize },

    #[error("need at least 8 client proofs for merged final agg (got {0})")]
    NeedAtLeastEight(usize),

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
    #[allow(dead_code)]
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
        assert!(!levels.is_empty());
        Self { levels }
    }

    fn max_level(&self) -> usize {
        self.levels.len()
    }

    pub fn get(&self, level: usize) -> &AggLevelKeys {
        assert!(
            (1..=self.levels.len()).contains(&level),
            "level {level} out of range"
        );
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
    _leaf_srs: &ParamsKZG<Bls12>,
    leaf_vk: &Vk,
    leaf_vk_name: &str,
    leaf_k: u32,
    k_leaf_agg: u32,
    k_internal: u32,
    num_client_proofs: usize,
    app_state_width: usize,
    client_pi_width: usize,
) -> Result<IvcSetup, AggregationError>
where
    L: LeafStep,
    Fo: FoldStep,
{
    if num_client_proofs == 0 || num_client_proofs % LEAF_CLIENT_ARITY != 0 {
        return Err(AggregationError::InvalidLeafArity {
            arity: LEAF_CLIENT_ARITY,
            got: num_client_proofs,
        });
    }

    let num_leaves = num_client_proofs / LEAF_CLIENT_ARITY;
    if num_leaves < 2 {
        return Err(AggregationError::NeedAtLeastEight(num_client_proofs));
    }

    let mut n = num_leaves;
    let mut internal_levels = 0usize;
    while n > 2 {
        if n % 4 != 0 {
            return Err(AggregationError::NotPowerOfTwo(num_leaves));
        }
        n /= 4;
        internal_levels += 1;
    }
    let max_agg_level = 1 + internal_levels;

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

    let fixed_base_names =
        compute_all_fixed_base_names(leaf_vk_name, &leaf_vk_data.cs, &agg_vk_names, &agg_cs);

    let mut levels = Vec::with_capacity(max_agg_level);

    for level in 1..=max_agg_level {
        let (child_vk_data, child_vk_name) = if level == 1 {
            (leaf_vk_data.clone(), leaf_vk_name.to_string())
        } else {
            let prev: &AggLevelKeys = &levels[level - 2];
            (prev.vk_data.clone(), prev.name.clone())
        };

        let name = agg_vk_names[level - 1].clone();
        let k = if level == 1 { k_leaf_agg } else { k_internal };
        let srs = if level == 1 {
            &agg_srs_leaf
        } else {
            &agg_srs_internal
        };

        let start = Instant::now();
        let (vk, pk) = if level == 1 {
            let circuit = IvcLeafCircuit::<L, 20> {
                step: leaf_step.clone(),
                client_items: [
                    Value::unknown(),
                    Value::unknown(),
                    Value::unknown(),
                    Value::unknown(),
                ],
                witness: Value::unknown(),
                client_pi_width,
                fw: FrameworkWitness {
                    child_vk: child_vk_data,
                    child_vk_name,
                    child_proofs: vec![
                        Value::unknown(),
                        Value::unknown(),
                        Value::unknown(),
                        Value::unknown(),
                    ],
                    child_pi_accs: vec![
                        Value::unknown(),
                        Value::unknown(),
                        Value::unknown(),
                        Value::unknown(),
                    ],
                    fixed_base_names: fixed_base_names.clone(),
                },
            };
            keygen_pair(srs, &circuit, k)?
        } else {
            let full_width = app_state_width + 1;
            let circuit = IvcNodeCircuit::<Fo, 19> {
                step: fold_step.clone(),
                app_state_width,
                child0_state: vec![Value::unknown(); full_width],
                child1_state: vec![Value::unknown(); full_width],
                child2_state: vec![Value::unknown(); full_width],
                child3_state: vec![Value::unknown(); full_width],
                fw: FrameworkWitness {
                    child_vk: child_vk_data,
                    child_vk_name,
                    child_proofs: vec![
                        Value::unknown(),
                        Value::unknown(),
                        Value::unknown(),
                        Value::unknown(),
                    ],
                    child_pi_accs: vec![
                        Value::unknown(),
                        Value::unknown(),
                        Value::unknown(),
                        Value::unknown(),
                    ],
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

/// Plan for one leaf aggregation node, produced by the host-side planning layer.
pub struct LeafPlan<W> {
    pub index: usize,
    pub clients: [ClientProof; LEAF_CLIENT_ARITY],
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
        if leaf_plans.len() != setup.num_leaves {
            return Err(AggregationError::LenMismatch {
                expected: setup.num_leaves,
                got: leaf_plans.len(),
            });
        }

        // 1. Prove leaves in parallel
        let leaf_nodes: Vec<TreeNode<Vec<F>>> = leaf_plans
            .into_par_iter()
            .map(|plan| prove_leaf(setup, client_srs, client_vk, leaf_step, plan))
            .collect::<Result<Vec<_>, _>>()?;

        // 2. Build internal levels up to the top pair
        let (_child_level, top_pair) =
            build_to_top_pair(setup, fold_step, &host_merge, 1, leaf_nodes)?;

        let (left, right) = top_pair;

        // 3. Compute the root state
        let root_digest = host_hash_pair(left.merkle_digest, right.merkle_digest);
        let root_app = host_merge(&left.app_state, &right.app_state);
        let root_state = NodeState {
            app_state: root_app,
            merkle_digest: root_digest,
        };

        Ok(TreeResult {
            left_top: left,
            right_top: right,
            root_state,
        })
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

    let circuit = IvcLeafCircuit::<L, 20> {
        step: leaf_step.clone(),
        client_items: [
            Value::known(plan.clients[0].public_inputs.clone()),
            Value::known(plan.clients[1].public_inputs.clone()),
            Value::known(plan.clients[2].public_inputs.clone()),
            Value::known(plan.clients[3].public_inputs.clone()),
        ],
        witness: Value::known(plan.witness),
        client_pi_width: setup.client_pi_width,
        fw: FrameworkWitness {
            child_vk: setup.leaf_vk_data.clone(),
            child_vk_name: setup.leaf_vk_name.clone(),
            child_proofs: plan
                .clients
                .iter()
                .map(|cp| Value::known(cp.proof.clone()))
                .collect(),
            child_pi_accs: (0..LEAF_CLIENT_ARITY)
                .map(|_| Value::known(setup.trivial_combined.clone()))
                .collect(),
            fixed_base_names: setup.fixed_base_names.clone(),
        },
    };

    // Verify client proofs and extract accumulators
    let proof_accs: Vec<Acc> = plan
        .clients
        .iter()
        .map(|cp| {
            verify_and_extract(
                client_srs,
                &setup.leaf_vk_name,
                client_vk,
                &setup.leaf_fixed_bases,
                &cp.proof,
                &cp.public_inputs,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    if plan.index == 0 {
        for (i, acc) in proof_accs.iter().enumerate() {
            eprintln!(
                "client proof_acc[{i}] rhs_fixed_keys={}",
                acc.rhs().fixed_base_scalars().len()
            );
        }
    }

    let neutral_trivial = neutralize_acc_for_client_children(&setup.trivial_combined);
    let mut fold_parts = Vec::with_capacity(LEAF_CLIENT_ARITY * 2);
    for acc in proof_accs {
        fold_parts.push(acc);
        fold_parts.push(neutral_trivial.clone());
    }
    let pi_acc = collapse(Accumulator::accumulate(&fold_parts));

    let mut full_state = plan.app_state.clone();
    full_state.push(plan.merkle_digest);
    let pi_acc_fields = AssignedAccumulator::as_public_input(&pi_acc);
    if plan.index == 0 {
        eprintln!(
            "leaf debug: fixed_base_names={}, trivial_combined_rhs_fixed={}, pi_acc_rhs_fixed={}, pi_acc_fields={}, total_pi={}",
            setup.fixed_base_names.len(),
            setup.trivial_combined.rhs().fixed_base_scalars().len(),
            pi_acc.rhs().fixed_base_scalars().len(),
            pi_acc_fields.len(),
            full_state.len() + pi_acc_fields.len()
        );
    }
    let pi_fields: Vec<F> = full_state.iter().copied().chain(pi_acc_fields).collect();

    if plan.index == 0 {
        match MockProver::run(20, &circuit, vec![vec![], pi_fields.clone()]) {
            Ok(prover) => {
                if let Err(errs) = prover.verify() {
                    eprintln!("leaf mock prover unsatisfied: {:?}", errs);
                } else {
                    eprintln!("leaf mock prover satisfied");
                }
            }
            Err(e) => {
                eprintln!("leaf mock prover failed to run: {:?}", e);
            }
        }
    }

    let start = Instant::now();
    let proof = create_agg_proof(srs, leaf_keys.pk.as_ref(), circuit, &pi_fields)?;
    println!("Leaf AGG {} in {:?}", plan.index, start.elapsed());

    let proof_acc = verify_and_extract(
        srs,
        &leaf_keys.name,
        leaf_keys.vk.as_ref(),
        &leaf_keys.fixed_bases,
        &proof,
        &pi_fields,
    )?;
    println!("hello");

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
    child0: &TreeNode<Vec<F>>,
    child1: &TreeNode<Vec<F>>,
    child2: &TreeNode<Vec<F>>,
    child3: &TreeNode<Vec<F>>,
) -> Result<TreeNode<Vec<F>>, AggregationError> {
    let child_keys = setup.agg_store.get(child_level);
    let parent_keys = setup.agg_store.get(parent_level);
    let srs = &setup.agg_srs_internal;

    let app01 = host_merge(&child0.app_state, &child1.app_state);
    let app012 = host_merge(&app01, &child2.app_state);
    let app_state = host_merge(&app012, &child3.app_state);

    let d01 = host_hash_pair(child0.merkle_digest, child1.merkle_digest);
    let d23 = host_hash_pair(child2.merkle_digest, child3.merkle_digest);
    let digest = host_hash_pair(d01, d23);

    let mut child0_full: Vec<F> = child0.app_state.clone();
    child0_full.push(child0.merkle_digest);
    let mut child1_full: Vec<F> = child1.app_state.clone();
    child1_full.push(child1.merkle_digest);
    let mut child2_full: Vec<F> = child2.app_state.clone();
    child2_full.push(child2.merkle_digest);
    let mut child3_full: Vec<F> = child3.app_state.clone();
    child3_full.push(child3.merkle_digest);

    let circuit = IvcNodeCircuit::<Fo, 19> {
        step: fold_step.clone(),
        app_state_width: setup.app_state_width,
        child0_state: child0_full.iter().map(|f| Value::known(*f)).collect(),
        child1_state: child1_full.iter().map(|f| Value::known(*f)).collect(),
        child2_state: child2_full.iter().map(|f| Value::known(*f)).collect(),
        child3_state: child3_full.iter().map(|f| Value::known(*f)).collect(),
        fw: FrameworkWitness {
            child_vk: child_keys.vk_data.clone(),
            child_vk_name: child_keys.name.clone(),
            child_proofs: vec![
                Value::known(child0.proof.clone()),
                Value::known(child1.proof.clone()),
                Value::known(child2.proof.clone()),
                Value::known(child3.proof.clone()),
            ],
            child_pi_accs: vec![
                Value::known(child0.pi_acc.clone()),
                Value::known(child1.pi_acc.clone()),
                Value::known(child2.pi_acc.clone()),
                Value::known(child3.pi_acc.clone()),
            ],
            fixed_base_names: setup.fixed_base_names.clone(),
        },
    };

    let pi_acc = collapse(Accumulator::accumulate(&[
        child0.proof_acc.clone(),
        child0.pi_acc.clone(),
        child1.proof_acc.clone(),
        child1.pi_acc.clone(),
        child2.proof_acc.clone(),
        child2.pi_acc.clone(),
        child3.proof_acc.clone(),
        child3.pi_acc.clone(),
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
        srs,
        &parent_keys.name,
        parent_keys.vk.as_ref(),
        &parent_keys.fixed_bases,
        &proof,
        &pi_fields,
    )?;

    Ok(TreeNode {
        app_state,
        merkle_digest: digest,
        proof,
        proof_acc,
        pi_acc,
    })
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
            1 => return Err(AggregationError::NeedAtLeastEight(LEAF_CLIENT_ARITY * 2)),
            2 => {
                let right = nodes.pop().unwrap();
                let left = nodes.pop().unwrap();
                return Ok((level, (left, right)));
            }
            _ => {
                if nodes.len() % 4 != 0 {
                    return Err(AggregationError::FoldValidation(format!(
                        "internal level {level} requires node count divisible by 4, got {}",
                        nodes.len()
                    )));
                }
                let parent_level = level + 1;
                let quads: Vec<_> = nodes.chunks_exact(4).collect();
                nodes = quads
                    .par_iter()
                    .map(|quad| {
                        prove_node(
                            setup,
                            fold_step,
                            host_merge,
                            parent_level,
                            level,
                            &quad[0],
                            &quad[1],
                            &quad[2],
                            &quad[3],
                        )
                    })
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
    vk_name: &str,
    vk: &Vk,
    fixed_bases: &BTreeMap<String, C>,
    proof: &[u8],
    public_inputs: &[F],
) -> Result<Acc, AggregationError> {
    let run_prepare = |committed: &[&[C]], instances: &[&[&[F]]]| {
        let mut transcript = CircuitTranscript::<PoseidonState<F>>::init_from_bytes(proof);
        midnight_proofs::plonk::prepare::<
            F,
            KZGCommitmentScheme<E>,
            CircuitTranscript<PoseidonState<F>>,
        >(vk, committed, instances, &mut transcript)
    };

    let committed: &[&[C]] = &[&[C::identity()]];
    let instances: &[&[&[F]]] = &[&[public_inputs]];

    let dual_msm =
        run_prepare(committed, instances).map_err(|_| AggregationError::PrepareFailed)?;

    if !dual_msm.clone().check(&srs.verifier_params()) {
        let empty_committed: &[&[C]] = &[&[]];
        let empty_instances: &[&[&[F]]] = &[&[]];
        let two_instance_cols: &[&[&[F]]] = &[&[&[], public_inputs]];

        let alt_empty_committed = run_prepare(empty_committed, instances)
            .map(|m| m.check(&srs.verifier_params()))
            .unwrap_or(false);
        let alt_no_instances = run_prepare(committed, empty_instances)
            .map(|m| m.check(&srs.verifier_params()))
            .unwrap_or(false);
        let alt_two_instance_cols = run_prepare(committed, two_instance_cols)
            .map(|m| m.check(&srs.verifier_params()))
            .unwrap_or(false);
        let alt_empty_committed_two_instances = run_prepare(empty_committed, two_instance_cols)
            .map(|m| m.check(&srs.verifier_params()))
            .unwrap_or(false);

        eprintln!(
            "dual_msm check failed for {vk_name}: proof_len={}, pi_len={}, alt(empty_committed+1inst)={}, alt(committed+0inst)={}, alt(committed+2inst)={}, alt(empty_committed+2inst)={}",
            proof.len(),
            public_inputs.len(),
            alt_empty_committed,
            alt_no_instances,
            alt_two_instance_cols,
            alt_empty_committed_two_instances
        );
        return Err(AggregationError::DualMsmFailed);
    }

    let mut acc = Accumulator::from_dual_msm(dual_msm, vk_name, fixed_bases);
    acc.collapse();

    if !acc.check(&srs.verifier_params(), fixed_bases) {
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
        srs,
        pk,
        &[circuit],
        1,
        &[&[&[], public_inputs]],
        OsRng,
        &mut transcript,
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
    let pk =
        keygen_pk(vk.clone(), circuit).map_err(|e| AggregationError::Setup(format!("pk: {e}")))?;
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
    midnight_circuits::verifier::fixed_base_names::<S>(
        name,
        cs.num_fixed_columns() + cs.num_selectors(),
        cs.permutation().columns.len(),
    )
}

fn compute_fixed_bases_for_vk(name: &str, vk: &Vk) -> BTreeMap<String, C> {
    midnight_circuits::verifier::fixed_bases::<S>(name, vk)
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
