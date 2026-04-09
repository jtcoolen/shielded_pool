//! Shielded-pool rollup: application-specific IVC step implementations.
//!
//! Implements [`LeafStep`], [`FoldStep`], [`DeciderStep`], and their host-side
//! counterparts for the shielded-pool rollup state transition.

use ff::Field;
use midnight_circuits::{
    hash::poseidon::PoseidonChip,
    instructions::{hash::HashCPU, map::MapInstructions},
    types::AssignedNative,
};
use midnight_proofs::circuit::{Layouter, Value};
use midnight_proofs::plonk::Error;

use crate::ivc::{
    self, ClientProof, DeciderStep, F, FoldStep, HostLeafStep, HostState, IvcCtx,
    LEAF_CLIENT_ARITY, LeafStep, Map, MapGadget, engine::AggregationError,
};

use midnight_circuits::instructions::map::MapCPU;

////////////////////////////////////////////////////////////////////////////////
// Application state
////////////////////////////////////////////////////////////////////////////////

pub const APP_STATE_WIDTH: usize = 6;

/// Rollup state: commitment/nullifier set roots, roots-set root, block level.
///
/// The Merkle digest (instance commitment) is managed by the IVC framework
/// and is NOT part of this struct.
#[derive(Clone, Copy, Debug)]
pub struct RollupAppState {
    pub c_pre: F,
    pub c_post: F,
    pub n_pre: F,
    pub n_post: F,
    pub commitment_roots_set_root: F,
    pub block_level: F,
}

impl HostState for RollupAppState {
    const WIDTH: usize = APP_STATE_WIDTH;

    fn to_fields(&self) -> Vec<F> {
        vec![
            self.c_pre,
            self.c_post,
            self.n_pre,
            self.n_post,
            self.commitment_roots_set_root,
            self.block_level,
        ]
    }

    fn from_fields(f: &[F]) -> Self {
        Self {
            c_pre: f[0],
            c_post: f[1],
            n_pre: f[2],
            n_post: f[3],
            commitment_roots_set_root: f[4],
            block_level: f[5],
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// SendableMap wrapper (Map contains Rc, needs unsafe Send for rayon)
////////////////////////////////////////////////////////////////////////////////

#[repr(transparent)]
#[derive(Clone)]
pub struct SendableMap(pub Map);

// SAFETY — Send: each rayon task receives its own cloned Map via LeafPlan.
// No Map instance is shared mutably across threads.
//
// SAFETY — Sync: required by rayon par_iter which hands out &LeafPlan.
// Map is an immutable Poseidon Merkle tree of field elements with no
// interior mutability beyond the Rc bookkeeping; the only use through
// &SendableMap is cloning.
unsafe impl Send for SendableMap {}
unsafe impl Sync for SendableMap {}

impl SendableMap {
    #[allow(dead_code)]
    pub fn inner(&self) -> &Map {
        &self.0
    }
    pub fn into_inner(self) -> Map {
        self.0
    }
}

////////////////////////////////////////////////////////////////////////////////
// Leaf step: processes six client transactions
////////////////////////////////////////////////////////////////////////////////

#[derive(Clone)]
pub struct RollupLeafStep;

#[derive(Clone)]
pub struct LeafWitness {
    pub pre_commitment_map: SendableMap,
    pub pre_nullifier_map: SendableMap,
    pub pre_commitment_roots_set_map: SendableMap,
    pub block_level: F,
}

impl LeafStep for RollupLeafStep {
    type Witness = LeafWitness;

    fn synthesize<L: Layouter<F>>(
        &self,
        ctx: &IvcCtx,
        layouter: &mut L,
        p0: &[AssignedNative<F>],
        p1: &[AssignedNative<F>],
        p2: &[AssignedNative<F>],
        p3: &[AssignedNative<F>],
        p4: &[AssignedNative<F>],
        p5: &[AssignedNative<F>],
        witness: Value<Self::Witness>,
    ) -> Result<Vec<AssignedNative<F>>, Error> {
        let one = ctx.one(layouter)?;
        let zero = ctx.zero(layouter)?;
        let blk = ctx.assign(layouter, witness.clone().map(|w| w.block_level))?;

        // Init rollup maps (unwrap SendableMap)
        let mut commit_map = ctx.init_map(
            layouter,
            witness.clone().map(|w| w.pre_commitment_map.into_inner()),
        )?;
        let c_pre = commit_map.succinct_repr();
        let mut null_map = ctx.init_map(
            layouter,
            witness.clone().map(|w| w.pre_nullifier_map.into_inner()),
        )?;
        let n_pre = null_map.succinct_repr();

        // Historic roots set (read-only at leaf level)
        let roots_set = ctx.init_map(
            layouter,
            witness.map(|w| w.pre_commitment_roots_set_map.into_inner()),
        )?;
        let roots_set_root = roots_set.succinct_repr();

        // Check each tx root ∈ historic roots set
        let roots = [&p0[0], &p1[0], &p2[0], &p3[0], &p4[0], &p5[0]];
        for root in roots {
            check_membership(ctx, layouter, &roots_set, root, &one)?;
        }

        apply_effects(
            ctx,
            layouter,
            &mut commit_map,
            &mut null_map,
            p0,
            &zero,
            &one,
        )?;
        apply_effects(
            ctx,
            layouter,
            &mut commit_map,
            &mut null_map,
            p1,
            &zero,
            &one,
        )?;
        apply_effects(
            ctx,
            layouter,
            &mut commit_map,
            &mut null_map,
            p2,
            &zero,
            &one,
        )?;
        apply_effects(
            ctx,
            layouter,
            &mut commit_map,
            &mut null_map,
            p3,
            &zero,
            &one,
        )?;
        apply_effects(
            ctx,
            layouter,
            &mut commit_map,
            &mut null_map,
            p4,
            &zero,
            &one,
        )?;
        apply_effects(
            ctx,
            layouter,
            &mut commit_map,
            &mut null_map,
            p5,
            &zero,
            &one,
        )?;

        Ok(vec![
            c_pre,
            commit_map.succinct_repr(),
            n_pre,
            null_map.succinct_repr(),
            roots_set_root,
            blk,
        ])
    }
}

////////////////////////////////////////////////////////////////////////////////
// Fold step: stitches two child rollup states
////////////////////////////////////////////////////////////////////////////////

#[derive(Clone)]
pub struct RollupFoldStep;

impl FoldStep for RollupFoldStep {
    fn synthesize<L: Layouter<F>>(
        &self,
        ctx: &IvcCtx,
        layouter: &mut L,
        left: &[AssignedNative<F>],
        right: &[AssignedNative<F>],
    ) -> Result<Vec<AssignedNative<F>>, Error> {
        // left = [c_pre, c_post, n_pre, n_post, roots_set_root, blk]
        // Stitch: left.c_post == right.c_pre, left.n_post == right.n_pre
        ctx.assert_eq(layouter, &left[1], &right[0])?;
        ctx.assert_eq(layouter, &left[3], &right[2])?;
        // Same roots_set_root and block_level
        ctx.assert_eq(layouter, &left[4], &right[4])?;
        ctx.assert_eq(layouter, &left[5], &right[5])?;

        Ok(vec![
            left[0].clone(),  // c_pre (from left)
            right[1].clone(), // c_post (from right)
            left[2].clone(),  // n_pre (from left)
            right[3].clone(), // n_post (from right)
            left[4].clone(),  // roots_set_root (shared)
            left[5].clone(),  // block_level (shared)
        ])
    }
}

////////////////////////////////////////////////////////////////////////////////
// Decider step: final wrap (roots-set update + block counter)
////////////////////////////////////////////////////////////////////////////////

#[derive(Clone)]
pub struct RollupDeciderStep;

#[derive(Clone)]
pub struct DeciderWitness {
    pub pre_commitment_roots_set_map: SendableMap,
    pub post_commitment_roots_set_root: F,
    pub blk_pre: F,
    pub blk_post: F,
}

impl DeciderStep for RollupDeciderStep {
    type Witness = DeciderWitness;

    fn synthesize<L: Layouter<F>>(
        &self,
        ctx: &IvcCtx,
        layouter: &mut L,
        left_full: &[AssignedNative<F>],
        right_full: &[AssignedNative<F>],
        merkle_root: &AssignedNative<F>,
        witness: Value<Self::Witness>,
    ) -> Result<Vec<AssignedNative<F>>, Error> {
        let w = APP_STATE_WIDTH;
        let (left_app, _left_digest) = left_full.split_at(w);
        let (right_app, _right_digest) = right_full.split_at(w);

        let one = ctx.one(layouter)?;

        // Stitch children
        ctx.assert_eq(layouter, &left_app[1], &right_app[0])?; // c_post == c_pre
        ctx.assert_eq(layouter, &left_app[3], &right_app[2])?; // n_post == n_pre

        let c_pre = &left_app[0];
        let c_post = &right_app[1];
        let n_pre = &left_app[2];
        let n_post = &right_app[3];

        // Block counter: blk_post = blk_pre + 1, agg.blk == blk_post
        let blk_pre = ctx.assign(layouter, witness.clone().map(|w| w.blk_pre))?;
        let blk_post = ctx.assign(layouter, witness.clone().map(|w| w.blk_post))?;
        let blk_pre_plus_one = ctx.add(layouter, &blk_pre, &one)?;
        ctx.assert_eq(layouter, &blk_post, &blk_pre_plus_one)?;
        ctx.assert_eq(layouter, &left_app[5], &blk_post)?;
        ctx.assert_eq(layouter, &right_app[5], &blk_post)?;

        // Historic roots-set: bind, check c_pre ∈ set, check c_post ∉ set, insert c_post
        let mut roots_set = ctx.init_map(
            layouter,
            witness
                .clone()
                .map(|w| w.pre_commitment_roots_set_map.into_inner()),
        )?;
        let pre_roots_root = roots_set.succinct_repr();
        let zero = ctx.zero(layouter)?;

        ctx.assert_eq(layouter, &left_app[4], &pre_roots_root)?;
        ctx.assert_eq(layouter, &right_app[4], &pre_roots_root)?;

        check_membership(ctx, layouter, &roots_set, c_pre, &one)?;

        let post_exists = roots_set.get(layouter, c_post)?;
        ctx.assert_eq(layouter, &post_exists, &zero)?;

        roots_set.insert(layouter, c_post, &one)?;
        let post_roots_root =
            ctx.assign(layouter, witness.map(|w| w.post_commitment_roots_set_root))?;
        ctx.assert_eq(layouter, &roots_set.succinct_repr(), &post_roots_root)?;

        // Public inputs: c_pre, c_post, n_pre, n_post, blk_pre, blk_post,
        //                merkle_root, pre_roots_root, post_roots_root
        Ok(vec![
            c_pre.clone(),
            c_post.clone(),
            n_pre.clone(),
            n_post.clone(),
            blk_pre,
            blk_post,
            merkle_root.clone(),
            pre_roots_root,
            post_roots_root,
        ])
    }
}

////////////////////////////////////////////////////////////////////////////////
// Host-side leaf step
////////////////////////////////////////////////////////////////////////////////

pub struct RollupHostLeaf {
    pub pre_commitment_map: Map,
    pub pre_nullifier_map: Map,
    pub pre_roots_set_map: Map,
    pub block_level: F,
}

#[allow(dead_code)]
impl HostLeafStep for RollupHostLeaf {
    type AppState = RollupAppState;

    fn plan_pair(
        &self,
        left: &ClientProof,
        right: &ClientProof,
    ) -> Result<RollupAppState, AggregationError> {
        // Validate roots membership
        if self.pre_roots_set_map.get(&left.public_inputs[0]) == F::ZERO {
            return Err(AggregationError::LeafValidation(
                "left tx root not in roots set".into(),
            ));
        }
        if self.pre_roots_set_map.get(&right.public_inputs[0]) == F::ZERO {
            return Err(AggregationError::LeafValidation(
                "right tx root not in roots set".into(),
            ));
        }

        let c_pre = self.pre_commitment_map.succinct_repr();
        let n_pre = self.pre_nullifier_map.succinct_repr();

        // Apply left then right tx effects to maps
        let mut cmap = self.pre_commitment_map.clone();
        let mut nmap = self.pre_nullifier_map.clone();
        host_apply_effects(&mut cmap, &mut nmap, &left.public_inputs)?;
        host_apply_effects(&mut cmap, &mut nmap, &right.public_inputs)?;

        Ok(RollupAppState {
            c_pre,
            c_post: cmap.succinct_repr(),
            n_pre,
            n_post: nmap.succinct_repr(),
            commitment_roots_set_root: self.pre_roots_set_map.succinct_repr(),
            block_level: self.block_level,
        })
    }
}

////////////////////////////////////////////////////////////////////////////////
// Host-side fold (merge two child app states)
////////////////////////////////////////////////////////////////////////////////

/// Mirror of `RollupFoldStep::synthesize` on the host side.
/// Used by `IvcProver::prove_tree` to compute parent app state.
pub fn rollup_host_merge(left: &[F], right: &[F]) -> Vec<F> {
    // left = [c_pre, c_post, n_pre, n_post, roots_set_root, blk]
    vec![
        left[0],  // c_pre (from left)
        right[1], // c_post (from right)
        left[2],  // n_pre (from left)
        right[3], // n_post (from right)
        left[4],  // roots_set_root (shared)
        left[5],  // block_level (shared)
    ]
}

////////////////////////////////////////////////////////////////////////////////
// Leaf plan construction
////////////////////////////////////////////////////////////////////////////////

use crate::ivc::engine::LeafPlan;

/// Build leaf plans from a batch of client proofs.
///
/// Each chunk of 6 client proofs becomes one leaf plan.
/// The running commitment/nullifier maps are updated sequentially so
/// that each leaf sees the correct pre-state.
pub fn plan_rollup_leaves(
    client_proofs: &[ivc::ClientProof],
    pre_commitment_map: Map,
    pre_nullifier_map: Map,
    pre_roots_set_map: &Map,
    block_level: F,
) -> Result<Vec<LeafPlan<LeafWitness>>, AggregationError> {
    let mut cmap = pre_commitment_map;
    let mut nmap = pre_nullifier_map;
    let roots_set_root = pre_roots_set_map.succinct_repr();

    if client_proofs.len() % LEAF_CLIENT_ARITY != 0 {
        return Err(AggregationError::LeafValidation(format!(
            "client proofs length must be divisible by {LEAF_CLIENT_ARITY}"
        )));
    }

    let mut plans = Vec::with_capacity(client_proofs.len() / LEAF_CLIENT_ARITY);

    for (i, chunk) in client_proofs.chunks_exact(LEAF_CLIENT_ARITY).enumerate() {
        for (j, cp) in chunk.iter().enumerate() {
            if pre_roots_set_map.get(&cp.public_inputs[0]) == F::ZERO {
                return Err(AggregationError::LeafValidation(format!(
                    "leaf {i} tx {j} root not in roots set"
                )));
            }
        }

        let c_pre = cmap.succinct_repr();
        let n_pre = nmap.succinct_repr();
        let pre_cmap_for_witness = cmap.clone();
        let pre_nmap_for_witness = nmap.clone();

        for cp in chunk {
            host_apply_effects(&mut cmap, &mut nmap, &cp.public_inputs)?;
        }

        let app_state = vec![
            c_pre,
            cmap.succinct_repr(),
            n_pre,
            nmap.succinct_repr(),
            roots_set_root,
            block_level,
        ];

        let h01 = host_hash_pair(chunk[0].instance_hash, chunk[1].instance_hash);
        let h23 = host_hash_pair(chunk[2].instance_hash, chunk[3].instance_hash);
        let h45 = host_hash_pair(chunk[4].instance_hash, chunk[5].instance_hash);
        let h0123 = host_hash_pair(h01, h23);
        let merkle_digest = host_hash_pair(h0123, h45);

        plans.push(LeafPlan {
            index: i,
            clients: [
                chunk[0].clone(),
                chunk[1].clone(),
                chunk[2].clone(),
                chunk[3].clone(),
                chunk[4].clone(),
                chunk[5].clone(),
            ],
            app_state,
            merkle_digest,
            witness: LeafWitness {
                pre_commitment_map: SendableMap(pre_cmap_for_witness),
                pre_nullifier_map: SendableMap(pre_nmap_for_witness),
                pre_commitment_roots_set_map: SendableMap(pre_roots_set_map.clone()),
                block_level,
            },
        });
    }

    Ok(plans)
}

fn host_hash_pair(a: F, b: F) -> F {
    <PoseidonChip<F> as HashCPU<F, F>>::hash(&[a, b])
}

////////////////////////////////////////////////////////////////////////////////
// Helpers
////////////////////////////////////////////////////////////////////////////////

fn check_membership(
    ctx: &IvcCtx,
    layouter: &mut impl Layouter<F>,
    map: &MapGadget,
    key: &AssignedNative<F>,
    one: &AssignedNative<F>,
) -> Result<(), Error> {
    let v = map.get(layouter, key)?;
    ctx.assert_eq(layouter, &v, one)
}

fn apply_effects(
    ctx: &IvcCtx,
    layouter: &mut impl Layouter<F>,
    commit_map: &mut MapGadget,
    null_map: &mut MapGadget,
    items: &[AssignedNative<F>],
    zero: &AssignedNative<F>,
    one: &AssignedNative<F>,
) -> Result<(), Error> {
    // items = [root, pk_bx, pk_by, new_c1, new_c2, nf1, nf2]
    let new_c1 = &items[3];
    let new_c2 = &items[4];
    let nf1 = &items[5];
    let nf2 = &items[6];

    // Commitment uniqueness
    let c1_exists = commit_map.get(layouter, new_c1)?;
    ctx.assert_eq(layouter, &c1_exists, zero)?;
    commit_map.insert(layouter, new_c1, one)?;

    let c2_exists = commit_map.get(layouter, new_c2)?;
    ctx.assert_eq(layouter, &c2_exists, zero)?;
    commit_map.insert(layouter, new_c2, one)?;

    // Nullifier freshness
    let n1_exists = null_map.get(layouter, nf1)?;
    ctx.assert_eq(layouter, &n1_exists, zero)?;
    null_map.insert(layouter, nf1, one)?;

    let n2_exists = null_map.get(layouter, nf2)?;
    ctx.assert_eq(layouter, &n2_exists, zero)?;
    null_map.insert(layouter, nf2, one)?;

    Ok(())
}

fn host_apply_effects(cmap: &mut Map, nmap: &mut Map, items: &[F]) -> Result<(), AggregationError> {
    let [_root, _bx, _by, c1, c2, nf1, nf2] = items[..7] else {
        return Err(AggregationError::LeafValidation(
            "expected 7 public items".into(),
        ));
    };

    if cmap.get(&c1) != F::ZERO {
        return Err(AggregationError::CommitmentAlreadyExists);
    }
    if cmap.get(&c2) != F::ZERO {
        return Err(AggregationError::CommitmentAlreadyExists);
    }
    if nmap.get(&nf1) != F::ZERO {
        return Err(AggregationError::NullifierAlreadySpent);
    }
    if nmap.get(&nf2) != F::ZERO {
        return Err(AggregationError::NullifierAlreadySpent);
    }

    cmap.insert(&c1, &F::ONE);
    cmap.insert(&c2, &F::ONE);
    nmap.insert(&nf1, &F::ONE);
    nmap.insert(&nf2, &F::ONE);
    Ok(())
}
