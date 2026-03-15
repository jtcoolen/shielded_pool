use ff::Field;
use group::Group;

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
        ArithInstructions, AssertionInstructions, AssignmentInstructions, HashInstructions,
        PublicInputInstructions, map::MapInstructions,
    },
    map::cpu::MapMt,
    types::{AssignedForeignPoint, AssignedNative, ComposableChip, Instantiable},
    verifier::{
        Accumulator, AssignedAccumulator, AssignedVk, BlstrsEmulation, SelfEmulation,
        VerifierGadget,
    },
};
use midnight_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{Circuit, ConstraintSystem, Error},
    poly::EvaluationDomain,
};

////////////////////////////////////////////////////////////////////////////////
// Small utilities (local helpers)
////////////////////////////////////////////////////////////////////////////////

fn alloc<T>(n: usize, mut f: impl FnMut() -> T) -> Vec<T> {
    (0..n).map(|_| f()).collect()
}

fn first<const N: usize, T: Copy>(xs: &[T], what: &'static str) -> [T; N] {
    xs.get(..N)
        .unwrap_or_else(|| panic!("not enough columns for {what}: need {N}, got {}", xs.len()))
        .try_into()
        .unwrap()
}

fn try_array_from_fn<const N: usize, T, E>(
    mut f: impl FnMut(usize) -> Result<T, E>,
) -> Result<[T; N], E> {
    let mut v = Vec::with_capacity(N);
    for i in 0..N {
        v.push(f(i)?);
    }
    Ok(v.try_into().ok().unwrap())
}

/// Project `Value<[F; N]>` into `[Value<F>; N]`.
fn project_value_array<const N: usize, F: Copy>(v: Value<[F; N]>) -> [Value<F>; N] {
    core::array::from_fn(|i| v.as_ref().map(|arr| arr[i]))
}

/// Assign `[Value<F>; N]` into `[AssignedNative<F>; N]`
fn assign_values<const N: usize>(
    ctx: &AggCtx,
    layouter: &mut impl Layouter<F>,
    values: [Value<F>; N],
) -> Result<[AssignedNative<F>; N], Error> {
    try_array_from_fn(|i| ctx.scalar.assign(layouter, values[i]))
}

////////////////////////////////////////////////////////////////////////////////
// Types & constants
////////////////////////////////////////////////////////////////////////////////

pub type S = BlstrsEmulation;

type F = <S as SelfEmulation>::F;
type C = <S as SelfEmulation>::C;
type CBase = <C as midnight_circuits::ecc::curves::CircuitCurve>::Base;

type NG = NativeGadget<F, P2RDecompositionChip<F>, NativeChip<F>>;
type Map = MapMt<F, PoseidonChip<F>>;

pub type CurveChip = ForeignEccChip<F, C, C, NG, NG>;
pub type MapGadget = midnight_circuits::map::map_gadget::MapGadget<F, NG, PoseidonChip<F>>;

pub type IdPoint = AssignedForeignPoint<
    midnight_curves::Fq,
    midnight_curves::G1Projective,
    midnight_curves::G1Projective,
>;

/// Leaf K.
pub const K_LEAF: u32 = 19;
/// Internal node K.
pub const K_INTERNAL: u32 = 19;

/// K used by the wrap circuit (final aggregation).
pub const AGG_K: u32 = K_INTERNAL;

/// Width of the aggregation state exposed by aggregation circuits.
/// This includes the historic commitment-roots-set Merkle map root, used to bind membership checks.
pub const AGG_STATE_WIDTH: usize = 7;

/// Width of client proof public items (canonical order).
pub const CLIENT_ITEMS_WIDTH: usize = 7;

////////////////////////////////////////////////////////////////////////////////
// Aggregation state
////////////////////////////////////////////////////////////////////////////////

/// Aggregation state carried between nodes.
///
/// The `commitment_roots_set_root` binds *all* root-membership checks inside leaf proofs to a single
/// historic commitment-roots-set Merkle map root. This prevents mixing membership checks performed
/// against different roots across the aggregation tree.
#[derive(Clone, Copy, Debug)]
pub struct AggState {
    pub c_pre: F,
    pub c_post: F,
    pub n_pre: F,
    pub n_post: F,
    pub subroot: F,
    pub commitment_roots_set_root: F,
    pub block_level: F,
}

impl AggState {
    /// Canonical encoding used for public inputs.
    #[inline]
    pub fn to_fields(&self) -> [F; AGG_STATE_WIDTH] {
        [
            self.c_pre,
            self.c_post,
            self.n_pre,
            self.n_post,
            self.subroot,
            self.commitment_roots_set_root,
            self.block_level,
        ]
    }
}

////////////////////////////////////////////////////////////////////////////////
// VK container
////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Debug)]
pub struct VkData {
    pub domain: EvaluationDomain<F>,
    pub cs: ConstraintSystem<F>,
    pub transcript_repr: F,
}

////////////////////////////////////////////////////////////////////////////////
// Circuit configuration
////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Debug)]
pub struct AggCircuitConfig {
    pub native: NativeConfig,
    pub decomp: P2RDecompositionConfig,
    pub curve: ForeignEccConfig<C>,
    pub poseidon: PoseidonConfig<F>,
}

pub fn configure_agg_circuit(meta: &mut ConstraintSystem<F>) -> AggCircuitConfig {
    // Ensure we allocate enough columns for all sub-chips (not just ECC).
    let nb_advice_cols = nb_foreign_ecc_chip_columns::<F, C, C, NG>().max(NB_POSEIDON_ADVICE_COLS);

    // Native arith uses NB_ARITH_COLS + 4 fixed columns; Poseidon needs NB_POSEIDON_FIXED_COLS.
    let nb_fixed_cols = (NB_ARITH_COLS + 4).max(NB_POSEIDON_FIXED_COLS);

    let advice_columns = alloc(nb_advice_cols, || meta.advice_column());
    let fixed_columns = alloc(nb_fixed_cols, || meta.fixed_column());

    let instance_cols = [meta.instance_column(), meta.instance_column()];

    let native_config = NativeChip::configure(
        meta,
        &(
            first::<NB_ARITH_COLS, _>(&advice_columns, "native advice"),
            first::<{ NB_ARITH_COLS + 4 }, _>(&fixed_columns, "native fixed"),
            instance_cols,
        ),
    );

    let decomp_config = {
        // NOTE: matches original pattern; Pow2Range uses a slice of native advice.
        let pow2_config = Pow2RangeChip::configure(meta, &advice_columns[1..NB_ARITH_COLS]);
        P2RDecompositionChip::configure(meta, &(native_config.clone(), pow2_config))
    };

    let base_config = FieldChip::<F, CBase, C, NG>::configure(meta, &advice_columns);
    let curve_config =
        ForeignEccChip::<F, C, C, NG, NG>::configure(meta, &base_config, &advice_columns);

    let poseidon_config = PoseidonChip::configure(
        meta,
        &(
            first::<NB_POSEIDON_ADVICE_COLS, _>(&advice_columns, "poseidon advice"),
            first::<NB_POSEIDON_FIXED_COLS, _>(&fixed_columns, "poseidon fixed"),
        ),
    );

    AggCircuitConfig {
        native: native_config,
        decomp: decomp_config,
        curve: curve_config,
        poseidon: poseidon_config,
    }
}

////////////////////////////////////////////////////////////////////////////////
// Encodings (struct <-> array)
////////////////////////////////////////////////////////////////////////////////

/// Typed encoding for the 7 public items in client proofs.
///
/// Canonical order: `[root_before, pk_bx, pk_by, new_c1, new_c2, nf1, nf2]`.
#[derive(Clone, Debug)]
pub struct ClientPublicItems<T> {
    pub root_before: T,
    pub pk_bx: T,
    pub pk_by: T,
    pub new_c1: T,
    pub new_c2: T,
    pub nf1: T,
    pub nf2: T,
}

impl<T> From<[T; CLIENT_ITEMS_WIDTH]> for ClientPublicItems<T> {
    fn from(arr: [T; CLIENT_ITEMS_WIDTH]) -> Self {
        let [root_before, pk_bx, pk_by, new_c1, new_c2, nf1, nf2] = arr;
        Self {
            root_before,
            pk_bx,
            pk_by,
            new_c1,
            new_c2,
            nf1,
            nf2,
        }
    }
}

impl<T: Clone> ClientPublicItems<T> {
    #[inline]
    pub fn as_array(&self) -> [T; CLIENT_ITEMS_WIDTH] {
        [
            self.root_before.clone(),
            self.pk_bx.clone(),
            self.pk_by.clone(),
            self.new_c1.clone(),
            self.new_c2.clone(),
            self.nf1.clone(),
            self.nf2.clone(),
        ]
    }

    #[inline]
    pub fn as_vec(&self) -> Vec<T> {
        Vec::from(self.as_array())
    }
}

impl<T> ClientPublicItems<T> {
    #[inline]
    pub fn commitments(&self) -> impl Iterator<Item = &T> {
        core::iter::once(&self.new_c1).chain(core::iter::once(&self.new_c2))
    }

    #[inline]
    pub fn nullifiers(&self) -> impl Iterator<Item = &T> {
        core::iter::once(&self.nf1).chain(core::iter::once(&self.nf2))
    }
}

/// Typed encoding for the Agg state public inputs (6 fields).
///
/// Canonical order (must match `AggState::to_fields()`):
/// `[c_pre, c_post, n_pre, n_post, subroot, commitment_roots_set_root]`.
#[derive(Clone, Debug)]
pub struct AggStateFields<T> {
    pub c_pre: T,
    pub c_post: T,
    pub n_pre: T,
    pub n_post: T,
    pub subroot: T,
    pub commitment_roots_set_root: T,
    pub block_level: T,
}

impl<T> AggStateFields<T> {
    #[inline]
    pub fn into_array(self) -> [T; AGG_STATE_WIDTH] {
        [
            self.c_pre,
            self.c_post,
            self.n_pre,
            self.n_post,
            self.subroot,
            self.commitment_roots_set_root,
            self.block_level,
        ]
    }
}

impl<T: Clone> AggStateFields<T> {
    #[inline]
    pub fn boundary4(&self) -> [T; 4] {
        [
            self.c_pre.clone(),
            self.c_post.clone(),
            self.n_pre.clone(),
            self.n_post.clone(),
        ]
    }

    #[inline]
    pub fn as_vec(&self) -> Vec<T> {
        vec![
            self.c_pre.clone(),
            self.c_post.clone(),
            self.n_pre.clone(),
            self.n_post.clone(),
            self.subroot.clone(),
            self.commitment_roots_set_root.clone(),
            self.block_level.clone(),
        ]
    }
}

////////////////////////////////////////////////////////////////////////////////
// Context (chips bundle)
////////////////////////////////////////////////////////////////////////////////

/// Bundles all chips required by the aggregation circuits.
///
/// This keeps synthesize methods focused on *workflow*, while gadget logic lives in
/// small helper functions.
#[derive(Clone)]
pub struct AggCtx {
    pub native: NativeChip<F>,
    pub core_decomp: P2RDecompositionChip<F>,
    pub scalar: NG,
    pub curve: CurveChip,
    pub poseidon: PoseidonChip<F>,
    pub verifier: VerifierGadget<S>,
}

impl AggCtx {
    pub fn new(cfg: &AggCircuitConfig, k_minus_1: usize) -> Self {
        let native = <NativeChip<F> as ComposableChip<F>>::new(&cfg.native, &());
        let core_decomp = P2RDecompositionChip::new(&cfg.decomp, &k_minus_1);
        let scalar = NativeGadget::new(core_decomp.clone(), native.clone());
        let curve = ForeignEccChip::new(&cfg.curve, &scalar, &scalar);
        let poseidon = PoseidonChip::new(&cfg.poseidon, &native);
        let verifier = VerifierGadget::<S>::new(&curve, &scalar, &poseidon);

        Self {
            native,
            core_decomp,
            scalar,
            curve,
            poseidon,
            verifier,
        }
    }

    pub fn load(&self, layouter: &mut impl Layouter<F>) -> Result<(), Error> {
        self.core_decomp.load(layouter)
    }

    pub fn id_point(&self, layouter: &mut impl Layouter<F>) -> Result<IdPoint, Error> {
        self.curve.assign_fixed(layouter, C::identity())
    }

    pub fn one(&self, layouter: &mut impl Layouter<F>) -> Result<AssignedNative<F>, Error> {
        self.scalar.assign_fixed(layouter, F::ONE)
    }

    pub fn zero(&self, layouter: &mut impl Layouter<F>) -> Result<AssignedNative<F>, Error> {
        self.scalar.assign_fixed(layouter, F::ZERO)
    }

    #[inline]
    pub fn assert_eq(
        &self,
        layouter: &mut impl Layouter<F>,
        a: &AssignedNative<F>,
        b: &AssignedNative<F>,
    ) -> Result<(), Error> {
        self.scalar.assert_equal(layouter, a, b)
    }

    #[inline]
    pub fn hash2(
        &self,
        layouter: &mut impl Layouter<F>,
        a: &AssignedNative<F>,
        b: &AssignedNative<F>,
    ) -> Result<AssignedNative<F>, Error> {
        self.poseidon.hash(layouter, &[a.clone(), b.clone()])
    }

    #[inline]
    pub fn hash_client_instance(
        &self,
        layouter: &mut impl Layouter<F>,
        items: &ClientPublicItems<AssignedNative<F>>,
    ) -> Result<AssignedNative<F>, Error> {
        self.poseidon.hash(layouter, &items.as_array())
    }

    pub fn assign_vkdata(
        &self,
        layouter: &mut impl Layouter<F>,
        name: &str,
        vk: &VkData,
    ) -> Result<AssignedVk<S>, Error> {
        self.verifier
            .assign_vk_to_fixed(layouter, name, &vk.domain, &vk.cs, vk.transcript_repr)
    }

    fn verify_map_membership_is_one(
        &self,
        layouter: &mut impl Layouter<F>,
        map: &MapGadget,
        key: &AssignedNative<F>,
        one: &AssignedNative<F>,
    ) -> Result<(), Error> {
        let v = map.get(layouter, key)?;
        self.assert_eq(layouter, &v, one)
    }

    fn apply_transaction_effects(
        &self,
        layouter: &mut impl Layouter<F>,
        commit_map: &mut MapGadget,
        null_map: &mut MapGadget,
        items: &ClientPublicItems<AssignedNative<F>>,
        zero: &AssignedNative<F>,
        one: &AssignedNative<F>,
    ) -> Result<(), Error> {
        for commitment in items.commitments() {
            let existing = commit_map.get(layouter, commitment)?;
            self.assert_eq(layouter, &existing, zero)?;
            commit_map.insert(layouter, commitment, one)?;
        }

        for nullifier in items.nullifiers() {
            let existing = null_map.get(layouter, nullifier)?;
            self.assert_eq(layouter, &existing, zero)?;
            null_map.insert(layouter, nullifier, one)?;
        }

        Ok(())
    }
}

////////////////////////////////////////////////////////////////////////////////
// State helpers
////////////////////////////////////////////////////////////////////////////////

pub fn assign_state_array(
    ctx: &AggCtx,
    layouter: &mut impl Layouter<F>,
    fields: [Value<F>; AGG_STATE_WIDTH],
) -> Result<AggStateFields<AssignedNative<F>>, Error> {
    let [
        c_pre,
        c_post,
        n_pre,
        n_post,
        subroot,
        commitment_roots_set_root,
        block_level,
    ] = assign_values(ctx, layouter, fields)?;

    Ok(AggStateFields {
        c_pre,
        c_post,
        n_pre,
        n_post,
        subroot,
        commitment_roots_set_root,
        block_level,
    })
}

pub fn assign_state_value(
    ctx: &AggCtx,
    layouter: &mut impl Layouter<F>,
    state: Value<AggState>,
) -> Result<AggStateFields<AssignedNative<F>>, Error> {
    let fields: Value<[F; AGG_STATE_WIDTH]> = state.map(|s| s.to_fields());
    assign_state_array(ctx, layouter, project_value_array(fields))
}

#[inline]
pub fn base_pi_from_state(state: &AggStateFields<AssignedNative<F>>) -> Vec<AssignedNative<F>> {
    state.as_vec()
}

fn assign_client_items(
    ctx: &AggCtx,
    layouter: &mut impl Layouter<F>,
    items: Value<[F; CLIENT_ITEMS_WIDTH]>,
) -> Result<ClientPublicItems<AssignedNative<F>>, Error> {
    let [root_before, pk_bx, pk_by, new_c1, new_c2, nf1, nf2] =
        assign_values(ctx, layouter, project_value_array(items))?;

    Ok(ClientPublicItems {
        root_before,
        pk_bx,
        pk_by,
        new_c1,
        new_c2,
        nf1,
        nf2,
    })
}

////////////////////////////////////////////////////////////////////////////////
// Map helpers
////////////////////////////////////////////////////////////////////////////////

fn init_map_with_root(
    ctx: &AggCtx,
    layouter: &mut impl Layouter<F>,
    map: Value<Map>,
) -> Result<(MapGadget, AssignedNative<F>), Error> {
    let mut gadget = MapGadget::new(&ctx.scalar, &ctx.poseidon);
    gadget.init(layouter, map)?;
    let root = gadget.succinct_repr();
    Ok((gadget, root))
}

fn init_map(
    ctx: &AggCtx,
    layouter: &mut impl Layouter<F>,
    map: Value<Map>,
) -> Result<MapGadget, Error> {
    let (gadget, _) = init_map_with_root(ctx, layouter, map)?;
    Ok(gadget)
}

////////////////////////////////////////////////////////////////////////////////
// base_step (Leaf layer)
////////////////////////////////////////////////////////////////////////////////

/// Leaf-layer state transition:
/// - initializes rollup commitment/nullifier maps
/// - checks tx roots exist in the historic commitment-roots-set
/// - updates commitment/nullifier maps
/// - computes leaf subroot = H(inst_left, inst_right)
///
/// Returns:
/// - updated aggregation state
/// - left base PI (instance hash)
/// - right base PI (instance hash)
pub fn base_step(
    ctx: &AggCtx,
    layouter: &mut impl Layouter<F>,
    pre_commitment_map: Value<Map>,
    pre_nullifier_map: Value<Map>,
    pre_commitment_roots_set_map: Value<Map>,
    left_items: Value<[F; CLIENT_ITEMS_WIDTH]>,
    right_items: Value<[F; CLIENT_ITEMS_WIDTH]>,
    block_level: Value<F>,
) -> Result<
    (
        AggStateFields<AssignedNative<F>>,
        Vec<AssignedNative<F>>, // left child public inputs (for proof verification)
        Vec<AssignedNative<F>>, // right child public inputs (for proof verification)
    ),
    Error,
> {
    let one = ctx.one(layouter)?;
    let zero = ctx.zero(layouter)?;

    // Batch block number (bound into state; used for swap expiry checks).
    let blk_assigned = ctx.scalar.assign(layouter, block_level)?;

    // Rollup sets.
    let (mut commit_map, c_pre) = init_map_with_root(ctx, layouter, pre_commitment_map)?;
    let (mut null_map, n_pre) = init_map_with_root(ctx, layouter, pre_nullifier_map)?;

    // Historic commitment-roots set (read-only here).
    let (commitment_roots_set_map, commitment_roots_set_root) =
        init_map_with_root(ctx, layouter, pre_commitment_roots_set_map)?;

    // Assign transaction public items.
    let left = assign_client_items(ctx, layouter, left_items)?;
    let right = assign_client_items(ctx, layouter, right_items)?;

    // Membership: tx_root ∈ historic_commitment_roots_set.
    ctx.verify_map_membership_is_one(layouter, &commitment_roots_set_map, &left.root_before, &one)?;
    ctx.verify_map_membership_is_one(
        layouter,
        &commitment_roots_set_map,
        &right.root_before,
        &one,
    )?;

    // Hash client public inputs inside the base circuit (used for leaf subroot / rollup binding).
    let inst_left_hash = ctx.hash_client_instance(layouter, &left)?;
    let inst_right_hash = ctx.hash_client_instance(layouter, &right)?;

    // Apply transaction effects.
    ctx.apply_transaction_effects(layouter, &mut commit_map, &mut null_map, &left, &zero, &one)?;
    ctx.apply_transaction_effects(
        layouter,
        &mut commit_map,
        &mut null_map,
        &right,
        &zero,
        &one,
    )?;

    // Leaf subroot.
    let subroot = ctx.hash2(layouter, &inst_left_hash, &inst_right_hash)?;

    Ok((
        AggStateFields {
            c_pre,
            c_post: commit_map.succinct_repr(),
            n_pre,
            n_post: null_map.succinct_repr(),
            subroot,
            commitment_roots_set_root,
            block_level: blk_assigned,
        },
        left.as_vec(),  // <-- verify client proof with its 7 unhashed public inputs
        right.as_vec(), // <-- verify client proof with its 7 unhashed public inputs
    ))
}

////////////////////////////////////////////////////////////////////////////////
// fold_step (Internal nodes)
////////////////////////////////////////////////////////////////////////////////

/// Internal-node state transition:
/// - assigns both child states
/// - enforces sequential stitching constraints (c/n boundary + commitment_roots_set_root)
/// - computes parent subroot = H(left.subroot, right.subroot)
///
/// Returns:
/// - updated aggregation state
/// - left base PI (child state fields)
/// - right base PI (child state fields)
pub fn fold_step(
    ctx: &AggCtx,
    layouter: &mut impl Layouter<F>,
    left_child_state: [Value<F>; AGG_STATE_WIDTH],
    right_child_state: [Value<F>; AGG_STATE_WIDTH],
) -> Result<
    (
        AggStateFields<AssignedNative<F>>,
        Vec<AssignedNative<F>>,
        Vec<AssignedNative<F>>,
    ),
    Error,
> {
    let left = assign_state_array(ctx, layouter, left_child_state)?;
    let right = assign_state_array(ctx, layouter, right_child_state)?;

    // Stitch constraints.
    ctx.assert_eq(layouter, &left.c_post, &right.c_pre)?;
    ctx.assert_eq(layouter, &left.n_post, &right.n_pre)?;
    ctx.assert_eq(
        layouter,
        &left.commitment_roots_set_root,
        &right.commitment_roots_set_root,
    )?;
    ctx.assert_eq(layouter, &left.block_level, &right.block_level)?;

    // Parent subroot.
    let subroot = ctx.hash2(layouter, &left.subroot, &right.subroot)?;

    let output_state = AggStateFields {
        c_pre: left.c_pre.clone(),
        c_post: right.c_post.clone(),
        n_pre: left.n_pre.clone(),
        n_post: right.n_post.clone(),
        subroot,
        commitment_roots_set_root: left.commitment_roots_set_root.clone(),
        block_level: left.block_level.clone(),
    };

    Ok((
        output_state,
        base_pi_from_state(&left),
        base_pi_from_state(&right),
    ))
}

////////////////////////////////////////////////////////////////////////////////
// wrap_step (Final agg node)
////////////////////////////////////////////////////////////////////////////////

/// Update and bind the historic commitment-roots set.
///
/// Guarantees:
/// - both children were built against the same `pre_root` of the commitment-roots-set
/// - `c_pre` is a member of the set
/// - `c_post` is not yet a member (replay protection)
/// - inserting `c_post` yields `expected_post_root`
pub fn wrap_step(
    ctx: &AggCtx,
    layouter: &mut impl Layouter<F>,
    pre_commitment_roots_set_map: Value<Map>,
    c_pre: &AssignedNative<F>,
    c_post: &AssignedNative<F>,
    left_commitment_roots_set_root: &AssignedNative<F>,
    right_commitment_roots_set_root: &AssignedNative<F>,
    expected_post_commitment_roots_set_root: Value<F>,
) -> Result<(AssignedNative<F>, AssignedNative<F>), Error> {
    let one = ctx.one(layouter)?;
    let zero = ctx.zero(layouter)?;

    let mut commitment_roots_set_map = init_map(ctx, layouter, pre_commitment_roots_set_map)?;
    let pre_commitment_roots_set_root = commitment_roots_set_map.succinct_repr();

    // Bind both children to THIS commitment-roots-set root.
    ctx.assert_eq(
        layouter,
        left_commitment_roots_set_root,
        &pre_commitment_roots_set_root,
    )?;
    ctx.assert_eq(
        layouter,
        right_commitment_roots_set_root,
        &pre_commitment_roots_set_root,
    )?;

    // Membership: c_pre must already be in set.
    let pre_exists = commitment_roots_set_map.get(layouter, c_pre)?;
    ctx.assert_eq(layouter, &pre_exists, &one)?;

    // Replay protection: c_post must be new.
    let post_exists = commitment_roots_set_map.get(layouter, c_post)?;
    ctx.assert_eq(layouter, &post_exists, &zero)?;

    // Insert c_post and bind expected resulting root.
    commitment_roots_set_map.insert(layouter, c_post, &one)?;
    let expected_post_commitment_roots_set_root_assigned = ctx
        .scalar
        .assign(layouter, expected_post_commitment_roots_set_root)?;
    ctx.assert_eq(
        layouter,
        &commitment_roots_set_map.succinct_repr(),
        &expected_post_commitment_roots_set_root_assigned,
    )?;

    Ok((
        pre_commitment_roots_set_root,
        expected_post_commitment_roots_set_root_assigned,
    ))
}

////////////////////////////////////////////////////////////////////////////////
// Recursive partial verification (formerly misnamed "fold_step")
////////////////////////////////////////////////////////////////////////////////

/// Input for recursive partial verification:
/// verify two child proofs and fold their accumulators.
pub struct RecursivePartialVerifyInput<'a> {
    pub assigned_vk: &'a AssignedVk<S>,

    /// If true, children are *client proofs* (leaf children) and do not carry pi-acc public inputs.
    /// If false, children are aggregation proofs and MUST include pi-acc public inputs.
    pub children_are_client_proofs: bool,

    pub fixed_base_names: &'a [String],

    pub left_base_pi: Vec<AssignedNative<F>>,
    pub right_base_pi: Vec<AssignedNative<F>>,

    pub left_proof: Value<Vec<u8>>,
    pub right_proof: Value<Vec<u8>>,

    /// Child pi-acc witnesses (placeholders when `children_are_client_proofs == true`).
    pub left_pi_acc: Value<Accumulator<S>>,
    pub right_pi_acc: Value<Accumulator<S>>,
}

pub struct RecursivePartialVerifyOutput {
    pub next_acc: AssignedAccumulator<S>,
}

/// Assign and collapse a pi-acc witness.
pub fn assign_pi_acc(
    ctx: &AggCtx,
    layouter: &mut impl Layouter<F>,
    fixed_base_names: &[String],
    acc: Value<Accumulator<S>>,
) -> Result<AssignedAccumulator<S>, Error> {
    let mut accumulator = AssignedAccumulator::assign(
        layouter,
        &ctx.curve,
        &ctx.scalar,
        1,
        1,
        &[],
        fixed_base_names,
        acc,
    )?;
    accumulator.collapse(layouter, &ctx.curve, &ctx.scalar)?;
    Ok(accumulator)
}

/// Neutralize pi-acc placeholders when verifying client proofs.
pub fn neutralize_pi_acc_for_client_children(
    ctx: &AggCtx,
    layouter: &mut impl Layouter<F>,
    children_are_client_proofs: bool,
    acc: &mut AssignedAccumulator<S>,
) -> Result<(), Error> {
    if children_are_client_proofs {
        let neutral = ctx.scalar.assign_fixed(layouter, false)?;
        AssignedAccumulator::scale_by_bit(layouter, &ctx.scalar, &neutral, acc)?;
        acc.collapse(layouter, &ctx.curve, &ctx.scalar)?;
    }
    Ok(())
}

pub fn prepare_proof_acc(
    ctx: &AggCtx,
    layouter: &mut impl Layouter<F>,
    assigned_vk: &AssignedVk<S>,
    id_point: IdPoint,
    public_inputs: &[AssignedNative<F>],
    proof: Value<Vec<u8>>,
) -> Result<AssignedAccumulator<S>, Error> {
    let mut proof_acc = ctx.verifier.prepare(
        layouter,
        assigned_vk,
        &[("com_instance", id_point)],
        &[public_inputs],
        proof,
    )?;
    proof_acc.collapse(layouter, &ctx.curve, &ctx.scalar)?;
    Ok(proof_acc)
}

pub fn accumulate_four(
    ctx: &AggCtx,
    layouter: &mut impl Layouter<F>,
    parts: [AssignedAccumulator<S>; 4],
) -> Result<AssignedAccumulator<S>, Error> {
    let mut next = AssignedAccumulator::<S>::accumulate(
        layouter,
        &ctx.verifier,
        &ctx.scalar,
        &ctx.poseidon,
        &parts,
    )?;
    next.collapse(layouter, &ctx.curve, &ctx.scalar)?;
    Ok(next)
}

/// Recursive partial verification:
/// - assigns (and possibly neutralizes) child pi-accs
/// - prepares proof accumulators for both children
/// - folds the four accumulator parts into `next_acc`
pub fn recursive_partial_verify(
    ctx: &AggCtx,
    layouter: &mut impl Layouter<F>,
    input: RecursivePartialVerifyInput<'_>,
) -> Result<RecursivePartialVerifyOutput, Error> {
    let RecursivePartialVerifyInput {
        assigned_vk,
        children_are_client_proofs,
        fixed_base_names,
        left_base_pi,
        right_base_pi,
        left_proof,
        right_proof,
        left_pi_acc,
        right_pi_acc,
    } = input;

    // 1) Assign pi-acc witnesses.
    let mut left_pi_acc_assigned = assign_pi_acc(ctx, layouter, fixed_base_names, left_pi_acc)?;
    let mut right_pi_acc_assigned = assign_pi_acc(ctx, layouter, fixed_base_names, right_pi_acc)?;

    // Leaf children: neutralize placeholders.
    neutralize_pi_acc_for_client_children(
        ctx,
        layouter,
        children_are_client_proofs,
        &mut left_pi_acc_assigned,
    )?;
    neutralize_pi_acc_for_client_children(
        ctx,
        layouter,
        children_are_client_proofs,
        &mut right_pi_acc_assigned,
    )?;

    // 2) Build child public inputs.
    let mut left_child_pi = left_base_pi;
    let mut right_child_pi = right_base_pi;

    if !children_are_client_proofs {
        left_child_pi.extend(
            ctx.verifier
                .as_public_input(layouter, &left_pi_acc_assigned)?,
        );
        right_child_pi.extend(
            ctx.verifier
                .as_public_input(layouter, &right_pi_acc_assigned)?,
        );
    }

    // 3) Prepare proof accumulators.
    let id_point = ctx.id_point(layouter)?;

    let left_proof_acc = prepare_proof_acc(
        ctx,
        layouter,
        assigned_vk,
        id_point.clone(),
        &left_child_pi,
        left_proof,
    )?;
    let right_proof_acc = prepare_proof_acc(
        ctx,
        layouter,
        assigned_vk,
        id_point,
        &right_child_pi,
        right_proof,
    )?;

    // 4) Fold (proof_acc_left, pi_acc_left, proof_acc_right, pi_acc_right).
    let next_acc = accumulate_four(
        ctx,
        layouter,
        [
            left_proof_acc,
            left_pi_acc_assigned,
            right_proof_acc,
            right_pi_acc_assigned,
        ],
    )?;

    Ok(RecursivePartialVerifyOutput { next_acc })
}

////////////////////////////////////////////////////////////////////////////////
// Public input exposure
////////////////////////////////////////////////////////////////////////////////

fn expose_with<L: Layouter<F>>(
    layouter: &mut L,
    values: impl IntoIterator<Item = AssignedNative<F>>,
    mut constrain: impl FnMut(&mut L, &AssignedNative<F>) -> Result<(), Error>,
) -> Result<(), Error> {
    for v in values {
        constrain(layouter, &v)?;
    }
    Ok(())
}

pub fn expose_native(
    ctx: &AggCtx,
    layouter: &mut impl Layouter<F>,
    values: impl IntoIterator<Item = AssignedNative<F>>,
) -> Result<(), Error> {
    expose_with(layouter, values, |l, v| {
        ctx.native.constrain_as_public_input(l, v)
    })
}

/// Expose the canonical aggregation-node outputs:
/// `state fields || folded accumulator PI`.
fn expose_agg_node_outputs(
    ctx: &AggCtx,
    layouter: &mut impl Layouter<F>,
    out_state: AggStateFields<AssignedNative<F>>,
    next_acc: &AssignedAccumulator<S>,
) -> Result<(), Error> {
    expose_native(ctx, layouter, out_state.into_array())?;
    let next_acc_pi = ctx.verifier.as_public_input(layouter, next_acc)?;
    expose_native(ctx, layouter, next_acc_pi)?;
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////
// Shared synthesize helper for 2-child aggregation circuits
////////////////////////////////////////////////////////////////////////////////

fn synthesize_two_child<const K: u32, L: Layouter<F>>(
    config: AggCircuitConfig,
    layouter: &mut L,
    child_vk_name: &str,
    child_vk: &VkData,
    fixed_base_names: &[String],
    children_are_client_proofs: bool,
    left_proof: Value<Vec<u8>>,
    right_proof: Value<Vec<u8>>,
    left_pi_acc: Value<Accumulator<S>>,
    right_pi_acc: Value<Accumulator<S>>,
    step: impl FnOnce(
        &AggCtx,
        &mut L,
    ) -> Result<
        (
            AggStateFields<AssignedNative<F>>,
            Vec<AssignedNative<F>>,
            Vec<AssignedNative<F>>,
        ),
        Error,
    >,
) -> Result<(), Error> {
    let ctx = AggCtx::new(&config, (K as usize).saturating_sub(1));

    // 1) Assign and bind child verification key.
    let assigned_vk = ctx.assign_vkdata(layouter, child_vk_name, child_vk)?;

    // 2) Run the circuit-specific state transition.
    let (out_state, left_base_pi, right_base_pi) = step(&ctx, layouter)?;

    // 3) Recursive partial verification.
    let rpv_out = recursive_partial_verify(
        &ctx,
        layouter,
        RecursivePartialVerifyInput {
            assigned_vk: &assigned_vk,
            children_are_client_proofs,
            fixed_base_names,
            left_base_pi,
            right_base_pi,
            left_proof,
            right_proof,
            left_pi_acc,
            right_pi_acc,
        },
    )?;

    // 4) Public outputs.
    expose_agg_node_outputs(&ctx, layouter, out_state, &rpv_out.next_acc)?;

    // 5) Load shared tables/lookups.
    ctx.load(layouter)
}

////////////////////////////////////////////////////////////////////////////////
// Circuit: base_step (Leaf layer)
////////////////////////////////////////////////////////////////////////////////

/// Leaf-layer aggregation circuit.
///
/// This circuit runs:
/// 1) `base_step` state transition (leaf semantics)
/// 2) recursive partial verification of two client proofs
/// 3) exposes `state || accumulator_PI`
#[derive(Clone, Debug)]
pub struct BaseStepCircuit<const K: u32> {
    pub child_vk: VkData,
    pub child_vk_name: String,

    pub left_items: Value<[F; CLIENT_ITEMS_WIDTH]>,
    pub right_items: Value<[F; CLIENT_ITEMS_WIDTH]>,

    pub pre_commitment_map: Value<Map>,
    pub pre_nullifier_map: Value<Map>,

    pub pre_commitment_roots_set_map: Value<Map>,
    /// Batch L2 rollup block number for this aggregated subtree.
    /// This is carried into `AggState.blk` and is later checked/advanced in the final wrap proof.
    pub block_level: Value<F>,

    pub left_proof: Value<Vec<u8>>,
    pub right_proof: Value<Vec<u8>>,
    pub left_pi_acc: Value<Accumulator<S>>,
    pub right_pi_acc: Value<Accumulator<S>>,
    pub fixed_base_names: Vec<String>,
}

impl<const K: u32> Circuit<F> for BaseStepCircuit<K> {
    type Config = AggCircuitConfig;
    type FloorPlanner = SimpleFloorPlanner;
    type Params = ();

    fn without_witnesses(&self) -> Self {
        Self {
            child_vk: self.child_vk.clone(),
            child_vk_name: self.child_vk_name.clone(),
            left_items: Value::unknown(),
            right_items: Value::unknown(),
            pre_commitment_map: Value::unknown(),
            pre_nullifier_map: Value::unknown(),
            pre_commitment_roots_set_map: Value::unknown(),
            block_level: Value::unknown(),
            left_proof: Value::unknown(),
            right_proof: Value::unknown(),
            left_pi_acc: Value::unknown(),
            right_pi_acc: Value::unknown(),
            fixed_base_names: self.fixed_base_names.clone(),
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
        synthesize_two_child::<K, _>(
            config,
            &mut layouter,
            &self.child_vk_name,
            &self.child_vk,
            &self.fixed_base_names,
            true, // children are client proofs
            self.left_proof.clone(),
            self.right_proof.clone(),
            self.left_pi_acc.clone(),
            self.right_pi_acc.clone(),
            |ctx, layouter| {
                base_step(
                    ctx,
                    layouter,
                    self.pre_commitment_map.clone(),
                    self.pre_nullifier_map.clone(),
                    self.pre_commitment_roots_set_map.clone(),
                    self.left_items.clone(),
                    self.right_items.clone(),
                    self.block_level,
                )
            },
        )
    }
}

pub type LeafAggCircuit = BaseStepCircuit<K_LEAF>;

////////////////////////////////////////////////////////////////////////////////
// Circuit: fold_step (Internal nodes)
////////////////////////////////////////////////////////////////////////////////

/// Internal aggregation circuit.
///
/// This circuit runs:
/// 1) `fold_step` state transition (internal stitching semantics)
/// 2) recursive partial verification of two aggregation proofs
/// 3) exposes `state || accumulator_PI`
#[derive(Clone, Debug)]
pub struct FoldStepCircuit<const K: u32> {
    pub child_vk: VkData,
    pub child_vk_name: String,

    pub left_child_state: [Value<F>; AGG_STATE_WIDTH],
    pub right_child_state: [Value<F>; AGG_STATE_WIDTH],

    pub left_proof: Value<Vec<u8>>,
    pub right_proof: Value<Vec<u8>>,
    pub left_pi_acc: Value<Accumulator<S>>,
    pub right_pi_acc: Value<Accumulator<S>>,
    pub fixed_base_names: Vec<String>,
}

impl<const K: u32> Circuit<F> for FoldStepCircuit<K> {
    type Config = AggCircuitConfig;
    type FloorPlanner = SimpleFloorPlanner;
    type Params = ();

    fn without_witnesses(&self) -> Self {
        Self {
            child_vk: self.child_vk.clone(),
            child_vk_name: self.child_vk_name.clone(),
            left_child_state: core::array::from_fn(|_| Value::unknown()),
            right_child_state: core::array::from_fn(|_| Value::unknown()),
            left_proof: Value::unknown(),
            right_proof: Value::unknown(),
            left_pi_acc: Value::unknown(),
            right_pi_acc: Value::unknown(),
            fixed_base_names: self.fixed_base_names.clone(),
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
        synthesize_two_child::<K, _>(
            config,
            &mut layouter,
            &self.child_vk_name,
            &self.child_vk,
            &self.fixed_base_names,
            false, // children are aggregation proofs
            self.left_proof.clone(),
            self.right_proof.clone(),
            self.left_pi_acc.clone(),
            self.right_pi_acc.clone(),
            |ctx, layouter| fold_step(ctx, layouter, self.left_child_state, self.right_child_state),
        )
    }
}

pub type InternalAggCircuit = FoldStepCircuit<K_INTERNAL>;

////////////////////////////////////////////////////////////////////////////////
// Circuit: wrap_step (Final agg node)
////////////////////////////////////////////////////////////////////////////////

pub type AggAccumulator = Accumulator<S>;

pub fn accumulator_as_public_input(acc: &AggAccumulator) -> Vec<F> {
    AssignedAccumulator::as_public_input(acc)
}

/// Final aggregation circuit.
///
/// This circuit:
/// - exposes the global (c/n) boundary as public input
/// - checks left/right child states stitch and match the declared boundary
/// - exposes the final subroot as public input
/// - runs `wrap_step` to bind + update the historic commitment-roots-set and exposes its pre/post roots
/// - performs recursive partial verification of the two top aggregation proofs
/// - exposes final accumulator PI
#[derive(Clone, Debug)]
pub struct WrapStepCircuit {
    pub child_vk: VkData,
    pub child_vk_name: String,

    pub left_proof: Value<Vec<u8>>,
    pub right_proof: Value<Vec<u8>>,
    pub left_pi_acc: Value<AggAccumulator>,
    pub right_pi_acc: Value<AggAccumulator>,
    pub fixed_base_names: Vec<String>,

    pub left_child_state: Value<AggState>,
    pub right_child_state: Value<AggState>,

    pub agg_state: Value<AggState>,

    pub pre_commitment_roots_set_map: Value<Map>,
    pub post_commitment_roots_set_root: Value<F>,

    /// Global block counter transition, enforced in-circuit:
    /// `blk_post = blk_pre + 1` and `agg.blk = blk_post`.
    pub blk_pre: Value<F>,
    pub blk_post: Value<F>,
}

impl Circuit<F> for WrapStepCircuit {
    type Config = AggCircuitConfig;
    type FloorPlanner = SimpleFloorPlanner;
    type Params = ();

    fn without_witnesses(&self) -> Self {
        Self {
            child_vk: self.child_vk.clone(),
            child_vk_name: self.child_vk_name.clone(),
            left_proof: Value::unknown(),
            right_proof: Value::unknown(),
            left_pi_acc: Value::unknown(),
            right_pi_acc: Value::unknown(),
            fixed_base_names: self.fixed_base_names.clone(),
            left_child_state: Value::unknown(),
            right_child_state: Value::unknown(),
            agg_state: Value::unknown(),
            pre_commitment_roots_set_map: Value::unknown(),
            post_commitment_roots_set_root: Value::unknown(),
            blk_pre: Value::unknown(),
            blk_post: Value::unknown(),
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
        let ctx = AggCtx::new(&config, (AGG_K as usize).saturating_sub(1));

        // --------------------- Assign and expose final aggregation boundary (c/n) ---------------------
        let agg = assign_state_value(&ctx, &mut layouter, self.agg_state.clone())?;

        let one = ctx.one(&mut layouter)?;

        // Public: c_pre, c_post, n_pre, n_post
        expose_native(&ctx, &mut layouter, agg.boundary4())?;

        // --------------------- Block counter transition (public) ---------------------
        let blk_pre = ctx.scalar.assign(&mut layouter, self.blk_pre.clone())?;
        let blk_post = ctx.scalar.assign(&mut layouter, self.blk_post.clone())?;

        // Enforce: blk_post = blk_pre + 1
        let blk_pre_plus_one = ctx.scalar.add(&mut layouter, &blk_pre, &one)?;
        ctx.assert_eq(&mut layouter, &blk_post, &blk_pre_plus_one)?;

        // Enforce: agg.blk = blk_post
        ctx.assert_eq(&mut layouter, &agg.block_level, &blk_post)?;

        // Public: blk_pre, blk_post (so L1 can pin blk_pre to head and accept blk_post as new head)
        expose_native(&ctx, &mut layouter, [blk_pre.clone(), blk_post.clone()])?;

        // --------------------- Assign children states and stitch checks ---------------------
        let left = assign_state_value(&ctx, &mut layouter, self.left_child_state.clone())?;
        let right = assign_state_value(&ctx, &mut layouter, self.right_child_state.clone())?;

        ctx.assert_eq(&mut layouter, &left.block_level, &blk_post)?;
        ctx.assert_eq(&mut layouter, &right.block_level, &blk_post)?;

        // left.c_post == right.c_pre, left.n_post == right.n_pre
        ctx.assert_eq(&mut layouter, &left.c_post, &right.c_pre)?;
        ctx.assert_eq(&mut layouter, &left.n_post, &right.n_pre)?;

        // Children stitch to declared boundary.
        ctx.assert_eq(&mut layouter, &agg.c_pre, &left.c_pre)?;
        ctx.assert_eq(&mut layouter, &agg.c_post, &right.c_post)?;
        ctx.assert_eq(&mut layouter, &agg.n_pre, &left.n_pre)?;
        ctx.assert_eq(&mut layouter, &agg.n_post, &right.n_post)?;

        // Compute + expose final subroot.
        let subroot = ctx.hash2(&mut layouter, &left.subroot, &right.subroot)?;
        // TODO we can avoid witnessing agg.subroot entirety
        ctx.assert_eq(&mut layouter, &agg.subroot, &subroot)?;
        ctx.scalar
            .constrain_as_public_input(&mut layouter, &subroot)?;

        // --------------------- wrap_step: bind and update historic commitment-roots-set ---------------------
        let (pre_commitment_roots_set_root, post_commitment_roots_set_root) = wrap_step(
            &ctx,
            &mut layouter,
            self.pre_commitment_roots_set_map.clone(),
            &agg.c_pre,
            &agg.c_post,
            &left.commitment_roots_set_root,
            &right.commitment_roots_set_root,
            self.post_commitment_roots_set_root.clone(),
        )?;
        // TODO we can avoid witnessing agg.commitment_roots_set_root entirety
        ctx.assert_eq(
            &mut layouter,
            &agg.commitment_roots_set_root,
            &pre_commitment_roots_set_root,
        )?;

        // Public: pre_commitment_roots_set_root, post_commitment_roots_set_root
        expose_native(
            &ctx,
            &mut layouter,
            [
                pre_commitment_roots_set_root,
                post_commitment_roots_set_root,
            ],
        )?;

        // --------------------- Verify top aggregation proofs and fold accumulators ---------------------
        let assigned_vk = ctx.assign_vkdata(&mut layouter, &self.child_vk_name, &self.child_vk)?;

        let rpv_out = recursive_partial_verify(
            &ctx,
            &mut layouter,
            RecursivePartialVerifyInput {
                assigned_vk: &assigned_vk,
                children_are_client_proofs: false,
                fixed_base_names: &self.fixed_base_names,
                left_base_pi: base_pi_from_state(&left),
                right_base_pi: base_pi_from_state(&right),
                left_proof: self.left_proof.clone(),
                right_proof: self.right_proof.clone(),
                left_pi_acc: self.left_pi_acc.clone(),
                right_pi_acc: self.right_pi_acc.clone(),
            },
        )?;

        // Public: final accumulator PI
        let final_acc_pi = ctx
            .verifier
            .as_public_input(&mut layouter, &rpv_out.next_acc)?;
        expose_native(&ctx, &mut layouter, final_acc_pi)?;

        ctx.load(&mut layouter)
    }
}
