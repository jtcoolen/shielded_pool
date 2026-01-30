use ff::Field;
use group::Group;

use midnight_circuits::{
    hash::poseidon::PoseidonChip,
    instructions::{
        AssertionInstructions, AssignmentInstructions, PublicInputInstructions,
        map::MapInstructions,
    },
    types::{AssignedNative, Instantiable},
    verifier::AssignedVk,
};
use midnight_proofs::plonk::Error;
use midnight_proofs::{circuit::Layouter, circuit::Value};

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
    hash::poseidon::{NB_POSEIDON_ADVICE_COLS, NB_POSEIDON_FIXED_COLS, PoseidonConfig},
    instructions::HashInstructions,
    types::{AssignedForeignPoint, ComposableChip},
    verifier::{Accumulator, AssignedAccumulator, BlstrsEmulation, SelfEmulation, VerifierGadget},
};
use midnight_proofs::{
    circuit::SimpleFloorPlanner,
    plonk::{Circuit, ConstraintSystem},
    poly::EvaluationDomain,
};

pub type S = BlstrsEmulation;
type F = <S as SelfEmulation>::F;
type C = <S as SelfEmulation>::C;
type CBase = <C as midnight_circuits::ecc::curves::CircuitCurve>::Base;
type NG = NativeGadget<F, P2RDecompositionChip<F>, NativeChip<F>>;
type Map = midnight_circuits::map::cpu::MapMt<F, PoseidonChip<F>>;

const K_LEAF: u32 = 19;
const K_INTERNAL: u32 = 19;

pub const AGG_K: u32 = K_INTERNAL;

pub(crate) type LeafAggCircuit = AggCircuit<K_LEAF>;
pub(crate) type InternalAggCircuit = AggCircuit<K_INTERNAL>;

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

#[derive(Clone, Debug)]
pub(crate) struct VkData {
    pub(crate) domain: EvaluationDomain<F>,
    pub(crate) cs: ConstraintSystem<F>,
    pub(crate) transcript_repr: F,
}

pub(crate) fn configure_agg_circuit(
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

#[derive(Clone, Debug)]
pub struct AggCircuit<const K: u32> {
    pub(crate) child_vk: VkData,
    pub(crate) child_vk_name: String,

    pub(crate) left_child_state: [Value<F>; AGG_STATE_WIDTH],
    pub(crate) right_child_state: [Value<F>; AGG_STATE_WIDTH],

    pub(crate) left_items: Value<[F; 7]>,
    pub(crate) right_items: Value<[F; 7]>,

    pub(crate) pre_commitment_map: Value<Map>,
    pub(crate) pre_nullifier_map: Value<Map>,

    // NEW: historic commitment-roots set (used to allow “lagging” tx roots)
    pub(crate) pre_commitment_roots_map: Value<Map>,

    pub(crate) left_proof: Value<Vec<u8>>,
    pub(crate) right_proof: Value<Vec<u8>>,
    pub(crate) left_acc: Value<Accumulator<S>>,
    pub(crate) right_acc: Value<Accumulator<S>>,
    pub(crate) fixed_base_names: Vec<String>,
    pub(crate) is_leaf: bool,
}

pub type CurveChip = ForeignEccChip<F, C, C, NG, NG>;
pub type MapGadget = midnight_circuits::map::map_gadget::MapGadget<F, NG, PoseidonChip<F>>;
pub type IdPoint = AssignedForeignPoint<
    midnight_curves::Fq,
    midnight_curves::G1Projective,
    midnight_curves::G1Projective,
>;

////////////////////////////////////////////////////////////////////////////////
// Encodings (struct <-> array)
////////////////////////////////////////////////////////////////////////////////

/// Typed encoding for the 7 public items in client proofs.
///
/// Canonical order: [root_before, pk_bx, pk_by, new_c1, new_c2, nf1, nf2]
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

impl<T> From<[T; 7]> for ClientPublicItems<T> {
    fn from(arr: [T; 7]) -> Self {
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

impl<T> ClientPublicItems<T> {
    pub fn into_array(self) -> [T; 7] {
        [
            self.root_before,
            self.pk_bx,
            self.pk_by,
            self.new_c1,
            self.new_c2,
            self.nf1,
            self.nf2,
        ]
    }
}

impl<T: Clone> ClientPublicItems<T> {
    pub fn as_array(&self) -> [T; 7] {
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

    pub fn commitments(&self) -> [T; 2] {
        [self.new_c1.clone(), self.new_c2.clone()]
    }

    pub fn nullifiers(&self) -> [T; 2] {
        [self.nf1.clone(), self.nf2.clone()]
    }
}

/// Typed encoding for the Agg state public inputs (6 fields).
///
/// Canonical order (must match `AggState::to_fields()`):
/// [c_pre, c_post, n_pre, n_post, subroot, roots_set_root]
#[derive(Clone, Debug)]
pub struct AggStateFields<T> {
    pub c_pre: T,
    pub c_post: T,
    pub n_pre: T,
    pub n_post: T,
    pub subroot: T,
    pub roots_set_root: T,
}

impl From<AggState> for AggStateFields<F> {
    fn from(state: AggState) -> Self {
        Self {
            c_pre: state.c_pre,
            c_post: state.c_post,
            n_pre: state.n_pre,
            n_post: state.n_post,
            subroot: state.subroot,
            roots_set_root: state.roots_set_root,
        }
    }
}

impl<T> From<[T; AGG_STATE_WIDTH]> for AggStateFields<T> {
    fn from(arr: [T; AGG_STATE_WIDTH]) -> Self {
        let [c_pre, c_post, n_pre, n_post, subroot, roots_set_root] = arr;
        Self {
            c_pre,
            c_post,
            n_pre,
            n_post,
            subroot,
            roots_set_root,
        }
    }
}

impl<T> AggStateFields<T> {
    pub fn into_array(self) -> [T; AGG_STATE_WIDTH] {
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

impl<T: Clone> AggStateFields<T> {
    pub fn as_array(&self) -> [T; AGG_STATE_WIDTH] {
        [
            self.c_pre.clone(),
            self.c_post.clone(),
            self.n_pre.clone(),
            self.n_post.clone(),
            self.subroot.clone(),
            self.roots_set_root.clone(),
        ]
    }

    pub fn boundary4(&self) -> [T; 4] {
        [
            self.c_pre.clone(),
            self.c_post.clone(),
            self.n_pre.clone(),
            self.n_post.clone(),
        ]
    }

    pub fn as_vec(&self) -> Vec<T> {
        vec![
            self.c_pre.clone(),
            self.c_post.clone(),
            self.n_pre.clone(),
            self.n_post.clone(),
            self.subroot.clone(),
            self.roots_set_root.clone(),
        ]
    }
}

////////////////////////////////////////////////////////////////////////////////
// Context (chips bundle)
////////////////////////////////////////////////////////////////////////////////

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
    pub fn new(
        cfg: &(
            NativeConfig,
            P2RDecompositionConfig,
            ForeignEccConfig<C>,
            PoseidonConfig<F>,
        ),
        k_minus_1: usize,
    ) -> Self {
        let native = <NativeChip<F> as ComposableChip<F>>::new(&cfg.0, &());
        let core_decomp = P2RDecompositionChip::new(&cfg.1, &k_minus_1);
        let scalar = NativeGadget::new(core_decomp.clone(), native.clone());
        let curve = ForeignEccChip::new(&cfg.2, &scalar, &scalar);
        let poseidon = PoseidonChip::new(&cfg.3, &native);
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
}

////////////////////////////////////////////////////////////////////////////////
// Traits: three-layer binary-tree IVC (base / fold / wrap)
////////////////////////////////////////////////////////////////////////////////

/// Base-layer input for a binary-tree IVC node.
pub enum BaseStepInput {
    /// Leaf node: verify client PIs + update rollup state (commit/nullifier sets) + check tx roots membership.
    Leaf {
        pre_commitment_map: Value<Map>,
        pre_nullifier_map: Value<Map>,
        pre_roots_map: Value<Map>,
        left_items: Value<[F; 7]>,
        right_items: Value<[F; 7]>,
    },
    /// Internal node: stitch child states sequentially + compute parent subroot.
    Internal {
        left_child_state: [Value<F>; AGG_STATE_WIDTH],
        right_child_state: [Value<F>; AGG_STATE_WIDTH],
    },
}

/// Base-layer output: updated state + the per-child “base public inputs” used for proof verification.
pub struct BaseStepOutput {
    pub out_state: AggStateFields<AssignedNative<F>>,
    pub left_base_pi: Vec<AssignedNative<F>>,
    pub right_base_pi: Vec<AssignedNative<F>>,
}

/// Fold-layer input: verify two child proofs and fold their accumulators.
pub struct FoldStepInput<'a> {
    pub assigned_vk: &'a AssignedVk<S>,
    pub is_leaf_child: bool,
    pub fixed_base_names: &'a [String],

    pub left_base_pi: Vec<AssignedNative<F>>,
    pub right_base_pi: Vec<AssignedNative<F>>,

    pub left_proof: Value<Vec<u8>>,
    pub right_proof: Value<Vec<u8>>,
    pub left_pi_acc: Value<Accumulator<S>>,
    pub right_pi_acc: Value<Accumulator<S>>,
}

/// Fold-layer output: next folded accumulator.
pub struct FoldStepOutput {
    pub next_acc: AssignedAccumulator<S>,
}

/// Wrap-layer input: final “roots-set” binding and update.
pub struct WrapStepInput<'a> {
    pub pre_commitment_roots_map: Value<Map>,
    pub c_pre: &'a AssignedNative<F>,
    pub c_post: &'a AssignedNative<F>,
    pub left_roots_set_root: &'a AssignedNative<F>,
    pub right_roots_set_root: &'a AssignedNative<F>,
    pub expected_post_root: Value<F>,
}

/// Wrap-layer output: (pre_roots_root, post_roots_root) exposed publicly.
pub struct WrapStepOutput {
    pub pre_roots_root: AssignedNative<F>,
    pub post_roots_root: AssignedNative<F>,
}

/// Base layer trait: state update + stitching.
pub trait IvcBaseLayer {
    fn base_step<L: Layouter<F>>(
        &self,
        layouter: &mut L,
        input: BaseStepInput,
    ) -> Result<BaseStepOutput, Error>;
}

/// Fold layer trait: verify children + fold accumulators.
pub trait IvcFoldLayer {
    fn fold_step<L: Layouter<F>>(
        &self,
        layouter: &mut L,
        input: FoldStepInput<'_>,
    ) -> Result<FoldStepOutput, Error>;
}

/// Wrap layer trait: final binding / update.
pub trait IvcWrapLayer {
    fn wrap_step<L: Layouter<F>>(
        &self,
        layouter: &mut L,
        input: WrapStepInput<'_>,
    ) -> Result<WrapStepOutput, Error>;
}

impl IvcBaseLayer for AggCtx {
    fn base_step<L: Layouter<F>>(
        &self,
        layouter: &mut L,
        input: BaseStepInput,
    ) -> Result<BaseStepOutput, Error> {
        match input {
            BaseStepInput::Leaf {
                pre_commitment_map,
                pre_nullifier_map,
                pre_roots_map,
                left_items,
                right_items,
            } => {
                let (out_state, left_pi, right_pi) = ivc_base_step_leaf(
                    self,
                    layouter,
                    pre_commitment_map,
                    pre_nullifier_map,
                    pre_roots_map,
                    left_items,
                    right_items,
                )?;
                Ok(BaseStepOutput {
                    out_state,
                    left_base_pi: left_pi,
                    right_base_pi: right_pi,
                })
            }
            BaseStepInput::Internal {
                left_child_state,
                right_child_state,
            } => {
                let (out_state, left_pi, right_pi) =
                    ivc_base_step_internal(self, layouter, left_child_state, right_child_state)?;
                Ok(BaseStepOutput {
                    out_state,
                    left_base_pi: left_pi,
                    right_base_pi: right_pi,
                })
            }
        }
    }
}

impl IvcFoldLayer for AggCtx {
    fn fold_step<L: Layouter<F>>(
        &self,
        layouter: &mut L,
        input: FoldStepInput<'_>,
    ) -> Result<FoldStepOutput, Error> {
        let FoldStepInput {
            assigned_vk,
            is_leaf_child,
            fixed_base_names,
            left_base_pi,
            right_base_pi,
            left_proof,
            right_proof,
            left_pi_acc,
            right_pi_acc,
        } = input;

        // 1) Assign (and possibly neutralize) the provided pi-acc placeholders (leaf case).
        let mut left_pi_acc_assigned =
            assign_pi_acc(self, layouter, fixed_base_names, left_pi_acc)?;
        let mut right_pi_acc_assigned =
            assign_pi_acc(self, layouter, fixed_base_names, right_pi_acc)?;

        neutralize_pi_acc_if_leaf(self, layouter, is_leaf_child, &mut left_pi_acc_assigned)?;
        neutralize_pi_acc_if_leaf(self, layouter, is_leaf_child, &mut right_pi_acc_assigned)?;

        // 2) Build child public inputs for verification.
        let left_child_pi = build_child_public_inputs(
            self,
            layouter,
            is_leaf_child,
            left_base_pi,
            Some(&left_pi_acc_assigned),
        )?;
        let right_child_pi = build_child_public_inputs(
            self,
            layouter,
            is_leaf_child,
            right_base_pi,
            Some(&right_pi_acc_assigned),
        )?;

        // 3) Prepare proof accumulators and fold them.
        let id_point = self.id_point(layouter)?;

        let left_proof_acc = prepare_proof_acc(
            self,
            layouter,
            assigned_vk,
            id_point.clone(),
            &left_child_pi,
            left_proof,
        )?;
        let right_proof_acc = prepare_proof_acc(
            self,
            layouter,
            assigned_vk,
            id_point,
            &right_child_pi,
            right_proof,
        )?;

        let next_acc = fold_step_accumulate(
            self,
            layouter,
            [
                left_proof_acc,
                left_pi_acc_assigned,
                right_proof_acc,
                right_pi_acc_assigned,
            ],
        )?;

        Ok(FoldStepOutput { next_acc })
    }
}

impl IvcWrapLayer for AggCtx {
    fn wrap_step<L: Layouter<F>>(
        &self,
        layouter: &mut L,
        input: WrapStepInput<'_>,
    ) -> Result<WrapStepOutput, Error> {
        let WrapStepInput {
            pre_commitment_roots_map,
            c_pre,
            c_post,
            left_roots_set_root,
            right_roots_set_root,
            expected_post_root,
        } = input;

        let (pre_root, post_root) = wrap_step_update_roots_set(
            self,
            layouter,
            pre_commitment_roots_map,
            c_pre,
            c_post,
            left_roots_set_root,
            right_roots_set_root,
            expected_post_root,
        )?;

        Ok(WrapStepOutput {
            pre_roots_root: pre_root,
            post_roots_root: post_root,
        })
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
    let [c_pre, c_post, n_pre, n_post, subroot, roots_set_root] = fields;

    Ok(AggStateFields {
        c_pre: ctx.scalar.assign(layouter, c_pre)?,
        c_post: ctx.scalar.assign(layouter, c_post)?,
        n_pre: ctx.scalar.assign(layouter, n_pre)?,
        n_post: ctx.scalar.assign(layouter, n_post)?,
        subroot: ctx.scalar.assign(layouter, subroot)?,
        roots_set_root: ctx.scalar.assign(layouter, roots_set_root)?,
    })
}

pub fn assign_state_value(
    ctx: &AggCtx,
    layouter: &mut impl Layouter<F>,
    state: Value<AggState>,
) -> Result<AggStateFields<AssignedNative<F>>, Error> {
    let state_fields = state.map(AggStateFields::<F>::from);

    Ok(AggStateFields {
        c_pre: ctx
            .scalar
            .assign(layouter, state_fields.as_ref().map(|s| s.c_pre))?,
        c_post: ctx
            .scalar
            .assign(layouter, state_fields.as_ref().map(|s| s.c_post))?,
        n_pre: ctx
            .scalar
            .assign(layouter, state_fields.as_ref().map(|s| s.n_pre))?,
        n_post: ctx
            .scalar
            .assign(layouter, state_fields.as_ref().map(|s| s.n_post))?,
        subroot: ctx
            .scalar
            .assign(layouter, state_fields.as_ref().map(|s| s.subroot))?,
        roots_set_root: ctx
            .scalar
            .assign(layouter, state_fields.as_ref().map(|s| s.roots_set_root))?,
    })
}

pub fn base_pi_from_state(state: &AggStateFields<AssignedNative<F>>) -> Vec<AssignedNative<F>> {
    state.as_vec()
}

fn assign_client_items(
    ctx: &AggCtx,
    layouter: &mut impl Layouter<F>,
    items: Value<[F; 7]>,
) -> Result<ClientPublicItems<AssignedNative<F>>, Error> {
    let typed_items = items.map(ClientPublicItems::<F>::from);

    Ok(ClientPublicItems {
        root_before: ctx
            .scalar
            .assign(layouter, typed_items.as_ref().map(|i| i.root_before))?,
        pk_bx: ctx
            .scalar
            .assign(layouter, typed_items.as_ref().map(|i| i.pk_bx))?,
        pk_by: ctx
            .scalar
            .assign(layouter, typed_items.as_ref().map(|i| i.pk_by))?,
        new_c1: ctx
            .scalar
            .assign(layouter, typed_items.as_ref().map(|i| i.new_c1))?,
        new_c2: ctx
            .scalar
            .assign(layouter, typed_items.as_ref().map(|i| i.new_c2))?,
        nf1: ctx
            .scalar
            .assign(layouter, typed_items.as_ref().map(|i| i.nf1))?,
        nf2: ctx
            .scalar
            .assign(layouter, typed_items.as_ref().map(|i| i.nf2))?,
    })
}

////////////////////////////////////////////////////////////////////////////////
// Map helpers
////////////////////////////////////////////////////////////////////////////////

fn init_map(
    ctx: &AggCtx,
    layouter: &mut impl Layouter<F>,
    map: Value<Map>,
) -> Result<MapGadget, Error> {
    let mut gadget = MapGadget::new(&ctx.scalar, &ctx.poseidon);
    gadget.init(layouter, map)?;
    Ok(gadget)
}

////////////////////////////////////////////////////////////////////////////////
// IVC base step (leaf + internal)
////////////////////////////////////////////////////////////////////////////////

/// Leaf base step: Merkle map updates (commitment/nullifier) + historic roots membership checks.
pub fn ivc_base_step_leaf(
    ctx: &AggCtx,
    layouter: &mut impl Layouter<F>,
    pre_commitment_map: Value<Map>,
    pre_nullifier_map: Value<Map>,
    pre_roots_map: Value<Map>,
    left_items: Value<[F; 7]>,
    right_items: Value<[F; 7]>,
) -> Result<
    (
        AggStateFields<AssignedNative<F>>,
        Vec<AssignedNative<F>>, // left base PI (instance hash)
        Vec<AssignedNative<F>>, // right base PI (instance hash)
    ),
    Error,
> {
    let one = ctx.one(layouter)?;
    let zero = ctx.zero(layouter)?;

    // Initialize current rollup sets
    let mut commit_map = init_map(ctx, layouter, pre_commitment_map)?;
    let c_pre = commit_map.succinct_repr();

    let mut null_map = init_map(ctx, layouter, pre_nullifier_map)?;
    let n_pre = null_map.succinct_repr();

    // Initialize historic roots set
    let roots_map = init_map(ctx, layouter, pre_roots_map)?;
    let roots_set_root = roots_map.succinct_repr();

    // Assign transaction public items (typed)
    let left = assign_client_items(ctx, layouter, left_items)?;
    let right = assign_client_items(ctx, layouter, right_items)?;

    // Verify membership: tx_root ∈ historic_roots_set
    verify_root_membership(ctx, layouter, &roots_map, &left.root_before, &one)?;
    verify_root_membership(ctx, layouter, &roots_map, &right.root_before, &one)?;

    // Compute client instance hashes (hash all 7 fields in canonical order)
    let left_arr = left.as_array();
    let right_arr = right.as_array();
    let inst_left = ctx.poseidon.hash(layouter, &left_arr)?;
    let inst_right = ctx.poseidon.hash(layouter, &right_arr)?;

    // Apply transaction effects to rollup sets
    apply_transaction_effects(
        ctx,
        layouter,
        &mut commit_map,
        &mut null_map,
        &left,
        &zero,
        &one,
    )?;
    apply_transaction_effects(
        ctx,
        layouter,
        &mut commit_map,
        &mut null_map,
        &right,
        &zero,
        &one,
    )?;

    let c_post = commit_map.succinct_repr();
    let n_post = null_map.succinct_repr();

    // Aggregate subtree root for this leaf aggregation node
    let subroot = ctx
        .poseidon
        .hash(layouter, &[inst_left.clone(), inst_right.clone()])?;

    Ok((
        AggStateFields {
            c_pre,
            c_post,
            n_pre,
            n_post,
            subroot,
            roots_set_root,
        },
        vec![inst_left],
        vec![inst_right],
    ))
}

/// Helper: Verify that a root exists in the historic roots set.
fn verify_root_membership(
    ctx: &AggCtx,
    layouter: &mut impl Layouter<F>,
    roots_map: &MapGadget,
    root: &AssignedNative<F>,
    one: &AssignedNative<F>,
) -> Result<(), Error> {
    let membership = roots_map.get(layouter, root)?;
    ctx.scalar.assert_equal(layouter, &membership, one)
}

/// Helper: Apply transaction effects (insert commitments, check and mark nullifiers).
fn apply_transaction_effects(
    ctx: &AggCtx,
    layouter: &mut impl Layouter<F>,
    commit_map: &mut MapGadget,
    null_map: &mut MapGadget,
    items: &ClientPublicItems<AssignedNative<F>>,
    zero: &AssignedNative<F>,
    one: &AssignedNative<F>,
) -> Result<(), Error> {
    // Insert commitments
    for commitment in &items.commitments() {
        commit_map.insert(layouter, commitment, one)?;
    }

    // Verify nullifiers are new, then mark them as spent
    for nullifier in &items.nullifiers() {
        let existing = null_map.get(layouter, nullifier)?;
        ctx.scalar.assert_equal(layouter, &existing, zero)?;
        null_map.insert(layouter, nullifier, one)?;
    }

    Ok(())
}

/// Internal base step: stitch children states (sequential application) + compute parent subroot.
pub fn ivc_base_step_internal(
    ctx: &AggCtx,
    layouter: &mut impl Layouter<F>,
    left_child_state: [Value<F>; AGG_STATE_WIDTH],
    right_child_state: [Value<F>; AGG_STATE_WIDTH],
) -> Result<
    (
        AggStateFields<AssignedNative<F>>,
        Vec<AssignedNative<F>>, // left base PI (child state fields)
        Vec<AssignedNative<F>>, // right base PI (child state fields)
    ),
    Error,
> {
    let left = assign_state_array(ctx, layouter, left_child_state)?;
    let right = assign_state_array(ctx, layouter, right_child_state)?;

    // Verify state transitions match across child boundaries
    ctx.scalar
        .assert_equal(layouter, &left.c_post, &right.c_pre)?;
    ctx.scalar
        .assert_equal(layouter, &left.n_post, &right.n_pre)?;
    ctx.scalar
        .assert_equal(layouter, &left.roots_set_root, &right.roots_set_root)?;

    // Compute parent subroot by hashing child subroots
    let subroot = ctx
        .poseidon
        .hash(layouter, &[left.subroot.clone(), right.subroot.clone()])?;

    let output_state = AggStateFields {
        c_pre: left.c_pre.clone(),
        c_post: right.c_post.clone(),
        n_pre: left.n_pre.clone(),
        n_post: right.n_post.clone(),
        subroot,
        roots_set_root: left.roots_set_root.clone(),
    };

    Ok((
        output_state,
        base_pi_from_state(&left),
        base_pi_from_state(&right),
    ))
}

////////////////////////////////////////////////////////////////////////////////
// IVC fold step helpers
////////////////////////////////////////////////////////////////////////////////

/// Assign and (optionally) neutralize leaf pi-acc; return assigned accumulators.
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

pub fn neutralize_pi_acc_if_leaf(
    ctx: &AggCtx,
    layouter: &mut impl Layouter<F>,
    is_leaf: bool,
    acc: &mut AssignedAccumulator<S>,
) -> Result<(), Error> {
    if is_leaf {
        let neutral = ctx.scalar.assign_fixed(layouter, false)?;
        AssignedAccumulator::scale_by_bit(layouter, &ctx.scalar, &neutral, acc)?;
        acc.collapse(layouter, &ctx.curve, &ctx.scalar)?;
    }
    Ok(())
}

/// Build the public inputs used to verify a child proof.
///
/// - For leaf children: just base_pi (client instance hash)
/// - For internal children: base_pi (child agg state fields) || child_pi_acc public input
pub fn build_child_public_inputs(
    ctx: &AggCtx,
    layouter: &mut impl Layouter<F>,
    is_leaf_child: bool,
    mut base_pi: Vec<AssignedNative<F>>,
    child_pi_acc: Option<&AssignedAccumulator<S>>,
) -> Result<Vec<AssignedNative<F>>, Error> {
    if !is_leaf_child {
        let acc = child_pi_acc.expect("Internal child requires pi_acc");
        base_pi.extend(ctx.verifier.as_public_input(layouter, acc)?);
    }
    Ok(base_pi)
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

pub fn fold_step_accumulate(
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

////////////////////////////////////////////////////////////////////////////////
// Public input exposure
////////////////////////////////////////////////////////////////////////////////

pub fn expose_native(
    ctx: &AggCtx,
    layouter: &mut impl Layouter<F>,
    values: impl IntoIterator<Item = AssignedNative<F>>,
) -> Result<(), Error> {
    for value in values {
        ctx.native.constrain_as_public_input(layouter, &value)?;
    }
    Ok(())
}

pub fn expose_scalar(
    ctx: &AggCtx,
    layouter: &mut impl Layouter<F>,
    values: impl IntoIterator<Item = AssignedNative<F>>,
) -> Result<(), Error> {
    for value in values {
        ctx.scalar.constrain_as_public_input(layouter, &value)?;
    }
    Ok(())
}

/// Expose the canonical Agg node public outputs:
/// state fields || folded accumulator PI.
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
// Final wrap: roots-set update
////////////////////////////////////////////////////////////////////////////////

pub fn wrap_step_update_roots_set(
    ctx: &AggCtx,
    layouter: &mut impl Layouter<F>,
    pre_commitment_roots_map: Value<Map>,
    c_pre: &AssignedNative<F>,
    c_post: &AssignedNative<F>,
    left_roots_set_root: &AssignedNative<F>,
    right_roots_set_root: &AssignedNative<F>,
    expected_post_root: Value<F>,
) -> Result<(AssignedNative<F>, AssignedNative<F>), Error> {
    let one = ctx.one(layouter)?;
    let zero = ctx.zero(layouter)?;

    let mut roots_map = init_map(ctx, layouter, pre_commitment_roots_map)?;
    let pre_root = roots_map.succinct_repr();

    // Verify child aggregation proofs are bound to THIS historic-roots-set root
    ctx.scalar
        .assert_equal(layouter, left_roots_set_root, &pre_root)?;
    ctx.scalar
        .assert_equal(layouter, right_roots_set_root, &pre_root)?;

    // Verify membership: c_pre must already be in set
    let pre_exists = roots_map.get(layouter, c_pre)?;
    ctx.scalar.assert_equal(layouter, &pre_exists, &one)?;

    // Replay protection: c_post must be new
    let post_exists = roots_map.get(layouter, c_post)?;
    ctx.scalar.assert_equal(layouter, &post_exists, &zero)?;

    // Insert c_post and bind expected resulting root
    roots_map.insert(layouter, c_post, &one)?;
    let post_root_expected = ctx.scalar.assign(layouter, expected_post_root)?;
    ctx.scalar
        .assert_equal(layouter, &roots_map.succinct_repr(), &post_root_expected)?;

    Ok((pre_root, post_root_expected))
}

////////////////////////////////////////////////////////////////////////////////
// Refactored AggCircuit (now orchestrates base/fold via traits)
////////////////////////////////////////////////////////////////////////////////

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
            left_child_state: core::array::from_fn(|_| Value::unknown()),
            right_child_state: core::array::from_fn(|_| Value::unknown()),
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
        let ctx = AggCtx::new(&config, (K as usize).saturating_sub(1));

        // 1) Assign and bind child verification key.
        let child_vk_val: AssignedNative<F> = ctx
            .native
            .assign_fixed(&mut layouter, self.child_vk.transcript_repr)?;
        let assigned_vk = ctx.verifier.assign_vk(
            &self.child_vk_name,
            &self.child_vk.domain,
            &self.child_vk.cs,
            child_vk_val,
        )?;

        // 2) Base layer (leaf update or internal stitch).
        let base_out = ctx.base_step(
            &mut layouter,
            if self.is_leaf {
                BaseStepInput::Leaf {
                    pre_commitment_map: self.pre_commitment_map.clone(),
                    pre_nullifier_map: self.pre_nullifier_map.clone(),
                    pre_roots_map: self.pre_commitment_roots_map.clone(),
                    left_items: self.left_items.clone(),
                    right_items: self.right_items.clone(),
                }
            } else {
                BaseStepInput::Internal {
                    left_child_state: self.left_child_state,
                    right_child_state: self.right_child_state,
                }
            },
        )?;

        // 3) Fold layer (verify children + fold accumulators).
        let fold_out = ctx.fold_step(
            &mut layouter,
            FoldStepInput {
                assigned_vk: &assigned_vk,
                is_leaf_child: self.is_leaf,
                fixed_base_names: &self.fixed_base_names,
                left_base_pi: base_out.left_base_pi,
                right_base_pi: base_out.right_base_pi,
                left_proof: self.left_proof.clone(),
                right_proof: self.right_proof.clone(),
                left_pi_acc: self.left_acc.clone(),
                right_pi_acc: self.right_acc.clone(),
            },
        )?;

        // 4) Public outputs for the node.
        expose_agg_node_outputs(&ctx, &mut layouter, base_out.out_state, &fold_out.next_acc)?;

        // 5) Finalize shared lookups.
        ctx.load(&mut layouter)
    }
}

////////////////////////////////////////////////////////////////////////////////
// Refactored FinalAggCircuit
////////////////////////////////////////////////////////////////////////////////

pub type AggAccumulator = Accumulator<S>;
pub fn accumulator_as_public_input(acc: &AggAccumulator) -> Vec<F> {
    AssignedAccumulator::as_public_input(acc)
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
        let ctx = AggCtx::new(&config, (AGG_K as usize).saturating_sub(1));

        // --------------------- Assign and expose final aggregation state (c/n boundaries) ---------------------
        let agg = assign_state_value(&ctx, &mut layouter, self.agg_state.clone())?;

        // Public: c_pre, c_post, n_pre, n_post
        expose_scalar(&ctx, &mut layouter, agg.boundary4())?;

        // --------------------- Assign children states ---------------------
        let left = assign_state_value(&ctx, &mut layouter, self.left_child_state.clone())?;
        let right = assign_state_value(&ctx, &mut layouter, self.right_child_state.clone())?;

        // Verify child boundaries stitch together: left.c_post == right.c_pre, left.n_post == right.n_pre
        ctx.scalar
            .assert_equal(&mut layouter, &left.c_post, &right.c_pre)?;
        ctx.scalar
            .assert_equal(&mut layouter, &left.n_post, &right.n_pre)?;

        // Verify children stitch to declared aggregation boundary
        ctx.scalar
            .assert_equal(&mut layouter, &agg.c_pre, &left.c_pre)?;
        ctx.scalar
            .assert_equal(&mut layouter, &agg.c_post, &right.c_post)?;
        ctx.scalar
            .assert_equal(&mut layouter, &agg.n_pre, &left.n_pre)?;
        ctx.scalar
            .assert_equal(&mut layouter, &agg.n_post, &right.n_post)?;

        // Compute and expose final subroot
        let subroot = ctx.poseidon.hash(
            &mut layouter,
            &[left.subroot.clone(), right.subroot.clone()],
        )?;
        ctx.scalar
            .constrain_as_public_input(&mut layouter, &subroot)?;

        // --------------------- Wrap step: bind and update historic-roots-set ---------------------
        let WrapStepOutput {
            pre_roots_root,
            post_roots_root,
        } = ctx.wrap_step(
            &mut layouter,
            WrapStepInput {
                pre_commitment_roots_map: self.pre_commitment_roots_map.clone(),
                c_pre: &agg.c_pre,
                c_post: &agg.c_post,
                left_roots_set_root: &left.roots_set_root,
                right_roots_set_root: &right.roots_set_root,
                expected_post_root: self.post_commitment_roots_root.clone(),
            },
        )?;

        // Public: pre_roots_root, post_roots_root
        expose_scalar(&ctx, &mut layouter, [pre_roots_root, post_roots_root])?;

        // --------------------- Verify top aggregation proofs and fold accumulators (wrap fold) ---------------------
        let vk_val: AssignedNative<F> = ctx.native.assign_fixed(&mut layouter, self.child_vk.2)?;
        let assigned_vk = ctx.verifier.assign_vk(
            &self.child_vk_name,
            &self.child_vk.0,
            &self.child_vk.1,
            vk_val,
        )?;

        let left_pi_acc = assign_pi_acc(
            &ctx,
            &mut layouter,
            &self.fixed_base_names,
            self.left_pi_acc.clone(),
        )?;
        let right_pi_acc = assign_pi_acc(
            &ctx,
            &mut layouter,
            &self.fixed_base_names,
            self.right_pi_acc.clone(),
        )?;

        // Child public inputs = child state fields || child pi_acc PI
        let left_pi = build_child_public_inputs(
            &ctx,
            &mut layouter,
            false,
            base_pi_from_state(&left),
            Some(&left_pi_acc),
        )?;
        let right_pi = build_child_public_inputs(
            &ctx,
            &mut layouter,
            false,
            base_pi_from_state(&right),
            Some(&right_pi_acc),
        )?;

        let id_point = ctx.id_point(&mut layouter)?;

        let left_proof_acc = prepare_proof_acc(
            &ctx,
            &mut layouter,
            &assigned_vk,
            id_point.clone(),
            &left_pi,
            self.left_proof.clone(),
        )?;
        let right_proof_acc = prepare_proof_acc(
            &ctx,
            &mut layouter,
            &assigned_vk,
            id_point,
            &right_pi,
            self.right_proof.clone(),
        )?;

        let final_acc = fold_step_accumulate(
            &ctx,
            &mut layouter,
            [left_proof_acc, left_pi_acc, right_proof_acc, right_pi_acc],
        )?;

        // Public: final accumulator PI
        let final_acc_pi = ctx.verifier.as_public_input(&mut layouter, &final_acc)?;
        expose_scalar(&ctx, &mut layouter, final_acc_pi)?;

        ctx.load(&mut layouter)
    }
}
