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
        ArithInstructions, AssertionInstructions, AssignmentInstructions, ControlFlowInstructions,
        ConversionInstructions, DecompositionInstructions, PublicInputInstructions, hash::HashCPU,
    },
    testing_utils::plonk_api::filecoin_srs,
    types::{AssignedBit, AssignedNative, AssignedNativePoint, Instantiable},
};

use midnight_curves::{Fq as F, Fr as JubjubScalar, JubjubExtended as Jubjub, JubjubSubgroup};
use midnight_proofs::{
    circuit::{Layouter, Value},
    plonk::Error,
};
use num_bigint::BigUint;
use rand::{Rng, SeedableRng, rngs::OsRng};
use rand_chacha::ChaCha8Rng;

const TREE_HEIGHT: usize = 64;
const UTXO_COMMIT_TAG: u64 = 0x0001;
const UTXO_NULLIFY_TAG: u64 = 0x0002;
const AMOUNT_BITS: u32 = 128; // 128-bit integers for amounts
const AMOUNT_GEN_BITS: u32 = 120; // generate up to 120 bits to avoid u128 overflow on sums

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

#[derive(Default)]
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
    // Single public input: Poseidon hash of (root, pk_x, pk_y, old_c1, old_c2, new_c1, new_c2, nf1, nf2)
    type Instance = F;

    // Witness unchanged (includes everything needed to recompute the values that used to be public)
    type Witness = (
        MerklePath<F>,
        MerklePath<F>,
        JubjubScalar,
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
        let mp1_val = witness.clone().map(|(mp1, _, _, _, _, _, _, _, _)| mp1);
        let mp2_val = witness.clone().map(|(_, mp2, _, _, _, _, _, _, _)| mp2);
        let sk_val = witness.clone().map(|(_, _, sk, _, _, _, _, _, _)| sk);
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
        let pk_fields = std_lib.jubjub().as_public_input(layouter, &pk_sender)?;
        let (pk_sx, pk_sy) = (pk_fields[0].clone(), pk_fields[1].clone());

        // Assign each UTXO's fields exactly once
        let old1_asg = assign_utxo(std_lib, layouter, &old1_val)?;
        let old2_asg = assign_utxo(std_lib, layouter, &old2_val)?;
        let new1_asg = assign_utxo(std_lib, layouter, &new1_val)?;
        let new2_asg = assign_utxo(std_lib, layouter, &new2_val)?;

        // old commitments (must match sender pk)
        let old_c1 = compute_commitment_from_parts(std_lib, layouter, &old1_asg, &pk_sx, &pk_sy)?;
        let old_c2 = compute_commitment_from_parts(std_lib, layouter, &old2_asg, &pk_sx, &pk_sy)?;

        // Verify Merkle proofs and check roots match
        let root1 = compute_merkle_root(std_lib, layouter, mp1_val, old_c1.clone())?;
        let root2 = compute_merkle_root(std_lib, layouter, mp2_val, old_c2.clone())?;
        std_lib.assert_equal(layouter, &root1, &root2)?;

        // Nullifiers (bound to sender pk)
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

        // ---- Single public input: Poseidon hash of the original public inputs ----
        // Sponge the nine values in groups of 3 using the same 3-arity Poseidon as elsewhere.
        let acc1 = std_lib.poseidon(layouter, &[root1.clone(), pk_sx.clone(), pk_sy.clone()])?;
        let acc2 = std_lib.poseidon(layouter, &[acc1, old_c1.clone(), old_c2.clone()])?;
        let acc3 = std_lib.poseidon(layouter, &[acc2, new_c1.clone(), new_c2.clone()])?;
        let instance_hash = std_lib.poseidon(layouter, &[acc3, nf1.clone(), nf2.clone()])?;

        // Expose only this hash as the single public input
        std_lib.constrain_as_public_input(layouter, &instance_hash)?;
        // -------------------------------------------------------------------------

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

// Poseidon sponge (3-arity) over the nine original public inputs:
// (root, pk_x, pk_y) -> acc1
// (acc1, old_c1, old_c2) -> acc2
// (acc2, new_c1, new_c2) -> acc3
// (acc3, nf1,  nf2)  -> final hash
fn host_instance_hash(items: [F; 9]) -> F {
    let acc1 = <PoseidonChip<F> as HashCPU<F, F>>::hash(&[items[0], items[1], items[2]]);
    let acc2 = <PoseidonChip<F> as HashCPU<F, F>>::hash(&[acc1, items[3], items[4]]);
    let acc3 = <PoseidonChip<F> as HashCPU<F, F>>::hash(&[acc2, items[5], items[6]]);
    <PoseidonChip<F> as HashCPU<F, F>>::hash(&[acc3, items[7], items[8]])
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
    const K: u32 = 13;
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
    let mut choose_sender = |rng: &mut ChaCha8Rng, accs: &mut [Account]| -> Option<usize> {
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

    for t in 0..NUM_TRANSFERS {
        let sender_idx = match choose_sender(&mut rng, &mut accounts) {
            Some(i) => i,
            None => {
                println!("[{}] no account has two spendable notes; stopping.", t);
                break;
            }
        };

        // Pick two distinct unspent notes from sender
        let (i_old1, i_old2) = {
            let unspent: Vec<usize> = accounts[sender_idx]
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

        // Sender & inputs
        let sender = accounts[sender_idx].clone();
        let old1 = accounts[sender_idx].wallet[i_old1].clone();
        let old2 = accounts[sender_idx].wallet[i_old2].clone();

        // Build membership proofs against current root
        let root_before = tree.root();
        let mp1 = tree.merkle_path(old1.idx);
        let mp2 = tree.merkle_path(old2.idx);
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
            accounts[r1].pk_x,
            accounts[r1].pk_y,
            new1.randomness,
        );
        let new2_commit = host_commit(
            new2.asset_id,
            new2.amount,
            accounts[r2].pk_x,
            accounts[r2].pk_y,
            new2.randomness,
        );

        // Nullifiers (bound to sender)
        let nf1 = host_nullify(old1.commit, sender.pk_x, sender.pk_y);
        let nf2 = host_nullify(old2.commit, sender.pk_x, sender.pk_y);

        // Compute single public instance hash (Poseidon sponge over original public inputs)
        let instance: F = host_instance_hash([
            root_before,
            sender.pk_x,
            sender.pk_y,
            old1.commit,
            old2.commit,
            new1_commit,
            new2_commit,
            nf1,
            nf2,
        ]);

        // Witness carries recipient keys for outputs (unchanged)
        let witness = (
            mp1,
            mp2,
            sender.sk,
            old1.utxo.clone(),
            old2.utxo.clone(),
            new1.clone(),
            new2.clone(),
            (accounts[r1].pk_x, accounts[r1].pk_y),
            (accounts[r2].pk_x, accounts[r2].pk_y),
        );

        // Prove + verify
        let now = Instant::now();
        let proof = compact_std_lib::prove::<Spend2Output2, PoseidonState<F>>(
            &srs, &pk, &relation, &instance, witness, OsRng,
        )
        .expect("Proof generation failed");
        println!("[{}] proof gen: {:?}", t, now.elapsed());

        let now = Instant::now();
        assert!(
            compact_std_lib::verify::<Spend2Output2, PoseidonState<F>>(
                &srs.verifier_params(),
                &vk,
                &instance,
                None,
                &proof
            )
            .is_ok()
        );
        println!("[{}] verify: {:?}", t, now.elapsed());

        // Apply to tree
        let (idx_new1, idx_new2) = tree.apply_transfer([nf1, nf2], [new1_commit, new2_commit]);

        // Mark inputs spent and credit recipients
        accounts[sender_idx].wallet[i_old1].spent = true;
        accounts[sender_idx].wallet[i_old2].spent = true;

        accounts[r1].wallet.push(Note {
            idx: idx_new1,
            utxo: new1,
            commit: new1_commit,
            spent: false,
        });
        accounts[r2].wallet.push(Note {
            idx: idx_new2,
            utxo: new2,
            commit: new2_commit,
            spent: false,
        });

        // quick inclusion checks
        let root_after = tree.root();
        let mp_out1 = tree.merkle_path(idx_new1);
        let mp_out2 = tree.merkle_path(idx_new2);
        assert_eq!(root_after, mp_out1.compute_root());
        assert_eq!(root_after, mp_out2.compute_root());

        println!("[{}] root updated: {:?}", t, root_after);
    }

    println!("Final root: {:?}", tree.root());

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
