// SPDX-License-Identifier: CC0-1.0

use ff::PrimeField;

use midnight_proofs::transcript::{Hashable, Sampleable, TranscriptHash};

use sha3::{Digest, Keccak256};
use std::{io, io::Read};

use ff::FromUniformBytes;
use group::GroupEncoding;

#[derive(Clone)]
pub struct KeccakTranscript(Keccak256);

impl TranscriptHash for KeccakTranscript {
    type Input = Vec<u8>;
    type Output = Vec<u8>; // we return 64 bytes for your existing sampling code

    fn init() -> Self {
        // Domain separation (on-chain: start transcript bytes with this literal)
        let mut h = Keccak256::new();
        h.update(b"Domain separator for transcript");
        Self(h)
    }

    fn absorb(&mut self, input: &Self::Input) {
        self.0.update([0]);
        self.0.update(input);
    }

    fn squeeze(&mut self) -> Self::Output {
        // Mutate transcript state (so multiple squeezes differ)
        self.0.update([1]);

        // EVM-compatible 64 bytes:
        // out = keccak256(preimage || 0x00) || keccak256(preimage || 0x01)
        let mut out = Vec::with_capacity(64);

        let r0 = {
            let mut t = self.0.clone();
            t.update([0u8]);
            t.finalize()
        };
        out.extend_from_slice(r0.as_slice());

        let r1 = {
            let mut t = self.0.clone();
            t.update([1u8]);
            t.finalize()
        };
        out.extend_from_slice(r1.as_slice());

        debug_assert_eq!(out.len(), 64);
        out
    }
}

impl Hashable<KeccakTranscript> for midnight_curves::G1Projective {
    fn to_input(&self) -> Vec<u8> {
        Hashable::<KeccakTranscript>::to_bytes(self)
    }

    fn to_bytes(&self) -> Vec<u8> {
        <Self as GroupEncoding>::to_bytes(self).as_ref().to_vec()
    }

    fn read(buffer: &mut impl Read) -> io::Result<Self> {
        let mut bytes = <Self as GroupEncoding>::Repr::default();
        buffer.read_exact(bytes.as_mut())?;

        Option::from(Self::from_bytes(&bytes))
            .ok_or_else(|| io::Error::other("Invalid BLS12-381 point encoding in proof"))
    }
}

impl Hashable<KeccakTranscript> for midnight_curves::Fq {
    fn to_input(&self) -> Vec<u8> {
        self.to_repr().to_vec()
    }

    fn to_bytes(&self) -> Vec<u8> {
        self.to_repr().to_vec()
    }

    fn read(buffer: &mut impl Read) -> io::Result<Self> {
        let mut bytes = <Self as PrimeField>::Repr::default();
        buffer.read_exact(bytes.as_mut())?;

        Option::from(Self::from_repr(bytes))
            .ok_or_else(|| io::Error::other("Invalid BLS12-381 scalar encoding in proof"))
    }
}

impl Sampleable<KeccakTranscript> for midnight_curves::Fq {
    fn sample(hash_output: Vec<u8>) -> Self {
        assert!(hash_output.len() <= 64);
        assert!(hash_output.len() >= (midnight_curves::Fq::NUM_BITS as usize / 8) + 12);

        let mut bytes = [0u8; 64];
        bytes[..hash_output.len()].copy_from_slice(&hash_output);

        midnight_curves::Fq::from_uniform_bytes(&bytes)
    }
}
