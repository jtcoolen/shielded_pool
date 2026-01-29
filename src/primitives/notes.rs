// SPDX-License-Identifier: CC0-1.0

use crate::Utxo;
use midnight_curves::Fq as F;

#[derive(Clone, Debug)]
pub struct Note {
    pub utxo: Utxo,
    pub commit: F,
    pub spent: bool,
    pub confirmed_at_root_idx: usize,
}
