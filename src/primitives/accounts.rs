// SPDX-License-Identifier: CC0-1.0

use crate::primitives::notes::Note;
use midnight_curves::{Fq as F, Fr as JubjubScalar, JubjubSubgroup};

#[derive(Clone)]
pub struct Account {
    pub id: usize,
    pub sk: JubjubScalar,
    pub pk_point: JubjubSubgroup,
    pub pk_x: F,
    pub pk_y: F,
    pub wallet: Vec<Note>,
}
