use na::Vector3;

pub const GROUND_NUM_TILES: i32 = 128;
pub const TILE_SIZE:   f64 = 1024.0;

pub const DIM:  f64 = GROUND_NUM_TILES as f64 * TILE_SIZE;
pub const HDIM: f64 = DIM * 0.5;

pub const FLAG_TIMER: f64 = 10.0;

#[derive(PartialEq, Eq, Copy, Clone)]
#[derive(RustcDecodable, RustcEncodable)]
pub enum ViewMode {
    Normal,
    Tactical,
    Strategic,
}

#[derive(RustcDecodable, RustcEncodable)]
pub enum FlagState {
    Free,
    Transition(Side),
    Owned(Side),
}

#[derive(RustcDecodable, RustcEncodable)]
pub struct Flag {
    pub position: Vector3<f64>,
    pub flag_state: FlagState,
    pub flag_timer: f64,
}

#[derive(PartialEq, Eq, Copy, Clone)]
#[derive(RustcDecodable, RustcEncodable)]
pub enum Side {
    Blue,
    Red,
}

pub fn side_to_index(side: Side) -> usize {
    if side == Side::Blue { 0 } else { 1 }
}
