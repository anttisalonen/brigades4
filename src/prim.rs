use na::Vector3;

pub const GROUND_NUM_TILES: i32 = 128;
pub const TILE_SIZE:   f64 = 1024.0;

pub const DIM:  f64 = GROUND_NUM_TILES as f64 * TILE_SIZE;
pub const HDIM: f64 = DIM * 0.5;

pub const FLAG_TIMER: f64 = 10.0;

#[derive(PartialEq, Eq, Copy, Clone)]
pub enum ViewMode {
    Normal,
    Tactical,
    Strategic,
}

pub enum FlagState {
    Free,
    Transition(Side),
    Owned(Side),
}

pub struct Flag {
    pub position: Vector3<f64>,
    pub flag_state: FlagState,
    pub flag_timer: f64,
}

#[derive(PartialEq, Eq, Copy, Clone)]
pub enum Side {
    Blue,
    Red,
}


