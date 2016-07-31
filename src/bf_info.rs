extern crate rand;

use std;
use std::collections::HashMap;

use na::Vector3;

use self::rand::{SeedableRng,Rng};

use prim;
use terrain;
use navmap;

pub const TIME_MULTIPLIER: i32 = 60;
pub const EAT_TIME: f64        = TIME_MULTIPLIER as f64 * 479.0;

pub const SOLDIER_SPEED: f64 = 1.3; // m/s

pub const TRUCK_NUM_PASSENGERS: i32 = 8;

const SUPPLY_MAX_FOOD: i32 = 800;
const SUPPLY_MAX_AMMO: i32 = 4000;

#[derive(PartialEq, Eq, Copy, Clone, Hash)]
pub struct SoldierID {
    pub id: usize,
}

#[derive(PartialEq, Eq, Copy, Clone, Hash)]
pub struct TruckID {
    pub id: usize,
}

pub struct Soldier {
    pub position: Vector3<f64>,
    pub direction: f64,
    pub alive: bool,
    pub side: prim::Side,
    pub id: SoldierID,
    pub shot_timer: f64,
    pub reap_timer: f64,
    pub ammo: i32,
    pub food: i32,
    pub eat_timer: f64,
}

pub struct Truck {
    pub position: Vector3<f64>,
    pub speed: f64,
    pub direction: Vector3<f64>,
    pub alive: bool,
    pub side: prim::Side,
    pub id: TruckID,
    pub reap_timer: f64,
    pub fuel: f64,
}

pub struct Camera {
    pub position:  Vector3<f32>,
    pub direction: Vector3<f32>,
    pub upvec:     Vector3<f32>,
    pub speed:     Vector3<f32>,
}

#[derive(PartialEq, Eq, Copy, Clone)]
pub enum BoardRole {
    Driver,
    Passenger,
}

pub struct Boarded {
    pub sid: SoldierID,
    pub role: BoardRole,
}

pub struct BoardedMap {
    pub map: HashMap<TruckID, Vec<Boarded>>,
}

pub struct SupplyPoint {
    pub position: Vector3<f64>,
    pub amount_food: i32,
    pub amount_ammo: i32,
}

pub struct Movers {
    pub soldiers: Vec<Soldier>,
    pub trucks: Vec<Truck>,
    pub boarded_map: BoardedMap,
}

pub struct Battlefield {
    pub camera: Camera,
    pub view_mode: prim::ViewMode,
    pub mouse_look: bool,
    pub prev_mouse_position: Option<(i32, i32)>,
    pub ground: terrain::Ground,
    pub curr_time: f64,
    pub frame_time: f64,
    pub winner: Option<prim::Side>,
    pub flags: Vec<prim::Flag>,
    pub time_accel: i32,
    pub rng: rand::StdRng,
    pub base_position: [Vector3<f64>; 2],
    pub supply_points: Vec<SupplyPoint>,
    pub movers: Movers,
    pub pause: bool,
}

pub trait Locatable {
    fn pos(&self) -> Vector3<f64>;
}

impl Locatable for Soldier {
    fn pos(&self) -> Vector3<f64> {
        self.position
    }
}

impl Locatable for SupplyPoint {
    fn pos(&self) -> Vector3<f64> {
        self.position
    }
}

impl Locatable for Truck {
    fn pos(&self) -> Vector3<f64> {
        self.position
    }
}

impl Locatable for prim::Flag {
    fn pos(&self) -> Vector3<f64> {
        self.position
    }
}

impl Battlefield {
    pub fn new(seed: usize, ground_params: &terrain::GroundParams) -> Battlefield {
        let ground = terrain::init_ground(&ground_params);
        let mut rng = rand::StdRng::from_seed(&[seed]);

        let base_positions = navmap::find_base_positions(&ground, &mut rng);
        let flags = navmap::find_flag_positions(&ground, &mut rng);

        Battlefield {
            camera: Camera {
                position:  Vector3::new(-prim::HDIM as f32,
                                        (terrain::get_height_at(&ground, -prim::HDIM, 0.0) + 70.0) as f32,
                                        0.0),
                direction: Vector3::new(1.0, 0.0, 0.0),
                upvec:     Vector3::new(0.0, 1.0, 0.0),
                speed:     Vector3::new(0.0, 0.0, 0.0),
            },
            view_mode: prim::ViewMode::Normal,
            mouse_look: false,
            prev_mouse_position: None,
            movers: Movers {
                soldiers: vec![],
                trucks: vec![],
                boarded_map: BoardedMap{map: HashMap::new()},
            },
            ground: ground,
            curr_time: TIME_MULTIPLIER as f64 * 360.0,
            frame_time: 0.0,
            winner: None,
            flags: flags,
            time_accel: 1,
            rng: rng,
            base_position: base_positions,
            supply_points: base_positions.iter().map(|p| SupplyPoint {
                position: *p,
                amount_food: 0,
                amount_ammo: 0,
            }).collect(),
            pause: false,
        }
    }

    pub fn rand(&mut self) -> f64 {
        self.rng.gen::<f64>()
    }
}

impl SupplyPoint {
    pub fn add_food(&mut self, i: i32) {
        self.amount_food += i;
        self.amount_food = std::cmp::min(self.amount_food, SUPPLY_MAX_FOOD);
    }

    pub fn add_ammo(&mut self, i: i32) {
        self.amount_ammo += i;
        self.amount_ammo = std::cmp::min(self.amount_ammo, SUPPLY_MAX_AMMO);
    }
}

pub fn get_base_position(bf: &Battlefield, side: prim::Side) -> Vector3<f64> {
    bf.base_position[if side == prim::Side::Red { 1 } else { 0 }]
}

pub fn soldier_boarded(boarded_map: &BoardedMap, s: SoldierID) -> Option<(TruckID, BoardRole)> {
    for (tid, bds) in &boarded_map.map {
        for bd in bds {
            if bd.sid == s {
                return Some((*tid, bd.role));
            }
        }
    }
    return None;
}

pub fn truck_free_by_id(bf: &Battlefield, truck: TruckID) -> bool {
    if !bf.movers.trucks[truck.id].alive {
        return false;
    }

    if let Some(ref bds) = bf.movers.boarded_map.map.get(&truck) {
        if bds.len() < TRUCK_NUM_PASSENGERS as usize + 1 {
            return true;
        } else {
            return false;
        }
    }

    return true;
}

pub fn truck_free(bf: &Battlefield, truck: &Truck) -> bool {
    truck_free_by_id(bf, truck.id)
}

pub fn num_passengers(bf: &Battlefield, truck: &Truck) -> i32 {
    match bf.movers.boarded_map.map.get(&truck.id) {
        None => 0,
        Some(b) => std::cmp::max(0, b.len() as i32 - 1),
    }
}

pub fn add_supplies(bf: &mut Battlefield) -> () {
    for i in 0..2 {
        bf.supply_points[i].add_food(10);
        bf.supply_points[i].add_ammo(100);
    }
}


