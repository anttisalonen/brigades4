extern crate rand;

use std;
use std::collections::HashMap;

use na::{Vector3, Norm};

use self::rand::{SeedableRng,Rng};

use prim;
use terrain;
use navmap;

pub const TIME_MULTIPLIER: i32 = 60;
pub const EAT_TIME: f64        = TIME_MULTIPLIER as f64 * 479.0;

pub const SOLDIER_SPEED: f64 = 1.3; // m/s

const SUPPLY_MAX_FOOD: i32 = 800;
const SUPPLY_MAX_AMMO: i32 = 4000;

#[derive(PartialEq, Eq, Copy, Clone, Hash)]
#[derive(RustcDecodable, RustcEncodable)]
pub struct SoldierID {
    pub id: usize,
}

#[derive(PartialEq, Eq, Copy, Clone, Hash)]
#[derive(RustcDecodable, RustcEncodable)]
pub struct VehicleID {
    pub id: usize,
}

#[derive(RustcDecodable, RustcEncodable)]
pub struct VehicleInfoID {
    pub side: prim::Side,
    pub id: usize,
}

#[derive(RustcDecodable, RustcEncodable)]
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

#[derive(RustcDecodable, RustcEncodable, Debug)]
#[derive(PartialEq, Eq, Copy, Clone)]
pub enum VehicleType {
    Land,
    Sea,
}

#[derive(RustcDecodable, RustcEncodable, Debug)]
pub struct VehicleInfo {
    pub name: String,
    pub vehicle_type: VehicleType,
    pub max_speed_grass: f64,
    pub max_speed_forest: f64,
    pub max_speed_sea: f64,
    pub num_passengers: i32,
    pub spawn_rate: i32,
    pub max_num_per_side: i32,
}

#[derive(RustcDecodable, RustcEncodable)]
pub struct Vehicle {
    pub info: VehicleInfoID,
    pub position: Vector3<f64>,
    pub speed: f64,
    pub direction: Vector3<f64>,
    pub alive: bool,
    pub side: prim::Side,
    pub id: VehicleID,
    pub reap_timer: f64,
    pub fuel: f64,
}

#[derive(RustcDecodable, RustcEncodable)]
pub struct Camera {
    pub position:  Vector3<f32>,
    pub direction: Vector3<f32>,
    pub upvec:     Vector3<f32>,
    pub speed:     Vector3<f32>,
}

#[derive(PartialEq, Eq, Copy, Clone)]
#[derive(RustcDecodable, RustcEncodable)]
pub enum BoardRole {
    Driver,
    Passenger,
}

#[derive(RustcDecodable, RustcEncodable)]
pub struct Boarded {
    pub sid: SoldierID,
    pub role: BoardRole,
}

#[derive(RustcDecodable, RustcEncodable)]
pub struct BoardedMap {
    pub map: HashMap<usize, Vec<Boarded>>,
}

#[derive(RustcDecodable, RustcEncodable)]
pub struct NavalBase {
    pub position: Vector3<f64>,
}

#[derive(RustcDecodable, RustcEncodable)]
pub struct SupplyPoint {
    pub position: Vector3<f64>,
    pub amount_food: i32,
    pub amount_ammo: i32,
}

#[derive(RustcDecodable, RustcEncodable)]
pub struct Movers {
    pub soldiers: Vec<Soldier>,
    pub vehicles: Vec<Vehicle>,
    pub boarded_map: BoardedMap,
    pub vehicle_info: VehicleInfoMap,
}

#[derive(RustcDecodable, RustcEncodable)]
pub struct VehicleInfoMap {
    vmap: [Vec<VehicleInfo>; 2],
}

#[derive(RustcDecodable, RustcEncodable)]
pub struct Battlefield {
    pub camera: Camera,
    pub view_mode: prim::ViewMode,
    pub mouse_look: bool,
    pub prev_mouse_position: Option<(i32, i32)>,
    pub ground: terrain::Ground,
    pub navmap: navmap::Navmap,
    pub curr_time: f64,
    pub frame_time: f64,
    pub winner: Option<prim::Side>,
    pub flags: Vec<prim::Flag>,
    pub time_accel: i32,
    pub base_position: [Vector3<f64>; 2],
    pub supply_points: Vec<SupplyPoint>,
    pub naval_bases: Vec<NavalBase>,
    pub movers: Movers,
    pub pause: bool,
    naval_spawn_positions: [Vector3<f64>; 2],
}

pub trait Locatable {
    fn pos(&self) -> Vector3<f64>;
}

impl Locatable for Soldier {
    fn pos(&self) -> Vector3<f64> {
        self.position
    }
}

impl Locatable for NavalBase {
    fn pos(&self) -> Vector3<f64> {
        self.position
    }
}

impl Locatable for SupplyPoint {
    fn pos(&self) -> Vector3<f64> {
        self.position
    }
}

impl Locatable for Vehicle {
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
    pub fn new(seed: usize, ground_params: &terrain::GroundParams, vehicle_params: [Vec<VehicleInfo>; 2]) -> Option<Battlefield> {
        let ground = terrain::init_ground(&ground_params);
        let mut rng = rand::StdRng::from_seed(&[seed]);

        let nv = navmap::Navmap::new(&ground);
        let config = nv.find_config(&ground, &mut rng);
        if config.base_positions.is_none() {
            println!("Unable to find base positions");
            return None;
        }
        let base_positions = config.base_positions.unwrap();
        let naval_bases = match config.naval_positions {
            None    => vec![],
            Some(n) => n.iter().map(|p| NavalBase {
                position: *p,
            }).collect(),
        };

        let naval_spawn_positions = [find_naval_spawn_position(&naval_bases[0].position,
                                                               &base_positions[0],
                                                               &ground),
                                     find_naval_spawn_position(&naval_bases[1].position,
                                                               &base_positions[1],
                                                               &ground)];

        Some(Battlefield {
            camera: Camera {
                position:  Vector3::new(0.0,
                                        91000.0,
                                        -56200.0),
                direction: Vector3::new(0.0, -0.93, 0.3667),
                upvec:     Vector3::new(0.0, 1.0, 0.0),
                speed:     Vector3::new(0.0, 0.0, 0.0),
            },
            view_mode: prim::ViewMode::Strategic,
            mouse_look: false,
            prev_mouse_position: None,
            movers: Movers {
                soldiers: vec![],
                vehicles: vec![],
                boarded_map: BoardedMap{map: HashMap::new()},
                vehicle_info: VehicleInfoMap{vmap: vehicle_params},
            },
            ground: ground,
            navmap: nv,
            curr_time: TIME_MULTIPLIER as f64 * 360.0,
            frame_time: 0.0,
            winner: None,
            flags: config.flags,
            time_accel: 1,
            base_position: base_positions,
            supply_points: base_positions.iter().map(|p| SupplyPoint {
                position: *p,
                amount_food: 0,
                amount_ammo: 0,
            }).collect(),
            naval_bases: naval_bases,
            pause: false,
            naval_spawn_positions: naval_spawn_positions,
        })
    }

    pub fn rand(&mut self) -> f64 {
        rand::thread_rng().gen::<f64>()
    }

    pub fn count_vehicles(&self, side: prim::Side, name: &str) -> i32 {
        self.movers.vehicles.iter().filter(|v| is_vehicle(v, &self.movers.vehicle_info, side, name)).count() as i32
    }

    pub fn get_naval_spawn_position(&self, side: prim::Side) -> Vector3<f64> {
        self.naval_spawn_positions[prim::side_to_index(side)]
    }

}

impl Movers {
    pub fn get_vehicle_info_from_vehicle_id(&self, id: VehicleID) -> &VehicleInfo {
        let ref veh = self.vehicles[id.id];
        &self.vehicle_info.vmap[prim::side_to_index(veh.info.side)][veh.info.id]
    }

    pub fn vehicle_type(&self, id: VehicleID) -> VehicleType {
        let ref veh = self.vehicles[id.id];
        self.vehicle_info.vmap[prim::side_to_index(veh.info.side)][veh.info.id].vehicle_type
    }

    pub fn search_profile(&self, id: VehicleID) -> navmap::SearchProfile {
        match self.vehicle_type(id) {
            VehicleType::Land => navmap::SearchProfile::Land,
            VehicleType::Sea  => navmap::SearchProfile::Sea,
        }
    }
}

impl VehicleInfoMap {
    pub fn get_vehicle_info_id(&self, side: prim::Side, name: &str) -> Option<VehicleInfoID> {
        for (i, v) in self.vmap[prim::side_to_index(side)].iter().enumerate() {
            if v.name == name {
                return Some(VehicleInfoID{side: side, id: i});
            }
        }
        None
    }

    pub fn get_vehicle_info(&self, side: prim::Side, name: &str) -> Option<&VehicleInfo> {
        for v in self.vmap[prim::side_to_index(side)].iter() {
            if v.name == name {
                return Some(v);
            }
        }
        None
    }

}

fn find_naval_spawn_position(naval_base: &Vector3<f64>, base_position: &Vector3<f64>, ground: &terrain::Ground) -> Vector3<f64> {
    // walk from base to naval position until coastline found
    let mut vec = *naval_base - *base_position;
    vec.y = 0.0;
    let len = vec.norm() as i64;
    let vec = vec.normalize();
    for i in 0..len {
        let pos = *base_position + vec * i as f64;
        let hgt = terrain::get_height_at(ground, pos.x, pos.z);
        if hgt < 0.0 {
            return Vector3::new(pos.x, 0.0, pos.z);
        }
    }
    assert!(false);
    *naval_base
}

pub fn is_vehicle(veh: &Vehicle, vmap: &VehicleInfoMap, side: prim::Side, name: &str) -> bool {
    let ref info = vmap.vmap[prim::side_to_index(side)][veh.info.id];
    info.name == name
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

pub fn get_base_or_naval_position(bf: &Battlefield, side: prim::Side, id: VehicleID) -> Vector3<f64> {
    match bf.movers.vehicle_type(id) {
        VehicleType::Land => get_base_position(bf, side),
        VehicleType::Sea  => get_naval_position(bf, side),
    }
}

pub fn get_naval_position(bf: &Battlefield, side: prim::Side) -> Vector3<f64> {
    bf.naval_bases[prim::side_to_index(side)].position
}

pub fn get_base_position(bf: &Battlefield, side: prim::Side) -> Vector3<f64> {
    bf.base_position[prim::side_to_index(side)]
}

pub fn soldier_boarded(boarded_map: &BoardedMap, s: SoldierID) -> Option<(VehicleID, BoardRole)> {
    for (tid, bds) in &boarded_map.map {
        for bd in bds {
            if bd.sid == s {
                return Some((VehicleID{id:*tid}, bd.role));
            }
        }
    }
    return None;
}

pub fn vehicle_free_by_id(bf: &Battlefield, vehicle: VehicleID) -> bool {
    if !bf.movers.vehicles[vehicle.id].alive {
        return false;
    }

    if let Some(ref bds) = bf.movers.boarded_map.map.get(&vehicle.id) {
        if bds.len() < bf.movers.get_vehicle_info_from_vehicle_id(vehicle).num_passengers as usize + 1 {
            return true;
        } else {
            return false;
        }
    }

    return true;
}

pub fn vehicle_free(bf: &Battlefield, vehicle: &Vehicle) -> bool {
    vehicle_free_by_id(bf, vehicle.id)
}

pub fn num_passengers(bf: &Battlefield, vehicle: &Vehicle) -> i32 {
    match bf.movers.boarded_map.map.get(&vehicle.id.id) {
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

pub enum Action {
    NoAction(SoldierID),
    MoveAction(SoldierID, Vector3<f64>),
    ShootAction(SoldierID, SoldierID),
    BoardAction(SoldierID, VehicleID),
    DriveAction(SoldierID, f64, f64),
    DisembarkAction(SoldierID),
}


