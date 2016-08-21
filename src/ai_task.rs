extern crate nalgebra as na;
extern crate core;
extern crate rand;

use std::collections::VecDeque;

use na::{Vector3, Norm, Dot, Cross};

use bf_info::*;

use gameutil;
use prim;
use actions;
use ai_prim;
use ai_prim::*;

const SHOOT_DISTANCE: f64 = 100.0;

pub trait Task {
    fn update(&mut self, soldier: &Soldier, bf: &Battlefield) -> Option<Action>;
}

impl Task for AiTask {
    fn update(&mut self, soldier: &Soldier, bf: &Battlefield) -> Option<Action> {
        match self {
            &mut AiTask::Goto(ref mut g)  => g.update(soldier, bf),
            &mut AiTask::Board(ref mut g) => g.update(soldier, bf),
            &mut AiTask::Drive(ref mut g) => g.update(soldier, bf),
            &mut AiTask::Sleep(ref mut g) => g.update(soldier, bf),
            &mut AiTask::Taxi(ref mut g)  => g.update(soldier, bf),
        }
    }
}

#[derive(RustcDecodable, RustcEncodable)]
pub struct SideAI {
    side: prim::Side,
}

impl SideAI {
    pub fn new(side: prim::Side) -> SideAI {
        SideAI {
            side: side,
        }
    }
}

#[derive(RustcDecodable, RustcEncodable)]
pub struct SoldierAI {
    pub tasks: VecDeque<AiTask>,
    pub replan_timer: f64,
    pub status: Status,
    pub my_vehicle: Option<VehicleID>,
    pub nearby_flag: Option<(Vector3<f64>, f64)>,
    pub nearby_vehicle: Option<(Vector3<f64>, VehicleID, f64)>,
}

impl SoldierAI {
    pub fn new() -> SoldierAI {
        SoldierAI {
            tasks: VecDeque::new(),
            replan_timer: rand::random::<f64>() * REPLAN_TIME,
            status: Status::OnFoot,
            my_vehicle: None,
            nearby_flag: None,
            nearby_vehicle: None,
        }
    }
}

pub fn game_minutes(m: f64) -> f64 {
    TIME_MULTIPLIER as f64 * m
}

#[derive(RustcDecodable, RustcEncodable)]
pub enum AiTask {
    Goto(AiGoto),
    Board(AiBoard),
    Drive(AiDrive),
    Sleep(AiSleep),
    Taxi(AiTaxi),
}

// goto: if enemy is within shooting range, stop to shoot.
// else go to given point and stay there.
#[derive(RustcDecodable, RustcEncodable)]
pub struct AiGoto {
    targetpos: Vector3<f64>,
}

// goto task - constructor
impl AiGoto {
    pub fn new(tgt: Vector3<f64>) -> AiGoto {
        AiGoto {
            targetpos: tgt
        }
    }
}

// goto task - update function
impl Task for AiGoto {
    fn update(&mut self, soldier: &Soldier, bf: &Battlefield) -> Option<Action> {
        let st = get_status(bf, soldier);
        match st {
            Status::OnFoot => {
                let enemy_tgt = find_enemy(soldier, bf);
                match enemy_tgt {
                    None    => ai_goto(&self.targetpos, soldier, st),
                    Some(e) => attack(e, soldier, &bf),
                }
            },
            Status::Boarded(_) => Some(Action::DisembarkAction(soldier.id)),
            Status::Driving(_) => None,
        }
    }
}

fn find_enemy(soldier: &Soldier, bf: &Battlefield) -> Option<SoldierID> {
    if soldier.ammo <= 0 {
        return None;
    }

    for i in 0..bf.movers.soldiers.len() {
        if bf.movers.soldiers[i].alive && bf.movers.soldiers[i].side != soldier.side && gameutil::dist(soldier, &bf.movers.soldiers[i].position) < SHOOT_DISTANCE {
            return Some(SoldierID{id: i})
        }
    }
    None
}

fn ai_goto(targetpos: &Vector3<f64>, soldier: &Soldier, st: Status) -> Option<Action> {
    let tgt_vec = gameutil::to_vec_on_map(soldier, targetpos);
    let dist_to_tgt = tgt_vec.norm();
    match st {
        Status::OnFoot => {
            if dist_to_tgt >= 1.0 {
                Some(Action::MoveAction(soldier.id, tgt_vec.normalize() * SOLDIER_SPEED))
            } else {
                None
            }
        },
        Status::Boarded(_) => {
            Some(Action::DisembarkAction(soldier.id))
        },
        Status::Driving(_) => {
            assert!(false);
            None
        }
    }
}

fn attack(e: SoldierID, soldier: &Soldier, bf: &Battlefield) -> Option<Action> {
    assert!(soldier.ammo > 0);
    let dist = (bf.movers.soldiers[e.id].position - soldier.position).norm();
    if dist > SHOOT_DISTANCE {
        let st = get_status(bf, soldier);
        ai_goto(&bf.movers.soldiers[e.id].position, soldier, st)
    } else {
        if soldier.shot_timer <= 0.0 {
            Some(Action::ShootAction(soldier.id, e))
        } else {
            Some(Action::NoAction(soldier.id))
        }
    }
}

// board: goto vehicle, then board it.
#[derive(RustcDecodable, RustcEncodable)]
pub struct AiBoard {
    targetpos: Vector3<f64>,
}

// board task - constructor
impl AiBoard {
    pub fn new(tgt: Vector3<f64>) -> AiBoard {
        AiBoard {
            targetpos: tgt
        }
    }
}

// board task - update function
impl Task for AiBoard {
    fn update(&mut self, soldier: &Soldier, bf: &Battlefield) -> Option<Action> {
        let st = get_status(bf, soldier);
        if st != Status::OnFoot {
            None
        } else {
            if gameutil::dist(soldier, &self.targetpos) < actions::MAX_BOARD_DISTANCE {
                if let Some(vehicle) = ai_prim::free_vehicle_nearby(soldier, bf, actions::MAX_BOARD_DISTANCE) {
                    Some(Action::BoardAction(soldier.id, vehicle.1))
                } else {
                    None
                }
            } else {
                ai_goto(&self.targetpos, soldier, st)
            }
        }
    }
}

// drive: goto given point and stay.
#[derive(RustcDecodable, RustcEncodable)]
pub struct AiDrive {
    targetpos: Vector3<f64>,
}

// drive task - constructor
impl AiDrive {
    pub fn new(tgt: Vector3<f64>) -> AiDrive {
        AiDrive {
            targetpos: tgt
        }
    }
}

// drive task - update function
impl Task for AiDrive {
    fn update(&mut self, soldier: &Soldier, bf: &Battlefield) -> Option<Action> {
        let st = get_status(bf, soldier);
        if let Status::Driving(tid) = st {
            let ref vehicle = bf.movers.vehicles[tid.id];
            let tgt_vec = gameutil::to_vec_on_map(soldier, &self.targetpos);
            let dist_to_tgt = tgt_vec.norm();
            let norm_tgt_vec = tgt_vec.normalize();
            if dist_to_tgt >= 3.0 {
                let curr_dir = vehicle.direction.normalize();
                let mut ang = f64::acos(norm_tgt_vec.dot(&curr_dir));
                let cross = norm_tgt_vec.cross(&curr_dir);
                if Vector3::new(0.0, 1.0, 0.0).dot(&cross) > 0.0 {
                    ang = -ang;
                }
                let gas = gameutil::clamp(0.0, 1.0, dist_to_tgt * 0.002) -
                    gameutil::clamp(0.0, 0.8, f64::abs(ang));
                Some(Action::DriveAction(soldier.id, ang, gas))
            } else {
                if vehicle.speed < 0.1 {
                    None
                } else {
                    Some(Action::DriveAction(soldier.id, 0.0, -1.0))
                }
            }
        } else {
            None
        }
    }
}

// sleep: do nothing for a while.
#[derive(RustcDecodable, RustcEncodable)]
pub struct AiSleep {
    time: f64, // seconds
}

// sleep task - constructor
impl AiSleep {
    pub fn new(time: f64) -> AiSleep {
        AiSleep {
            time: time
        }
    }
}

// sleep task - update function
impl Task for AiSleep {
    fn update(&mut self, soldier: &Soldier, bf: &Battlefield) -> Option<Action> {
        self.time -= bf.frame_time;
        if self.time < 0.0 {
            None
        } else {
            Some(Action::NoAction(soldier.id))
        }
    }
}

// taxi: sleep until the vehicle has stopped.
#[derive(RustcDecodable, RustcEncodable)]
pub struct AiTaxi {
    wait_time: f64,
}

impl AiTaxi {
    pub fn new() -> AiTaxi {
        AiTaxi {
            wait_time: game_minutes(2.0)
        }
    }
}

impl Task for AiTaxi {
    fn update(&mut self, soldier: &Soldier, bf: &Battlefield) -> Option<Action> {
        self.wait_time -= bf.frame_time;
        if self.wait_time <= 0.0 {
            self.wait_time += game_minutes(2.0);
            let st = get_status(bf, soldier);
            match st {
                Status::Boarded(_) => {
                    if ai_prim::flag_nearby(soldier, bf, 1000.0).is_some() {
                        Some(Action::DisembarkAction(soldier.id))
                    } else {
                        Some(Action::NoAction(soldier.id))
                    }
                },
                _ => None,
            }
        } else {
            Some(Action::NoAction(soldier.id))
        }
    }
}


