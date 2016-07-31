extern crate nalgebra as na;
extern crate core;
extern crate rand;

use self::core::iter::FromIterator;

use na::{Vector3, Norm, Dot, Cross};

use bf_info::*;

use gameutil;
use prim;
use navmap;
use terrain;

const SHOOT_DISTANCE: f64 = 100.0;
const REPLAN_TIME: f64       = TIME_MULTIPLIER as f64 * 120.0; // seconds
const FOOD_FETCH_BUFFER: f64 = TIME_MULTIPLIER as f64 * 480.0; // seconds

// data structures
pub struct SoldierAI {
    tasks: Vec<AiTask>,
    replan_timer: f64,
}

pub enum Action {
    NoAction(SoldierID),
    MoveAction(SoldierID, Vector3<f64>),
    ShootAction(SoldierID, SoldierID),
    BoardAction(SoldierID, TruckID),
    DriveAction(SoldierID, f64, f64),
    DisembarkAction(SoldierID),
}

macro_rules! sort_by_distance {
    ( $soldier:expr, $items:expr ) => {
        {
            let distances = $items.iter().map(|item| gameutil::dist($soldier, &item.pos()));
            let zm = distances.zip($items.iter());
            let mut zm = Vec::from_iter(zm);
            zm.sort_by(|&(d1, _), &(d2, _)| d1.partial_cmp(&d2).unwrap());
            zm
        }
    };
}

// core code
trait Task {
    fn update(&mut self, soldier: &Soldier, bf: &Battlefield) -> Option<Action>;
}

impl Task for AiTask {
    fn update(&mut self, soldier: &Soldier, bf: &Battlefield) -> Option<Action> {
        match self {
            &mut AiTask::Goto(ref mut g)  => g.update(soldier, bf),
            &mut AiTask::Board(ref mut g) => g.update(soldier, bf),
            &mut AiTask::Drive(ref mut g) => g.update(soldier, bf),
            &mut AiTask::Sleep(ref mut g) => g.update(soldier, bf),
        }
    }
}

impl SoldierAI {
    pub fn new() -> SoldierAI {
        SoldierAI {
            tasks: vec![],
            replan_timer: rand::random::<f64>() * REPLAN_TIME,
        }
    }
}

pub fn soldier_ai_update(sai: &mut SoldierAI, soldier: &Soldier, bf: &Battlefield) -> Action {
    sai.replan_timer -= bf.frame_time;
    if sai.replan_timer <= 0.0 {
        sai.replan_timer += REPLAN_TIME * (rand::random::<f64>() * 2.0);
        sai.tasks = vec![];
    }
    if sai.tasks.len() == 0 {
        sai.tasks.append(&mut ai_arbitrate_task(soldier, bf));
    }
    let num_tasks = sai.tasks.len();
    if num_tasks > 0 {
        let act = sai.tasks[num_tasks - 1].update(soldier, bf);
        match act {
            None    => sai.tasks.truncate(num_tasks - 1),
            Some(a) => return a,
        }
    }
    return Action::NoAction(soldier.id);
}

fn find_nearest_supply(bf: &Battlefield, s: &Soldier, st: Status) -> Option<Vector3<f64>> {
    match st {
        Status::Driving(_) => return Some(get_base_position(bf, s.side)),
        _                  => (),
    };
    let zm = sort_by_distance!(s, &bf.supply_points);
    for (dist, supply) in zm {
        if dist > 1000.0 {
            return None;
        }
        if supply.amount_food < 3 && supply.amount_ammo < 3 {
            continue;
        }
        return Some(supply.position);
    }
    return None;
}

#[derive(PartialEq, Eq, Copy, Clone)]
enum Status {
    OnFoot,
    Driving(TruckID),
    Boarded(TruckID),
}

fn get_status(bf: &Battlefield, s: &Soldier) -> Status {
    match soldier_boarded(&bf.movers.boarded_map, s.id) {
        Some((tid, BoardRole::Driver))    => Status::Driving(tid),
        Some((tid, BoardRole::Passenger)) => Status::Boarded(tid),
        _                                 => Status::OnFoot,
    }
}

fn ai_arbitrate_task(s: &Soldier, bf: &Battlefield) -> Vec<AiTask> {
    let st = get_status(bf, s);
    let supply = find_nearest_supply(bf, s, st);

    let time_with_food = EAT_TIME * (s.food - 1) as f64;

    if supply != None && (time_with_food < FOOD_FETCH_BUFFER || s.ammo < 5) {
        match st {
            Status::Driving(_) => find_drive_path(&bf.ground, s.position, supply.unwrap()),
            _                  => vec![AiTask::Goto(AiGoto::new(supply.unwrap()))],
        }
    } else {
        match st {
            Status::Driving(tid) => {
                let ref truck = bf.movers.trucks[tid.id];
                if have_enough_passengers(bf, truck) {
                    // taxi
                    find_drive_path(&bf.ground, s.position, flag_target_position(s, bf))
                } else {
                    // back home
                    let bp = get_base_position(bf, s.side);
                    let dist = gameutil::dist(s, &bp);
                    if dist < 50.0 {
                        vec![AiTask::Sleep(AiSleep::new(15.0 * 60.0))]
                    } else {
                        find_drive_path(&bf.ground, s.position, bp)
                    }
                }
            },
            Status::Boarded(_) => vec![AiTask::Goto(AiGoto::new(flag_target_position(s, bf)))],
            Status::OnFoot     => {
                if let Some(truck) = free_truck_nearby(s, bf) {
                    vec![AiTask::Board(AiBoard::new(truck.0))]
                } else {
                    vec![AiTask::Goto(AiGoto::new(flag_target_position(s, bf)))]
                }
            }
        }
    }
}

fn find_drive_path(ground: &terrain::Ground, mypos: Vector3<f64>, targetpos: Vector3<f64>) -> Vec<AiTask> {
    let mpath = navmap::find_path(ground, mypos, targetpos);
    match mpath {
        None       => vec![AiTask::Sleep(AiSleep::new(15.0 * 60.0))],
        Some(path) => {
            path.iter().rev().map(|node| {
                AiTask::Drive(AiDrive::new(Vector3::new(node.x as f64, 0.0, node.y as f64)))
            }).collect()
        }
    }
}

// generic helpers
fn free_truck_nearby(s: &Soldier, bf: &Battlefield) -> Option<(Vector3<f64>, TruckID)> {
    let zm = sort_by_distance!(s, &bf.movers.trucks);
    for (dist, truck) in zm {
        if dist > 1000.0 {
            return None;
        }
        if !truck_free(bf, &truck) {
            continue;
        }
        return Some((truck.position, truck.id));
    }
    return None;
}

fn flag_target_position(sold: &Soldier, bf: &Battlefield) -> Vector3<f64> {
    let side = sold.side;
    let zm = sort_by_distance!(sold, &bf.flags);
    for (_, flag) in zm {
        match flag.flag_state {
            prim::FlagState::Free                       => return flag.position,
            prim::FlagState::Transition(s) if s == side => (),
            prim::FlagState::Transition(_)              => return flag.position,
            prim::FlagState::Owned(s) if s == side      => if flag_lone_holder(sold, bf, &flag.position) { return flag.position; } else { () },
            prim::FlagState::Owned(_)                   => return flag.position,
        }
    }
    bf.flags[0].position
}

fn flag_lone_holder(sold: &Soldier, bf: &Battlefield, pos: &Vector3<f64>) -> bool {
    let mut num_guards = 3i16;
    if gameutil::dist(sold, pos) > 5.0 {
        return false;
    }

    let side = sold.side;
    for sn in bf.movers.soldiers.iter() {
        if sn.side == side && gameutil::dist(sn, pos) < 5.0 {
            num_guards -= 1;
            if num_guards == 0 {
                return false;
            }
        }
    }
    return true;
}


// task listing
enum AiTask {
    Goto(AiGoto),
    Board(AiBoard),
    Drive(AiDrive),
    Sleep(AiSleep),
}

// goto: if enemy is within shooting range, stop to shoot.
// else go to given point and stay there.
struct AiGoto {
    targetpos: Vector3<f64>,
}

// goto task - constructor
impl AiGoto {
    fn new(tgt: Vector3<f64>) -> AiGoto {
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
                    None    => Some(ai_goto(&self.targetpos, soldier, st)),
                    Some(e) => Some(attack(e, soldier, &bf)),
                }
            },
            Status::Boarded(_) => Some(ai_goto(&self.targetpos, soldier, st)),
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

fn ai_goto(targetpos: &Vector3<f64>, soldier: &Soldier, st: Status) -> Action {
    let tgt_vec = gameutil::to_vec_on_map(soldier, targetpos);
    let dist_to_tgt = tgt_vec.norm();
    match st {
        Status::OnFoot => {
            if dist_to_tgt >= 1.0 {
                Action::MoveAction(soldier.id, tgt_vec.normalize() * SOLDIER_SPEED)
            } else {
                Action::NoAction(soldier.id)
            }
        },
        Status::Boarded(_) => {
            if dist_to_tgt < 1000.0 {
                Action::DisembarkAction(soldier.id)
            } else {
                Action::NoAction(soldier.id)
            }
        },
        Status::Driving(_) => {
            assert!(false);
            Action::NoAction(soldier.id)
        }
    }
}

fn attack(e: SoldierID, soldier: &Soldier, bf: &Battlefield) -> Action {
    assert!(soldier.ammo > 0);
    let dist = (bf.movers.soldiers[e.id].position - soldier.position).norm();
    if dist > SHOOT_DISTANCE {
        let st = get_status(bf, soldier);
        ai_goto(&bf.movers.soldiers[e.id].position, soldier, st)
    } else {
        if soldier.shot_timer <= 0.0 {
            Action::ShootAction(soldier.id, e)
        } else {
            Action::NoAction(soldier.id)
        }
    }
}

// board: goto truck, then board it.
struct AiBoard {
    targetpos: Vector3<f64>,
}

// board task - constructor
impl AiBoard {
    fn new(tgt: Vector3<f64>) -> AiBoard {
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
            if gameutil::dist(soldier, &self.targetpos) < 3.0 {
                if let Some(truck) = free_truck_nearby(soldier, bf) {
                    Some(Action::BoardAction(soldier.id, truck.1))
                } else {
                    None
                }
            } else {
                Some(ai_goto(&self.targetpos, soldier, st))
            }
        }
    }
}

// drive: goto given point and stay.
struct AiDrive {
    targetpos: Vector3<f64>,
}

// drive task - constructor
impl AiDrive {
    fn new(tgt: Vector3<f64>) -> AiDrive {
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
            let ref truck = bf.movers.trucks[tid.id];
            let tgt_vec = gameutil::to_vec_on_map(soldier, &self.targetpos);
            let dist_to_tgt = tgt_vec.norm();
            let norm_tgt_vec = tgt_vec.normalize();
            if dist_to_tgt >= 3.0 {
                let curr_dir = truck.direction.normalize();
                let mut ang = f64::acos(norm_tgt_vec.dot(&curr_dir));
                let cross = norm_tgt_vec.cross(&curr_dir);
                if Vector3::new(0.0, 1.0, 0.0).dot(&cross) > 0.0 {
                    ang = -ang;
                }
                let gas = gameutil::clamp(0.0, 1.0, dist_to_tgt * 0.002) -
                    gameutil::clamp(0.0, 0.8, f64::abs(ang));
                Some(Action::DriveAction(soldier.id, ang, gas))
            } else {
                if truck.speed < 0.1 {
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

fn have_enough_passengers(bf: &Battlefield, truck: &Truck) -> bool {
    return num_passengers(bf, truck) > 3;
}

// sleep: do nothing for a while.
struct AiSleep {
    time: f64, // seconds
}

// sleep task - constructor
impl AiSleep {
    fn new(time: f64) -> AiSleep {
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


