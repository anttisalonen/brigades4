extern crate nalgebra as na;
extern crate core;
extern crate rand;

use std::collections::VecDeque;
use self::core::iter::FromIterator;

use na::{Vector3, Norm, Dot, Cross};

use bf_info::*;

use gameutil;
use prim;
use navmap;

const SHOOT_DISTANCE: f64 = 100.0;
const REPLAN_TIME: f64       = TIME_MULTIPLIER as f64 * 3600.0; // seconds
const FOOD_FETCH_BUFFER: f64 = TIME_MULTIPLIER as f64 * 480.0; // seconds

// data structures
pub struct SoldierAI {
    tasks: VecDeque<AiTask>,
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

macro_rules! find_best_path {
    ( $ground:expr, $soldier:expr, $items:expr, $name:expr ) => {
        {
            let maybe_paths = $items.iter().map(|item| navmap::find_path($ground, $soldier.position, item.pos(), $name, 2000));
            let zm = maybe_paths.zip($items.iter());
            let zm = zm.filter(|&(ref p, _)| p.is_some());
            let zm = zm.map(|(p, i)| (p.unwrap(), i));
            let mut zm = Vec::from_iter(zm);
            zm.sort_by(|&(ref p1, _), &(ref p2, _)| p1.len().partial_cmp(&p2.len()).unwrap());
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
            &mut AiTask::Taxi(ref mut g)  => g.update(soldier, bf),
        }
    }
}

impl SoldierAI {
    pub fn new() -> SoldierAI {
        SoldierAI {
            tasks: VecDeque::new(),
            replan_timer: rand::random::<f64>() * REPLAN_TIME,
        }
    }
}

pub fn soldier_ai_update(sai: &mut SoldierAI, soldier: &Soldier, bf: &Battlefield) -> Action {
    sai.replan_timer -= bf.frame_time;
    if sai.replan_timer <= 0.0 {
        sai.replan_timer += REPLAN_TIME * 0.5 + REPLAN_TIME * (rand::random::<f64>());
        sai.tasks = VecDeque::new();
    }
    if sai.tasks.len() == 0 {
        sai.tasks.append(&mut ai_arbitrate_task(soldier, bf));
    }
    let num_tasks = sai.tasks.len();
    if num_tasks > 0 {
        let act = sai.tasks.front_mut().unwrap().update(soldier, bf);
        match act {
            None    => { sai.tasks.pop_front(); (); }
            Some(a) => return a,
        }
    }
    return Action::NoAction(soldier.id);
}

fn find_nearest_supply(bf: &Battlefield, s: &Soldier, max_dist: f64) -> Option<Vector3<f64>> {
    let zm = sort_by_distance!(s, &bf.supply_points);
    for (dist, supply) in zm {
        if dist > max_dist {
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

fn ai_arbitrate_task(s: &Soldier, bf: &Battlefield) -> VecDeque<AiTask> {
    let st = get_status(bf, s);

    let time_with_food = EAT_TIME * (s.food - 1) as f64;

    if time_with_food < FOOD_FETCH_BUFFER || s.ammo < 5 {
        match st {
            Status::Driving(_) => drive_path(navmap::find_path(&bf.ground, s.position, get_base_position(bf, s.side), "ai:driving:nearest supply", 2000), false),
            _                  => {
                if let Some(pos) = find_nearest_supply(bf, s, 2000.0) {
                    vec![AiTask::Goto(AiGoto::new(pos))].into_iter().collect()
                } else if let Some(pos) = find_nearest_supply(bf, s, 30000.0) {
                    walk_path(s, bf, pos, "ai:walking:nearest supply")
                } else {
                    vec![AiTask::Sleep(AiSleep::new(game_minutes(10.0)))].into_iter().collect()
                }
            },
        }
    } else {
        match st {
            Status::Driving(tid) => {
                let ref truck = bf.movers.trucks[tid.id];
                if have_enough_passengers(bf, truck) {
                    // taxi
                    drive_path(flag_target_position(s, bf), true)
                } else {
                    // back home
                    let bp = get_base_position(bf, s.side);
                    let dist = gameutil::dist(s, &bp);
                    if dist < 50.0 {
                        vec![AiTask::Sleep(AiSleep::new(game_minutes(60.0)))].into_iter().collect()
                    } else {
                        drive_path(navmap::find_path(&bf.ground, s.position, bp, "ai:arbitrate:driving:back home", 2000), false)
                    }
                }
            },
            Status::Boarded(_) => taxi_to_dest(),
            Status::OnFoot     => {
                let walk_dist = flag_nearby(s, bf, 2000.0);
                match walk_dist {
                    Some(pos) => vec![AiTask::Goto(AiGoto::new(pos))].into_iter().collect(),
                    None      => {
                        if let Some(truck) = free_truck_nearby(s, bf) {
                            vec![AiTask::Board(AiBoard::new(truck.0))].into_iter().collect()
                        } else if let Some(pos) = flag_nearby(s, bf, 30000.0) {
                            walk_path(s, bf, pos, "ai:walking:nearest flag")
                        } else {
                            vec![AiTask::Sleep(AiSleep::new(game_minutes(10.0)))].into_iter().collect()
                        }
                    }
                }
            }
        }
    }
}

fn taxi_to_dest() -> VecDeque<AiTask> {
    vec![AiTask::Taxi(AiTaxi::new())].into_iter().collect()
}

fn path_to_tasks<F>(mpath: Option<navmap::Path>, to_task: F, stop_before_end: bool, direct_if_no_path: Option<Vector3<f64>>) -> VecDeque<AiTask>
    where F: Fn(Vector3<f64>) -> AiTask {
    match mpath {
        None       => match direct_if_no_path {
            Some(pos) => vec![to_task(pos)].into_iter().collect(),
            None      => vec![AiTask::Sleep(AiSleep::new(game_minutes(60.0)))].into_iter().collect()
        },
        Some(path) => {
            let mut ret: VecDeque<AiTask> = path.iter().map(|node| {
                to_task(Vector3::new(node.x as f64, 0.0, node.y as f64))
            }).collect();
            if stop_before_end && path.len() > 3 {
                ret.pop_back();
                ret.pop_back();
            }
            ret
        }
    }
}

fn walk_path(s: &Soldier, bf: &Battlefield, pos: Vector3<f64>, msg: &str) -> VecDeque<AiTask> {
    let mpath = navmap::find_path(&bf.ground, s.position, pos, msg, 100);
    path_to_tasks(mpath, |p| AiTask::Goto(AiGoto::new(p)), false, Some(pos))
}

fn drive_path(mpath: Option<navmap::Path>, stop_before_end: bool) -> VecDeque<AiTask> {
    path_to_tasks(mpath, |p| AiTask::Drive(AiDrive::new(p)), stop_before_end, None)
}

// generic helpers
fn free_truck_nearby(s: &Soldier, bf: &Battlefield) -> Option<(Vector3<f64>, TruckID)> {
    let zm = sort_by_distance!(s, &bf.movers.trucks);
    for (dist, truck) in zm {
        if dist > 5000.0 {
            return None;
        }
        if !truck_free(bf, &truck) {
            continue;
        }
        return Some((truck.position, truck.id));
    }
    return None;
}

fn flag_nearby(sold: &Soldier, bf: &Battlefield, max_dist: f64) -> Option<Vector3<f64>> {
    let zm = sort_by_distance!(sold, &bf.flags);
    for (dist, flag) in zm {
        if dist > max_dist {
            return None;
        }
        if interesting_flag(sold, bf, &flag) {
            return Some(flag.position);
        }
    }
    return None;
}

fn interesting_flag(sold: &Soldier, bf: &Battlefield, flag: &prim::Flag) -> bool {
    match flag.flag_state {
        prim::FlagState::Free                            => true,
        prim::FlagState::Transition(s) if s == sold.side => false,
        prim::FlagState::Transition(_)                   => true,
        prim::FlagState::Owned(s) if s == sold.side      => flag_lone_holder(sold, bf, &flag.position),
        prim::FlagState::Owned(_)                        => true,
    }
}

fn flag_target_position(sold: &Soldier, bf: &Battlefield) -> Option<navmap::Path> {
    let zm = find_best_path!(&bf.ground, sold, &bf.flags, "ai:nearest flag");
    for (path, flag) in zm {
        if interesting_flag(sold, bf, &flag) {
            return Some(path);
        }
    }
    None
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
    Taxi(AiTaxi),
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
                ai_goto(&self.targetpos, soldier, st)
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

// taxi: sleep until the vehicle has stopped.
struct AiTaxi {
    wait_time: f64,
}

fn game_minutes(m: f64) -> f64 {
    TIME_MULTIPLIER as f64 * m
}

impl AiTaxi {
    fn new() -> AiTaxi {
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
                    if flag_nearby(soldier, bf, 1000.0).is_some() {
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


