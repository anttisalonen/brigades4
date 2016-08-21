extern crate nalgebra as na;
extern crate core;
extern crate rand;

use std::collections::VecDeque;

use na::{Vector3};

use bf_info::*;

use navmap;
use gameutil;
use ai_task::*;
use ai_prim::*;

pub fn walk_to_supply(_: &mut SideAI, _: &mut SoldierAI, s: &Soldier, bf: &Battlefield) -> VecDeque<AiTask> {
    if let Some(pos) = find_nearest_supply(bf, s, 2000.0) {
        vec![AiTask::Goto(AiGoto::new(pos))].into_iter().collect()
    } else if let Some(pos) = find_nearest_supply(bf, s, 30000.0) {
        walk_path(s, bf, pos, "ai:walking:nearest supply")
    } else {
        vec![AiTask::Sleep(AiSleep::new(game_minutes(10.0)))].into_iter().collect()
    }
}

pub fn drive_to_food(_: &mut SideAI, sai: &mut SoldierAI, s: &Soldier, bf: &Battlefield) -> VecDeque<AiTask> {
    let tid = sai.my_vehicle;
    assert!(tid.is_some());
    let tid = tid.unwrap();
    let prof = bf.movers.search_profile(tid);
    // TODO: won't work for boats
    drive_path(bf.navmap.find_path(&bf.ground, s.position,
                                   get_base_or_naval_position(bf, s.side, tid),
                                   "ai:driving:nearest supply", 2000, prof),
                                   false)
}

pub fn walk_path(s: &Soldier, bf: &Battlefield, pos: Vector3<f64>, msg: &str) -> VecDeque<AiTask> {
    let mpath = bf.navmap.find_path(&bf.ground, s.position, pos, msg, 100, navmap::SearchProfile::Land);
    path_to_tasks(mpath, |p| AiTask::Goto(AiGoto::new(p)), false, Some(pos))
}

pub fn drive_path(mpath: Option<navmap::Path>, stop_before_end: bool) -> VecDeque<AiTask> {
    path_to_tasks(mpath, |p| AiTask::Drive(AiDrive::new(p)), stop_before_end, None)
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

pub fn drive_to_flag(_: &mut SideAI, sai: &mut SoldierAI, s: &Soldier, bf: &Battlefield) -> VecDeque<AiTask> {
    let tid = sai.my_vehicle.unwrap();
    let prof = bf.movers.search_profile(tid);
    drive_path(flag_target_position(s, bf, prof), true)
}

pub fn drive_to_base(_: &mut SideAI, sai: &mut SoldierAI, s: &Soldier, bf: &Battlefield) -> VecDeque<AiTask> {
    let tid = sai.my_vehicle.unwrap();
    let bp = get_base_or_naval_position(bf, s.side, tid);
    let dist = gameutil::dist(s, &bp);
    if dist < 50.0 {
        vec![AiTask::Sleep(AiSleep::new(game_minutes(60.0)))].into_iter().collect()
    } else {
        let prof = bf.movers.search_profile(tid);
        drive_path(bf.navmap.find_path(&bf.ground, s.position, bp, "ai:arbitrate:driving:back home", 2000, prof), false)
    }
}

pub fn get_driven_to_destination(_: &mut SideAI, _: &mut SoldierAI, _: &Soldier, _: &Battlefield) -> VecDeque<AiTask> {
    vec![AiTask::Taxi(AiTaxi::new())].into_iter().collect()
}

pub fn walk_to_nearest_flag(_: &mut SideAI, sai: &mut SoldierAI, s: &Soldier, bf: &Battlefield) -> VecDeque<AiTask> {
    match sai.nearby_flag {
        Some((pos, dist)) => {
            if dist < 2000.0 {
                vec![AiTask::Goto(AiGoto::new(pos))].into_iter().collect()
            } else {
                // TODO: risk of not finding path
                walk_path(s, bf, pos, "ai:walking:nearest flag")
            }
        },
        None => { assert!(false, "no flag"); vec![].into_iter().collect() }
    }
}

pub fn board_nearby_vehicle(_: &mut SideAI, sai: &mut SoldierAI, _: &Soldier, _: &Battlefield) -> VecDeque<AiTask> {
    let veh = sai.nearby_vehicle.unwrap();
    vec![AiTask::Board(AiBoard::new(veh.0))].into_iter().collect()
}

pub fn rest(_: &mut SideAI, _: &mut SoldierAI, _: &Soldier, _: &Battlefield) -> VecDeque<AiTask> {
    vec![AiTask::Sleep(AiSleep::new(game_minutes(10.0)))].into_iter().collect()
}

