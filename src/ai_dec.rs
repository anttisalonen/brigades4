extern crate nalgebra as na;
extern crate core;
extern crate rand;

use bf_info::*;

use ai_task::*;
use ai_prim::*;

const FOOD_FETCH_BUFFER: f64 = TIME_MULTIPLIER as f64 * 480.0; // seconds

pub fn need_food(_: &mut SideAI, _: &mut SoldierAI, s: &Soldier, _: &Battlefield) -> bool {
    let time_with_food = EAT_TIME * (s.food - 1) as f64;
    time_with_food < FOOD_FETCH_BUFFER || s.ammo < 5
}

pub fn am_driving(_: &mut SideAI, sai: &mut SoldierAI, _: &Soldier, _: &Battlefield) -> bool {
    match sai.status {
        Status::Driving(_) => true,
        _                  => false,
    }
}

pub fn have_enough_passengers(_: &mut SideAI, sai: &mut SoldierAI, _: &Soldier, bf: &Battlefield) -> bool {
    let tid = sai.my_vehicle;
    assert!(tid.is_some());
    let tid = tid.unwrap();
    let ref vehicle = bf.movers.vehicles[tid.id];
    num_passengers(bf, vehicle) > 3
}

pub fn am_boarded(_: &mut SideAI, sai: &mut SoldierAI, _: &Soldier, _: &Battlefield) -> bool {
    match sai.status {
        Status::Boarded(_) => true,
        _                  => false,
    }
}

pub fn flag_within_walking_distance(_: &mut SideAI, sai: &mut SoldierAI, _: &Soldier, _: &Battlefield) -> bool {
    match sai.nearby_flag {
        Some((_, d)) => d < 2000.0,
        _            => false,
    }
}

pub fn vehicle_within_walking_distance(_: &mut SideAI, sai: &mut SoldierAI, _: &Soldier, _: &Battlefield) -> bool {
    match sai.nearby_vehicle {
        Some((_, _, d)) => d < 5000.0,
        _               => false,
    }
}

pub fn flag_within_days_march(_: &mut SideAI, sai: &mut SoldierAI, _: &Soldier, _: &Battlefield) -> bool {
    match sai.nearby_flag {
        Some((_, d)) => d < 30000.0,
        _            => false,
    }
}

