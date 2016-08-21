extern crate nalgebra as na;
extern crate core;
extern crate rand;

use self::core::iter::FromIterator;

use na::{Vector3};

use navmap;
use bf_info::*;
use gameutil;
use prim;

#[macro_export]
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
    ( $navmap:expr, $ground:expr, $soldier:expr, $items:expr, $name:expr, $prof:expr ) => {
        {
            let maybe_paths = $items.iter().map(|item| $navmap.find_path($ground, $soldier.position, item.pos(), $name, 2000, $prof));
            let zm = maybe_paths.zip($items.iter());
            let zm = zm.filter(|&(ref p, _)| p.is_some());
            let zm = zm.map(|(p, i)| (p.unwrap(), i));
            let mut zm = Vec::from_iter(zm);
            zm.sort_by(|&(ref p1, _), &(ref p2, _)| p1.len().partial_cmp(&p2.len()).unwrap());
            zm
        }
    };
}

pub const REPLAN_TIME: f64 = TIME_MULTIPLIER as f64 * 3600.0; // seconds

#[derive(PartialEq, Eq, Copy, Clone)]
#[derive(RustcDecodable, RustcEncodable)]
pub enum Status {
    OnFoot,
    Driving(VehicleID),
    Boarded(VehicleID),
}

pub fn interesting_flag(sold: &Soldier, bf: &Battlefield, flag: &prim::Flag) -> bool {
    match flag.flag_state {
        prim::FlagState::Free                            => true,
        prim::FlagState::Transition(s) if s == sold.side => false,
        prim::FlagState::Transition(_)                   => true,
        prim::FlagState::Owned(s) if s == sold.side      => flag_lone_holder(sold, bf, &flag.position),
        prim::FlagState::Owned(_)                        => true,
    }
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

pub fn get_status(bf: &Battlefield, s: &Soldier) -> Status {
    match soldier_boarded(&bf.movers.boarded_map, s.id) {
        Some((tid, BoardRole::Driver))    => Status::Driving(tid),
        Some((tid, BoardRole::Passenger)) => Status::Boarded(tid),
        _                                 => Status::OnFoot,
    }
}

pub fn find_nearest_supply(bf: &Battlefield, s: &Soldier, max_dist: f64) -> Option<Vector3<f64>> {
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

pub fn flag_target_position(sold: &Soldier, bf: &Battlefield, prof: navmap::SearchProfile) -> Option<navmap::Path> {
    let zm = find_best_path!(&bf.navmap, &bf.ground, sold, &bf.flags, "ai:nearest flag", prof);
    for (path, flag) in zm {
        if interesting_flag(sold, bf, &flag) {
            return Some(path);
        }
    }
    None
}

pub fn flag_nearby(sold: &Soldier, bf: &Battlefield, max_dist: f64) -> Option<(Vector3<f64>, f64)> {
    let zm = sort_by_distance!(sold, &bf.flags);
    for (dist, flag) in zm {
        if dist > max_dist {
            return None;
        }
        if interesting_flag(sold, bf, &flag) {
            return Some((flag.position, dist));
        }
    }
    return None;
}

pub fn free_vehicle_nearby(s: &Soldier, bf: &Battlefield, max_dist: f64) -> Option<(Vector3<f64>, VehicleID, f64)> {
    let zm = sort_by_distance!(s, &bf.movers.vehicles);
    for (dist, vehicle) in zm {
        if dist > max_dist {
            return None;
        }
        if !vehicle_free(bf, &vehicle) {
            continue;
        }
        return Some((vehicle.position, vehicle.id, dist));
    }
    return None;
}


