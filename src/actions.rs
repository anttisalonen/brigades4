extern crate nalgebra as na;

use na::{Vector3, Rotation3};

use gameutil;
use bf_info;
use terrain;

const SUPPLY_TIME: i32         = bf_info::TIME_MULTIPLIER * 1; // how often are supplies picked up
const SUPPLY_DISTANCE: f64     = 5.0; // distance where supply can be picked up

const SOLDIER_MAX_FOOD: i32 = 8;
const SOLDIER_MAX_AMMO: i32 = 40;

pub const MAX_BOARD_DISTANCE: f64 = 3.0;

pub fn execute_action(action: &bf_info::Action, bf: &mut bf_info::Battlefield, prev_curr_time: f64) -> () {
    match action {
        &bf_info::Action::NoAction(s)            => idle_soldier(bf, s, prev_curr_time),
        &bf_info::Action::MoveAction(s, diff)    => move_soldier(bf, s, diff),
        &bf_info::Action::ShootAction(from, to)  => shoot_soldier(from, to, bf),
        &bf_info::Action::BoardAction(s, tr)     => board_vehicle(bf, s, tr),
        &bf_info::Action::DriveAction(s, st, ga) => drive_vehicle(bf, s, st, ga),
        &bf_info::Action::DisembarkAction(s)     => disembark_vehicle(&mut bf.movers.boarded_map, s),
    }
}

// true every <tick> seconds
pub fn has_tick(bf: &bf_info::Battlefield, prev_curr_time: f64, tick: i32) -> bool {
    let pt = prev_curr_time as u64 / tick as u64;
    let ct = bf.curr_time   as u64 / tick as u64;
    pt != ct
}

fn move_soldier(bf: &mut bf_info::Battlefield, sid: bf_info::SoldierID, diff: Vector3<f64>) -> () {
    if bf_info::soldier_boarded(&bf.movers.boarded_map, sid) != None {
        return;
    }
    let ref mut s = bf.movers.soldiers[sid.id];
    s.position += gameutil::truncate(diff, bf_info::SOLDIER_SPEED * bf.frame_time as f64);
}

fn idle_soldier(bf: &mut bf_info::Battlefield, sid: bf_info::SoldierID, prev_curr_time: f64) -> () {
    let check_supply = has_tick(bf, prev_curr_time, SUPPLY_TIME);
    let ref mut soldier = bf.movers.soldiers[sid.id];
    if soldier.shot_timer > 0.0 {
        soldier.shot_timer -= bf.frame_time as f64;
    }

    if check_supply {
        for ref mut supply in &mut bf.supply_points {
            let dist = gameutil::dist(soldier, &supply.position);
            if dist < SUPPLY_DISTANCE {
                if soldier.food <= SOLDIER_MAX_FOOD - 1  && supply.amount_food >= 1 {
                    soldier.food += 1;
                    supply.amount_food -= 1;
                }
                if soldier.ammo <= SOLDIER_MAX_AMMO - 10 && supply.amount_ammo >= 10 {
                    soldier.ammo += 10;
                    supply.amount_ammo -= 10;
                }
            }
        }
    }
}

fn disembark_all_from_vehicle(boarded_map: &mut bf_info::BoardedMap, tid: bf_info::VehicleID) -> () {
    if let Some(bds) = boarded_map.map.get_mut(&tid.id) {
        bds.clear();
    }
}

pub fn destroy_vehicle(boarded_map: &mut bf_info::BoardedMap, vehicle: &mut bf_info::Vehicle) -> () {
    vehicle.alive = false;
    disembark_all_from_vehicle(boarded_map, vehicle.id);
}

pub fn kill_soldier(boarded_map: &mut bf_info::BoardedMap, soldier: &mut bf_info::Soldier) -> () {
    soldier.alive = false;
    disembark_vehicle(boarded_map, soldier.id);
}

fn shoot_soldier(from: bf_info::SoldierID, to: bf_info::SoldierID, mut bf: &mut bf_info::Battlefield) {
    if bf_info::soldier_boarded(&bf.movers.boarded_map, from) != None {
        return;
    }

    if bf.movers.soldiers[from.id].ammo <= 0 {
        return;
    }

    if bf.movers.soldiers[from.id].shot_timer <= 0.0 {
        bf.movers.soldiers[from.id].shot_timer = 1.0;
        bf.movers.soldiers[from.id].ammo -= 1;
        let dist = gameutil::dist(&bf.movers.soldiers[from.id], &bf.movers.soldiers[to.id].position);
        let threshold = if dist > 100.0 { 0.0 } else { -dist * 0.005 + 1.0 };
        let hit_num = bf.rand();
        if hit_num < threshold {
            kill_soldier(&mut bf.movers.boarded_map, &mut bf.movers.soldiers[to.id]);
        }
        println!("{} shoots at {}! {} ({} - threshold was {})",
        from.id, to.id, hit_num, !bf.movers.soldiers[to.id].alive, threshold);
    }
}

fn board_vehicle(bf: &mut bf_info::Battlefield, sid: bf_info::SoldierID, tid: bf_info::VehicleID) -> () {
    if bf_info::vehicle_free_by_id(bf, tid) &&
        gameutil::dist(&bf.movers.soldiers[sid.id], &bf.movers.vehicles[tid.id].position) < 3.0 {
        set_boarded(&mut bf.movers, sid, tid);
    }
}

fn drive_vehicle(bf: &mut bf_info::Battlefield, sid: bf_info::SoldierID, steering: f64, gas: f64) -> () {
    let mtid = bf_info::soldier_boarded(&bf.movers.boarded_map, sid);
    if mtid == None {
        return;
    }
    let vid = mtid.unwrap().0;
    let (grass_speed, forest_speed, sea_speed) = {
        let vinfo = bf.movers.get_vehicle_info_from_vehicle_id(vid);
        (vinfo.max_speed_grass, vinfo.max_speed_forest, vinfo.max_speed_sea)
    };
    let ref mut vehicle = bf.movers.vehicles[vid.id];

    if !vehicle.alive {
        return;
    }

    vehicle.direction = na::rotate(
        &Rotation3::new(Vector3::new(0.0, 0.2 * gameutil::clamp(-1.0, 1.0, steering) * bf.frame_time, 0.0)),
        &vehicle.direction);
    vehicle.speed += gameutil::clamp(0.0,  1.0, gas) * 2.0  * bf.frame_time;
    vehicle.speed += gameutil::clamp(-1.0, 0.0, gas) * 10.0 * bf.frame_time;
    let hgt = terrain::get_height_at(&bf.ground, vehicle.position.x, vehicle.position.z);
    let forest = terrain::get_forest_at(&bf.ground, vehicle.position.x, vehicle.position.z);
    let max_land_speed = gameutil::mix(grass_speed, forest_speed, forest);
    let land_speed = gameutil::clamp(0.0, max_land_speed, vehicle.speed);
    let sea_speed = gameutil::clamp(0.0, sea_speed, vehicle.speed);
    if hgt > -2.0 && hgt < 2.0 {
        vehicle.speed = f64::max(land_speed, sea_speed);
    } else if hgt >= 2.0 {
        vehicle.speed = land_speed;
    } else {
        vehicle.speed = sea_speed;
    }
    vehicle.position += vehicle.direction * vehicle.speed * bf.frame_time;
}

fn disembark_vehicle(boarded_map: &mut bf_info::BoardedMap, sid: bf_info::SoldierID) -> () {
    let mid = bf_info::soldier_boarded(boarded_map, sid);
    match mid {
        None => (),
        Some((tid, role)) => {
            unset_boarded(boarded_map, sid);
            println!("Soldier {} disembarked", sid.id);
            if role == bf_info::BoardRole::Driver {
                let boarded = boarded_map.map.get_mut(&tid.id).unwrap();
                if boarded.len() > 0 {
                    boarded[0].role = bf_info::BoardRole::Driver;
                }
            }
        },
    }
}

fn set_boarded(mut mov: &mut bf_info::Movers, sid: bf_info::SoldierID, tid: bf_info::VehicleID) -> () {
    if bf_info::soldier_boarded(&mov.boarded_map, sid) != None {
        println!("{} tried to board but is already boarded", sid.id);
        return;
    }

    let num_pass = mov.get_vehicle_info_from_vehicle_id(tid).num_passengers;
    match mov.boarded_map.map.get_mut(&tid.id) {
        Some(bds) => {
            let ln = bds.len();
            let role = if ln == 0 {
                bf_info::BoardRole::Driver
            } else {
                bf_info::BoardRole::Passenger
            };
            if ln < num_pass as usize + 1 {
                bds.push(bf_info::Boarded{sid: sid, role: role });
                println!("{} embarked vehicle {}!", sid.id, tid.id);
            } else {
                println!("{} tried to board but failed", sid.id);
            }
            return;
        },
        None => (),
    }
    mov.boarded_map.map.insert(tid.id, vec![bf_info::Boarded{sid: sid, role: bf_info::BoardRole::Driver}]);
    println!("{} embarked vehicle {}!", sid.id, tid.id);
}

fn unset_boarded(mut boarded_map: &mut bf_info::BoardedMap, sid: bf_info::SoldierID) -> () {
    for (_, ref mut bds) in &mut boarded_map.map {
        bds.retain(|ref mut bd| bd.sid != sid);
    }
}


