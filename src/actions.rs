extern crate nalgebra as na;

use na::{Vector3, Rotation3};

use ai;
use gameutil;
use bf_info;
use terrain;

const SUPPLY_TIME: i32         = bf_info::TIME_MULTIPLIER * 1; // how often are supplies picked up
const SUPPLY_DISTANCE: f64     = 5.0; // distance where supply can be picked up

const SOLDIER_MAX_FOOD: i32 = 8;
const SOLDIER_MAX_AMMO: i32 = 40;

const MAX_TRUCK_SPEED_GRASS:  f64 = 18.0;
const MAX_TRUCK_SPEED_FOREST: f64 = 2.0;

pub fn execute_action(action: &ai::Action, bf: &mut bf_info::Battlefield, prev_curr_time: f64) -> () {
    match action {
        &ai::Action::NoAction(s)            => idle_soldier(bf, s, prev_curr_time),
        &ai::Action::MoveAction(s, diff)    => move_soldier(bf, s, diff),
        &ai::Action::ShootAction(from, to)  => shoot_soldier(from, to, bf),
        &ai::Action::BoardAction(s, tr)     => board_truck(bf, s, tr),
        &ai::Action::DriveAction(s, st, ga) => drive_truck(bf, s, st, ga),
        &ai::Action::DisembarkAction(s)     => disembark_truck(&mut bf.movers.boarded_map, s),
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

fn disembark_all_from_truck(boarded_map: &mut bf_info::BoardedMap, tid: bf_info::TruckID) -> () {
    if let Some(bds) = boarded_map.map.get_mut(&tid) {
        bds.clear();
    }
}

pub fn destroy_truck(boarded_map: &mut bf_info::BoardedMap, truck: &mut bf_info::Truck) -> () {
    truck.alive = false;
    disembark_all_from_truck(boarded_map, truck.id);
}

pub fn kill_soldier(boarded_map: &mut bf_info::BoardedMap, soldier: &mut bf_info::Soldier) -> () {
    soldier.alive = false;
    disembark_truck(boarded_map, soldier.id);
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

fn board_truck(bf: &mut bf_info::Battlefield, sid: bf_info::SoldierID, tid: bf_info::TruckID) -> () {
    if bf_info::truck_free_by_id(bf, tid) &&
        gameutil::dist(&bf.movers.soldiers[sid.id], &bf.movers.trucks[tid.id].position) < 3.0 {
        set_boarded(&mut bf.movers, sid, tid);
    }
}

fn drive_truck(bf: &mut bf_info::Battlefield, sid: bf_info::SoldierID, steering: f64, gas: f64) -> () {
    let mtid = bf_info::soldier_boarded(&bf.movers.boarded_map, sid);
    if mtid == None {
        return;
    }

    let ref mut truck = bf.movers.trucks[mtid.unwrap().0.id];

    if !truck.alive {
        return;
    }

    truck.direction = na::rotate(
        &Rotation3::new(Vector3::new(0.0, 0.2 * gameutil::clamp(-1.0, 1.0, steering) * bf.frame_time, 0.0)),
        &truck.direction);
    truck.speed += gameutil::clamp(0.0,  1.0, gas) * 2.0  * bf.frame_time;
    truck.speed += gameutil::clamp(-1.0, 0.0, gas) * 10.0 * bf.frame_time;
    let forest = terrain::get_forest_at(&bf.ground, truck.position.x, truck.position.z);
    let max_speed = gameutil::mix(MAX_TRUCK_SPEED_GRASS, MAX_TRUCK_SPEED_FOREST, forest);
    truck.speed = gameutil::clamp(0.0, max_speed, truck.speed);
    truck.position += truck.direction * truck.speed * bf.frame_time;
}

fn disembark_truck(boarded_map: &mut bf_info::BoardedMap, sid: bf_info::SoldierID) -> () {
    let mid = bf_info::soldier_boarded(boarded_map, sid);
    match mid {
        None => (),
        Some((tid, role)) => {
            unset_boarded(boarded_map, sid);
            println!("Soldier {} disembarked", sid.id);
            if role == bf_info::BoardRole::Driver {
                let boarded = boarded_map.map.get_mut(&tid).unwrap();
                if boarded.len() > 0 {
                    boarded[0].role = bf_info::BoardRole::Driver;
                }
            }
        },
    }
}

fn set_boarded(mut mov: &mut bf_info::Movers, sid: bf_info::SoldierID, tid: bf_info::TruckID) -> () {
    if bf_info::soldier_boarded(&mov.boarded_map, sid) != None {
        println!("{} tried to board but is already boarded", sid.id);
        return;
    }

    match mov.boarded_map.map.get_mut(&tid) {
        Some(bds) => {
            let ln = bds.len();
            let role = if ln == 0 {
                bf_info::BoardRole::Driver
            } else {
                bf_info::BoardRole::Passenger
            };
            if ln < bf_info::TRUCK_NUM_PASSENGERS as usize + 1 {
                bds.push(bf_info::Boarded{sid: sid, role: role });
                println!("{} embarked truck {}!", sid.id, tid.id);
            } else {
                println!("{} tried to board but failed", sid.id);
            }
            return;
        },
        None => (),
    }
    mov.boarded_map.map.insert(tid, vec![bf_info::Boarded{sid: sid, role: bf_info::BoardRole::Driver}]);
    println!("{} embarked truck {}!", sid.id, tid.id);
}

fn unset_boarded(mut boarded_map: &mut bf_info::BoardedMap, sid: bf_info::SoldierID) -> () {
    for (_, ref mut bds) in &mut boarded_map.map {
        bds.retain(|ref mut bd| bd.sid != sid);
    }
}


