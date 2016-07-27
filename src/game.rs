extern crate glium;
extern crate rand;
extern crate nalgebra as na;

use std;

use na::{Vector3, Rotation3};

use ai;
use gameutil;
use prim;
use bf_info;
use terrain;
use actions;

const CAM_SPEED_FACTOR_TAC: f32 = 20.0;
const CAM_SPEED_FACTOR_STR: f32 = 150.0;
const CAM_SPEED: f32            = 100.0;

// times in seconds
const REINFORCEMENT_TIME: i32  = bf_info::TIME_MULTIPLIER * 60;
const DAY_TIME: f64            = bf_info::TIME_MULTIPLIER as f64 * 60.0 * 24.0;

const MAX_SOLDIERS_PER_SIDE: i32 = 40;

const MAX_TRUCKS_PER_SIDE: i32 = 8;

struct AiState {
    soldier_ai: Vec<ai::SoldierAI>,
}

pub struct GameState {
    pub display: glium::Display,
    pub bf: bf_info::Battlefield,
    ai: AiState,
}

#[derive(RustcDecodable, RustcEncodable)]
pub struct GameParams {
    seed: u64,
    ground_params: terrain::GroundParams,
}

impl GameState {
    pub fn new(d: glium::Display, game_params: &GameParams) -> GameState {
        let seed = game_params.seed;
        let bf = bf_info::Battlefield::new(seed as usize, &game_params.ground_params);

        let mut gs = {
            GameState {
                display: d,
                bf: bf,
                ai: AiState {
                    soldier_ai: vec![],
                }
            }
        };
        spawn_soldiers(&mut gs, 10);
        spawn_trucks(&mut gs, 1);
        gs
    }
}

pub fn won(game_state: &GameState) -> Option<prim::Side> {
    return game_state.bf.winner;
}

pub fn update_game_state(game_state: &mut GameState, frame_time: f64) -> bool {
    if !game_state.bf.pause {
        for _ in 0..game_state.bf.time_accel {
            game_state.bf.frame_time = frame_time;
            let prev_curr_time = game_state.bf.curr_time;
            game_state.bf.curr_time += frame_time;
            spawn_reinforcements(game_state, prev_curr_time);
            update_soldiers(game_state, prev_curr_time);
            check_winner(game_state);
        }
    }

    game_state.bf.camera.position += na::rotate(
        &Rotation3::new_observer_frame(&game_state.bf.camera.direction,
                                       &game_state.bf.camera.upvec),
        &(game_state.bf.camera.speed * cam_speed(game_state.bf.view_mode) * frame_time as f32));

    for ev in game_state.display.poll_events() {
        match ev {
            glium::glutin::Event::Closed => return false,
            glium::glutin::Event::KeyboardInput(
                glium::glutin::ElementState::Pressed,
                _, Some(key)) => {
                match key {
                    glium::glutin::VirtualKeyCode::Escape => return false,
                    glium::glutin::VirtualKeyCode::W => game_state.bf.camera.speed.z = 1.0,
                    glium::glutin::VirtualKeyCode::S => game_state.bf.camera.speed.z = -1.0,
                    glium::glutin::VirtualKeyCode::A => game_state.bf.camera.speed.x = -1.0,
                    glium::glutin::VirtualKeyCode::D => game_state.bf.camera.speed.x = 1.0,
                    glium::glutin::VirtualKeyCode::Q => game_state.bf.camera.speed.y = 1.0,
                    glium::glutin::VirtualKeyCode::E => game_state.bf.camera.speed.y = -1.0,
                    glium::glutin::VirtualKeyCode::Add      => game_state.bf.time_accel = change_time_accel(game_state.bf.time_accel, true),
                    glium::glutin::VirtualKeyCode::Subtract => game_state.bf.time_accel = change_time_accel(game_state.bf.time_accel, false),
                    glium::glutin::VirtualKeyCode::P => game_state.bf.pause = !game_state.bf.pause,
                    glium::glutin::VirtualKeyCode::I => println!("Position: {}\nTime: {} {}",
                                                                 game_state.bf.camera.position,
                                                                 game_state.bf.curr_time,
                                                                 curr_day_time_str(game_state)),
                    glium::glutin::VirtualKeyCode::Key1 => game_state.bf.view_mode = prim::ViewMode::Normal,
                    glium::glutin::VirtualKeyCode::Key2 => game_state.bf.view_mode = prim::ViewMode::Tactical,
                    glium::glutin::VirtualKeyCode::Key3 => game_state.bf.view_mode = prim::ViewMode::Strategic,
                    _ => ()
                }
            }
            glium::glutin::Event::KeyboardInput(
                glium::glutin::ElementState::Released,
                _, Some(key)) => {
                match key {
                    glium::glutin::VirtualKeyCode::W => game_state.bf.camera.speed.z = 0.0,
                    glium::glutin::VirtualKeyCode::S => game_state.bf.camera.speed.z = 0.0,
                    glium::glutin::VirtualKeyCode::A => game_state.bf.camera.speed.x = 0.0,
                    glium::glutin::VirtualKeyCode::D => game_state.bf.camera.speed.x = 0.0,
                    glium::glutin::VirtualKeyCode::Q => game_state.bf.camera.speed.y = 0.0,
                    glium::glutin::VirtualKeyCode::E => game_state.bf.camera.speed.y = 0.0,
                    _ => ()
                }
            }
            glium::glutin::Event::MouseInput(
                pressed,
                glium::glutin::MouseButton::Left) => {
                game_state.bf.mouse_look = pressed == glium::glutin::ElementState::Pressed;
                game_state.bf.prev_mouse_position = None;
            }
            glium::glutin::Event::MouseMoved(x, y) =>
            {
                if game_state.bf.mouse_look {
                    match game_state.bf.prev_mouse_position {
                        Some((px, py)) => {
                            let dx = x - px;
                            let dy = y - py;
                            let rx = (dx as f32) * 0.01;
                            let ry = (dy as f32) * 0.01;

                            // yaw
                            game_state.bf.camera.direction = na::rotate(
                                &Rotation3::new(Vector3::new(0.0, rx, 0.0)),
                                &game_state.bf.camera.direction);

                            // pitch
                            let x_rot_axis = Vector3::new(game_state.bf.camera.direction.z, 0.0, -game_state.bf.camera.direction.x);
                            game_state.bf.camera.direction = na::rotate(
                                &Rotation3::new(x_rot_axis * ry),
                                &game_state.bf.camera.direction);
                        }
                        _ => ()
                    }
                    game_state.bf.prev_mouse_position = Some((x, y));
                }
            }
            _ => ()
        }
    }

    return true;
}

fn get_actions(ai: &mut AiState, bf: &bf_info::Battlefield) -> Vec<ai::Action> {
    let mut ret = Vec::new();
    for i in 0..bf.soldiers.len() {
        if bf.soldiers[i].alive {
            ret.push(ai::soldier_ai_update(&mut ai.soldier_ai[i], &bf.soldiers[i], &bf));
        }
    }
    return ret;
}

fn check_flags(gs: &mut GameState) -> () {
    for ref mut flag in &mut gs.bf.flags {
        let mut holding = [false, false];
        for sold in gs.bf.soldiers.iter() {
            if sold.alive {
                let dist = gameutil::dist(&sold, &flag.position);
                if dist < 20.0 {
                    holding[if sold.side == prim::Side::Blue { 0 } else { 1 }] = true;
                }
            }
        }

        if holding[0] ^ holding[1] {
            flag.flag_timer -= gs.bf.frame_time as f64;
            let s = if holding[0] { prim::Side::Blue } else { prim::Side::Red };
            if flag.flag_timer <= 0.0 {
                flag.flag_state = prim::FlagState::Owned(s);
            } else {
                flag.flag_state = prim::FlagState::Transition(s);
            }
        } else {
            flag.flag_state = prim::FlagState::Free;
            flag.flag_timer = prim::FLAG_TIMER;
        }
    }
}

fn check_winner(game_state: &mut GameState) -> () {
    if game_state.bf.winner != None {
        return;
    }

    check_flags(game_state);
    let mut holding: [usize; 2] = [0, 0];
    for flag in game_state.bf.flags.iter() {
        match flag.flag_state {
            prim::FlagState::Owned(prim::Side::Blue) => holding[0] += 1,
            prim::FlagState::Owned(prim::Side::Red)  => holding[1] += 1,
            _                            => (),
        }
    }

    if holding[0] == game_state.bf.flags.len() {
        game_state.bf.winner = Some(prim::Side::Blue);
    }

    if holding[1] == game_state.bf.flags.len() {
        game_state.bf.winner = Some(prim::Side::Red);
    }
}

fn update_soldiers(mut game_state: &mut GameState, prev_curr_time: f64) -> () {
    let actions = get_actions(&mut game_state.ai, &game_state.bf);
    for action in actions {
        actions::execute_action(&action, &mut game_state.bf, prev_curr_time);
    }
    for ref mut sold in &mut game_state.bf.soldiers {
        sold.position.y = f64::max(0.0, terrain::get_height_at(&game_state.bf.ground, sold.position.x, sold.position.z));
    }
    for ref mut truck in &mut game_state.bf.trucks {
        truck.position.y = f64::max(0.0, terrain::get_height_at(&game_state.bf.ground, truck.position.x, truck.position.z));
        truck.speed = truck.speed * (1.0 - 0.10 * game_state.bf.frame_time);
    }
    for (tid, bds) in &game_state.bf.boarded_map.map {
        for bd in bds.iter() {
            game_state.bf.soldiers[bd.sid.id].position = game_state.bf.trucks[tid.id].position;
        }
    }

    let mut reaped = false;
    for i in 0..game_state.bf.soldiers.len() {
        if reaped && i >= game_state.bf.soldiers.len() {
            break;
        }
        if ! game_state.bf.soldiers[i].alive {
            game_state.bf.soldiers[i].reap_timer -= game_state.bf.frame_time as f64;
            if game_state.bf.soldiers[i].reap_timer < 0.0 {
                game_state.ai.soldier_ai.swap_remove(i);
                game_state.bf.soldiers.swap_remove(i);
                reaped = true;
            }
        } else {
            game_state.bf.soldiers[i].eat_timer -= game_state.bf.frame_time as f64;
            if game_state.bf.soldiers[i].eat_timer <= 0.0 {
                game_state.bf.soldiers[i].eat_timer += bf_info::EAT_TIME;
                game_state.bf.soldiers[i].food -= 1;
                if game_state.bf.soldiers[i].food < 0 {
                    println!("Soldier starved!");
                    actions::kill_soldier(&mut game_state.bf, bf_info::SoldierID{id: i});
                }
            }
        }
    }
    if reaped {
        for i in 0..game_state.bf.soldiers.len() {
            let old_id = game_state.bf.soldiers[i].id;
            let id = bf_info::SoldierID{id: i};
            game_state.bf.soldiers[i].id = id;
            update_boarded(&mut game_state.bf.boarded_map, old_id, id);
        }
    }
}

fn update_boarded(boarded_map: &mut bf_info::BoardedMap, old_id: bf_info::SoldierID, new_id: bf_info::SoldierID) -> () {
    for (_, mut bds) in &mut boarded_map.map {
        let index = bds.iter_mut().position(|ref mut bd| bd.sid == old_id);
        match index {
            None => (),
            Some(i) => {
                bds[i].sid = new_id;
                return;
            },
        }
    }
}

fn spawn_soldiers(gs: &mut GameState, num: i32) -> () {
    for side in [prim::Side::Blue, prim::Side::Red].iter() {
        let num_soldiers = gs.bf.soldiers.iter().filter(|s| s.side == *side).count() as i32;
        let mut pos = gs.bf.base_position[if *side == prim::Side::Red { 1 } else { 0 }];
        for _ in 0..(std::cmp::min(num, MAX_SOLDIERS_PER_SIDE - num_soldiers)) {
            pos.z += 10.0;
            spawn_soldier(pos,
                          &mut gs.bf.soldiers,
                          &mut gs.ai.soldier_ai, *side);
        }
    }
}

fn spawn_trucks(gs: &mut GameState, num: i32) -> () {
    for side in [prim::Side::Blue, prim::Side::Red].iter() {
        let num_trucks = gs.bf.trucks.iter().filter(|s| s.side == *side).count() as i32;
        let mut pos = gs.bf.base_position[if *side == prim::Side::Red { 1 } else { 0 }];
        pos.x += 100.0;
        for _ in 0..(std::cmp::min(num, MAX_TRUCKS_PER_SIDE - num_trucks)) {
            pos.z += 20.0;
            pos.y = terrain::get_height_at(&gs.bf.ground, pos.x, pos.z);
            spawn_truck(pos,
                        &mut gs.bf.trucks,
                        *side);
        }
    }
}

fn spawn_soldier(pos: Vector3<f64>, soldiers: &mut Vec<bf_info::Soldier>, soldier_ai: &mut Vec<ai::SoldierAI>, side: prim::Side) -> () {
    let s = bf_info::Soldier {
        position: pos,
        direction: 0.0,
        alive: true,
        side: side,
        id: bf_info::SoldierID{id: soldiers.len()},
        shot_timer: 0.0,
        reap_timer: 10.0,
        ammo: 40,
        food: 8,
        eat_timer: bf_info::EAT_TIME,
    };
    soldiers.push(s);
    soldier_ai.push(ai::SoldierAI::new());
}

fn spawn_truck(pos: Vector3<f64>, trucks: &mut Vec<bf_info::Truck>, side: prim::Side) -> () {
    let s = bf_info::Truck {
        position: pos,
        speed: 0.0,
        direction: Vector3::new(0.0, 0.0, 1.0),
        alive: true,
        side: side,
        id: bf_info::TruckID{id: trucks.len()},
        reap_timer: 10.0,
        fuel: 40.0,
    };
    trucks.push(s);
}

fn change_time_accel(time_accel: i32, incr: bool) -> i32 {
    if time_accel <= 1 && !incr {
        return time_accel;
    }
    if time_accel > 80 * bf_info::TIME_MULTIPLIER && incr {
        return time_accel;
    }

    if incr {
        println!("Time acceleration: {}", time_accel * 2);
        time_accel * 2
    } else {
        println!("Time acceleration: {}", time_accel / 2);
        time_accel / 2
    }
}

fn spawn_reinforcements(mut gs: &mut GameState, prev_curr_time: f64) -> () {
    if actions::has_tick(&gs.bf, prev_curr_time, REINFORCEMENT_TIME) {
        spawn_soldiers(&mut gs, 2);
        spawn_trucks(&mut gs, 1);
        bf_info::add_supplies(&mut gs.bf);
        println!("Reinforcements have arrived!");
    }
}

fn curr_day_time_str(gs: &GameState) -> String {
    let dt = curr_day_time(gs);
    let d  = (gs.bf.curr_time as f64 / DAY_TIME) as i32 + 1;
    let h  = dt * 24.0;
    let m  = f64::fract(h) * 60.0;
    let s  = f64::fract(m) * 60.0;
    format!("Day {} {:02}:{:02}:{:02}", d, h as i32, m as i32, s as i32)
}

pub fn curr_day_time(gs: &GameState) -> f64 {
    f64::fract(gs.bf.curr_time as f64 / DAY_TIME)
}

fn cam_speed(vm: prim::ViewMode) -> f32 {
    match vm {
        prim::ViewMode::Normal    => CAM_SPEED,
        prim::ViewMode::Tactical  => CAM_SPEED * CAM_SPEED_FACTOR_TAC,
        prim::ViewMode::Strategic => CAM_SPEED * CAM_SPEED_FACTOR_STR,
    }
}


