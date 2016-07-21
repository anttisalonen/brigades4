extern crate glium;
extern crate rand;
extern crate nalgebra as na;

use std;

use na::{Vector2, Vector3, Norm, Rotation3, Cross};

use ai;
use geom;
use gameutil;

use game::rand::{SeedableRng,Rng};

pub struct Camera {
    pub position:  Vector3<f32>,
    pub direction: Vector3<f32>,
    pub upvec:     Vector3<f32>,
    pub speed:     Vector3<f32>,
}

#[derive(PartialEq, Eq, Copy, Clone)]
pub enum Side {
    Blue,
    Red,
}

pub struct Soldier {
    pub position: Vector3<f32>,
    pub direction: f32,
    pub alive: bool,
    pub side: Side,
    pub id: usize,
    pub shot_timer: f32,
    pub reap_timer: f32,
    pub ammo: i32,
    pub food: i32,
}

pub const GROUND_SIZE: i32 = 64;
pub const TILE_SIZE:   f32 = 16.0;

// times in seconds
const REINFORCEMENT_TIME: i32 = 60;
const EAT_TIME: i32           = 479;
const DAY_TIME: f32           = 60.0 * 24.0;

const MAX_SOLDIERS_PER_SIDE: i32 = 40;

pub struct Ground {
    height: [[f32; GROUND_SIZE as usize]; GROUND_SIZE as usize],
}

pub fn init_ground() -> Ground {
    let mut g: Ground = Ground { height: [[0.0; GROUND_SIZE as usize]; GROUND_SIZE as usize] };
    for j in 0..GROUND_SIZE as usize {
        for i in 0..GROUND_SIZE as usize {
            g.height[i][j] =
                f32::sin(i as f32 * 0.3) * 30.0 +
                f32::cos(j as f32 * 0.2) * -25.0 + 30.0;
        }
    }
    g
}

pub fn get_ground_geometry(ground: &Ground) -> geom::Geom {
    let gu = GROUND_SIZE as usize;
    let mut geo = geom::new_geom(gu * gu, (gu - 1) * (gu - 1) * 6);
    for j in 0..gu {
        for i in 0..gu {
            geo.vertices[j * gu + i] = geom::Vertex{position:
                (i as f32 * TILE_SIZE,
                get_height_at_i(ground, i as i32, j as i32),
                j as f32 * TILE_SIZE)
            };

            let dy_x = get_height_at_i(ground, i as i32 + 1, j as i32)     - get_height_at_i(ground, i as i32 - 1, j as i32);
            let dy_z = get_height_at_i(ground, i as i32    , j as i32 + 1) - get_height_at_i(ground, i as i32,     j as i32 - 1);
            let norm_x = Vector3::new(2.0 * TILE_SIZE, dy_x, 0.0);
            let norm_z = Vector3::new(0.0, dy_z, 2.0 * TILE_SIZE);
            let norm = norm_z.cross(&norm_x);
            let norm = norm.normalize();
            geo.normals[j * gu + i] = geom::Normal{normal: (norm.x, norm.y, norm.z)};
        }
    }
    for j in 0..gu - 1 {
        for i in 0..gu - 1 {
            geo.indices[(j * (gu - 1) + i) * 6 + 0] = ((j + 0) * gu + i) as u16;
            geo.indices[(j * (gu - 1) + i) * 6 + 1] = ((j + 1) * gu + i) as u16;
            geo.indices[(j * (gu - 1) + i) * 6 + 2] = ((j + 0) * gu + i + 1) as u16;
            geo.indices[(j * (gu - 1) + i) * 6 + 3] = ((j + 0) * gu + i + 1) as u16;
            geo.indices[(j * (gu - 1) + i) * 6 + 4] = ((j + 1) * gu + i) as u16;
            geo.indices[(j * (gu - 1) + i) * 6 + 5] = ((j + 1) * gu + i + 1) as u16;
        }
    }
    geo
}

pub fn get_height_at_i(ground: &Ground, x: i32, y: i32) -> f32 {
    let ix = gameutil::clamp_i(0, GROUND_SIZE - 1, x) as usize;
    let iy = gameutil::clamp_i(0, GROUND_SIZE - 1, y) as usize;
    return ground.height[ix][iy];
}

pub fn get_height_at(ground: &Ground, x: f32, y: f32) -> f32 {
    let ix = gameutil::clamp_i(0, GROUND_SIZE - 2, x as i32) as usize;
    let iy = gameutil::clamp_i(0, GROUND_SIZE - 2, y as i32) as usize;
    let fx = f32::fract(x);
    let fy = f32::fract(y);
    let h1 = ground.height[ix + 0][iy + 0];
    let h2 = ground.height[ix + 1][iy + 0];
    let h3 = ground.height[ix + 0][iy + 1];
    let h4 = ground.height[ix + 1][iy + 1];
    let ha = h1 + (h2 - h1) * fx;
    let hb = h3 + (h4 - h3) * fx;
    let hc = ha + (hb - ha) * fy;
    return hc;
}

struct AiState {
    soldier_ai: Vec<ai::SoldierAI>,
}

const FLAG_TIMER: f32 = 10.0;

pub enum FlagState {
    Free,
    Transition(Side),
    Owned(Side),
}

pub struct Flag {
    pub position: Vector2<f32>,
    pub flag_state: FlagState,
    flag_timer: f32,
}

pub struct Battlefield {
    pub display: glium::Display,
    pub camera: Camera,
    pub mouse_look: bool,
    pub prev_mouse_position: Option<(i32, i32)>,
    pub soldiers: Vec<Soldier>,
    pub ground: Ground,
    pub curr_time: f64,
    pub frame_time: f64,
    winner: Option<Side>,
    pub flags: Vec<Flag>,
    time_accel: f32,
    rng: rand::StdRng,
}

pub struct GameState {
    pub bf: Battlefield,
    ai: AiState,
}

impl GameState {
    pub fn new(d: glium::Display) -> GameState {
        let seed = std::env::args().last().unwrap_or(String::from("")).parse::<usize>().unwrap_or(21);
        let mut rng = rand::StdRng::from_seed(&[seed]);
        let cpx = GROUND_SIZE as f32 * TILE_SIZE * 0.5;
        let cpy = GROUND_SIZE as f32 * TILE_SIZE * 0.5;
        let mut flag_positions = Vec::new();
        for _ in 0..10 {
            let xp = (rng.gen::<f32>() * 0.8 + 0.1) * GROUND_SIZE as f32 * TILE_SIZE;
            let yp = (rng.gen::<f32>() * 0.8 + 0.1) * GROUND_SIZE as f32 * TILE_SIZE;
            flag_positions.push(Vector2::new(xp, yp));
        }
        let flags = flag_positions.into_iter().map(|p| Flag {
            position: p,
            flag_state: FlagState::Free,
            flag_timer: FLAG_TIMER,
        }).collect();

        let mut gs = {
            let bf = Battlefield {
                display: d,
                camera: Camera {
                    position:  Vector3::new(cpx, cpy, 0.0),
                    direction: Vector3::new(0.0, -0.866, 0.5),
                    upvec:     Vector3::new(0.0, 1.0, 0.0),
                    speed:     Vector3::new(0.0, 0.0, 0.0),
                },
                mouse_look: false,
                prev_mouse_position: None,
                soldiers: vec![],
                ground: init_ground(),
                curr_time: 360.0,
                frame_time: 0.0,
                winner: None,
                flags: flags,
                time_accel: 1.0,
                rng: rng,
            };
            let ai = AiState {
                soldier_ai: vec![],
            };

            GameState {
                bf: bf,
                ai: ai
            }
        };
        init_soldiers(&mut gs, 10);
        gs
    }
}

pub fn won(game_state: &GameState) -> Option<Side> {
    return game_state.bf.winner;
}

pub fn update_game_state(game_state: &mut GameState, frame_time: f64) -> bool {
    game_state.bf.frame_time = frame_time * game_state.bf.time_accel as f64;
    let prev_curr_time = game_state.bf.curr_time;
    game_state.bf.curr_time += frame_time * game_state.bf.time_accel as f64;
    spawn_reinforcements(game_state, prev_curr_time);
    update_soldiers(game_state, prev_curr_time);
    check_winner(game_state);

    game_state.bf.camera.position += na::rotate(
        &Rotation3::new_observer_frame(&game_state.bf.camera.direction,
                                       &game_state.bf.camera.upvec),
        &(game_state.bf.camera.speed * frame_time as f32));

    for ev in game_state.bf.display.poll_events() {
        match ev {
            glium::glutin::Event::Closed => return false,
            glium::glutin::Event::KeyboardInput(
                glium::glutin::ElementState::Pressed,
                _, Some(key)) => {
                match key {
                    glium::glutin::VirtualKeyCode::Escape => return false,
                    glium::glutin::VirtualKeyCode::W => game_state.bf.camera.speed.z = 30.0,
                    glium::glutin::VirtualKeyCode::S => game_state.bf.camera.speed.z = -30.0,
                    glium::glutin::VirtualKeyCode::A => game_state.bf.camera.speed.x = -30.0,
                    glium::glutin::VirtualKeyCode::D => game_state.bf.camera.speed.x = 30.0,
                    glium::glutin::VirtualKeyCode::Q => game_state.bf.camera.speed.y = 30.0,
                    glium::glutin::VirtualKeyCode::E => game_state.bf.camera.speed.y = -30.0,
                    glium::glutin::VirtualKeyCode::M => kill_soldier(&mut game_state.bf.soldiers[0]),
                    glium::glutin::VirtualKeyCode::N => spawn_soldier(game_state.bf.camera.position,
                                                                      &mut game_state.bf.soldiers,
                                                                      &mut game_state.ai.soldier_ai, Side::Red),
                    glium::glutin::VirtualKeyCode::B => spawn_soldier(game_state.bf.camera.position,
                                                                      &mut game_state.bf.soldiers,
                                                                      &mut game_state.ai.soldier_ai, Side::Blue),
                    glium::glutin::VirtualKeyCode::Add      => game_state.bf.time_accel = change_time_accel(game_state.bf.time_accel, true),
                    glium::glutin::VirtualKeyCode::Subtract => game_state.bf.time_accel = change_time_accel(game_state.bf.time_accel, false),
                    glium::glutin::VirtualKeyCode::P => println!("Position: {}\nTime: {} {}",
                                                                 game_state.bf.camera.position,
                                                                 game_state.bf.curr_time,
                                                                 curr_day_time_str(game_state)),
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

fn get_actions(ai: &mut AiState, bf: &Battlefield) -> Vec<ai::Action> {
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
                let dist = (flag.position - Vector2::<f32>::new(sold.position.x, sold.position.z)).norm();
                if dist < 20.0 {
                    holding[if sold.side == Side::Blue { 0 } else { 1 }] = true;
                }
            }
        }

        if holding[0] ^ holding[1] {
            flag.flag_timer -= gs.bf.frame_time as f32;
            let s = if holding[0] { Side::Blue } else { Side::Red };
            if flag.flag_timer <= 0.0 {
                flag.flag_state = FlagState::Owned(s);
            } else {
                flag.flag_state = FlagState::Transition(s);
            }
        } else {
            flag.flag_state = FlagState::Free;
            flag.flag_timer = FLAG_TIMER;
        }
    }
}

fn check_winner(game_state: &mut GameState) -> () {
    check_flags(game_state);
    let mut holding: [usize; 2] = [0, 0];
    for flag in game_state.bf.flags.iter() {
        match flag.flag_state {
            FlagState::Owned(Side::Blue) => holding[0] += 1,
            FlagState::Owned(Side::Red)  => holding[1] += 1,
            _                            => (),
        }
    }

    if holding[0] == game_state.bf.flags.len() {
        game_state.bf.winner = Some(Side::Blue);
    }

    if holding[1] == game_state.bf.flags.len() {
        game_state.bf.winner = Some(Side::Red);
    }
}

fn update_soldiers(mut game_state: &mut GameState, prev_curr_time: f64) -> () {
    let actions = get_actions(&mut game_state.ai, &game_state.bf);
    for action in actions {
        execute_action(&action, &mut game_state.bf);
    }
    for ref mut sold in &mut game_state.bf.soldiers {
        sold.position.y = f32::max(0.0, get_height_at(&game_state.bf.ground, sold.position.x / TILE_SIZE, sold.position.z / TILE_SIZE)) + 0.5;
    }

    let mut reaped = false;
    for i in 0..game_state.bf.soldiers.len() {
        if reaped && i >= game_state.bf.soldiers.len() {
            break;
        }
        if ! game_state.bf.soldiers[i].alive {
            game_state.bf.soldiers[i].reap_timer -= game_state.bf.frame_time as f32;
            if game_state.bf.soldiers[i].reap_timer < 0.0 {
                game_state.ai.soldier_ai.swap_remove(i);
                game_state.bf.soldiers.swap_remove(i);
                reaped = true;
            }
        } else if has_tick(game_state, prev_curr_time, EAT_TIME) {
            game_state.bf.soldiers[i].food -= 1;
            if game_state.bf.soldiers[i].food < 0 {
                game_state.bf.soldiers[i].alive = false;
            }
        }
    }
    if reaped {
        for i in 0..game_state.bf.soldiers.len() {
            game_state.bf.soldiers[i].id = i;
        }
    }
}

fn execute_action(action: &ai::Action, bf: &mut Battlefield) -> () {
    match action {
        &ai::Action::NoAction(s)           => idle_soldier(&mut bf.soldiers[s], bf.frame_time),
        &ai::Action::MoveAction(s, diff)   => bf.soldiers[s].position += diff,
        &ai::Action::ShootAction(from, to) => shoot_soldier(from, to, bf),
    }
}

fn idle_soldier(soldier: &mut Soldier, frame_time: f64) -> () {
    if soldier.shot_timer > 0.0 {
        soldier.shot_timer -= frame_time as f32;
    }
}

fn kill_soldier(soldier: &mut Soldier) -> () {
    soldier.alive = false;
}

fn shoot_soldier(from: usize, to: usize, bf: &mut Battlefield) {
    if bf.soldiers[from].ammo <= 0 {
        return;
    }

    if bf.soldiers[from].shot_timer <= 0.0 {
        bf.soldiers[from].shot_timer = 1.0;
        bf.soldiers[from].ammo -= 1;
        let dist = gameutil::dist(&bf.soldiers[from], &bf.soldiers[to].position);
        let threshold = if dist > 100.0 { 0.0 } else { -dist * 0.005 + 1.0 };
        let hit_num = bf.rng.gen::<f32>();
        if hit_num < threshold {
            bf.soldiers[to].alive = false;
        }
        println!("{} shoots at {}! {} ({} - threshold was {})",
        from, to, hit_num, !bf.soldiers[to].alive, threshold);
    }
}

fn init_soldiers(gs: &mut GameState, num: i32) -> () {
    for side in [Side::Blue, Side::Red].iter() {
        let num_soldiers = gs.bf.soldiers.iter().filter(|s| s.side == *side).count() as i32;
        for i in 0..(std::cmp::min(num, MAX_SOLDIERS_PER_SIDE - num_soldiers)) {
            let xp = if *side == Side::Red { 20.0 } else { GROUND_SIZE as f32 * TILE_SIZE - 20.0 };
            let yp = 0.0;
            let zp = i as f32 * 10.0 + GROUND_SIZE as f32 * TILE_SIZE * 0.5;
            spawn_soldier(Vector3::new(xp, yp, zp),
                          &mut gs.bf.soldiers,
                          &mut gs.ai.soldier_ai, *side);
        }
    }
}

fn spawn_soldier(pos: Vector3<f32>, soldiers: &mut Vec<Soldier>, soldier_ai: &mut Vec<ai::SoldierAI>, side: Side) -> () {
    let s = Soldier {
        position: pos,
        direction: 0.0,
        alive: true,
        side: side,
        id: soldiers.len(),
        shot_timer: 0.0,
        reap_timer: 10.0,
        ammo: 40,
        food: 8,
    };
    soldiers.push(s);
    soldier_ai.push(ai::SoldierAI::new());
}

fn change_time_accel(time_accel: f32, incr: bool) -> f32 {
    if time_accel < 0.2 && !incr {
        return time_accel;
    }
    if time_accel > 100.0 && incr {
        return time_accel;
    }

    if incr {
        println!("Time acceleration: {}", time_accel * 2.0);
        time_accel * 2.0
    } else {
        println!("Time acceleration: {}", time_accel * 0.5);
        time_accel * 0.5
    }
}

// true every <tick> seconds
fn has_tick(gs: &GameState, prev_curr_time: f64, tick: i32) -> bool {
    let pt = prev_curr_time  as u64 / tick as u64;
    let ct = gs.bf.curr_time as u64 / tick as u64;
    pt != ct
}

fn spawn_reinforcements(mut gs: &mut GameState, prev_curr_time: f64) -> () {
    if has_tick(gs, prev_curr_time, REINFORCEMENT_TIME) {
        init_soldiers(&mut gs, 2);
        println!("Reinforcements have arrived!");
    }
}

fn curr_day_time_str(gs: &GameState) -> String {
    let dt = curr_day_time(gs);
    let d  = (gs.bf.curr_time as f32 / DAY_TIME) as i32 + 1;
    let h  = dt * 24.0;
    let m  = f32::fract(h) * 60.0;
    let s  = f32::fract(m) * 60.0;
    format!("Day {} {:02}:{:02}:{:02}", d, h as i32, m as i32, s as i32)
}

pub fn curr_day_time(gs: &GameState) -> f32 {
    f32::fract(gs.bf.curr_time as f32 / DAY_TIME)
}

