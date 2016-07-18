extern crate glium;
extern crate rand;
extern crate nalgebra as na;

use std;

use na::{Vector2, Vector3, Norm, Rotation3, Cross};

use ai;
use geom;

pub struct Camera {
    pub position:  Vector3<f32>,
    pub direction: Vector3<f32>,
    pub upvec:     Vector3<f32>,
    pub speed:     Vector3<f32>,
}

fn clamp(a: i32, b: i32, x: i32) -> i32 {
    return std::cmp::max(std::cmp::min(b, x), a);
}

pub struct Soldier {
    pub position: Vector3<f32>,
    pub direction: f32,
    pub alive: bool,
    pub side: bool,
    pub id: usize,
    pub shot_timer: f32,
    pub reap_timer: f32,
}

pub const GROUND_SIZE: i32 = 64;
pub const TILE_SIZE:   f32 = 16.0;

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
    let ix = clamp(0, GROUND_SIZE - 1, x) as usize;
    let iy = clamp(0, GROUND_SIZE - 1, y) as usize;
    return ground.height[ix][iy];
}

pub fn get_height_at(ground: &Ground, x: f32, y: f32) -> f32 {
    let ix = clamp(0, GROUND_SIZE - 2, x as i32) as usize;
    let iy = clamp(0, GROUND_SIZE - 2, y as i32) as usize;
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

const WINNER_TIMER: f32 = 3.0;

pub struct Battlefield {
    pub display: glium::Display,
    pub camera: Camera,
    pub mouse_look: bool,
    pub prev_mouse_position: Option<(i32, i32)>,
    pub soldiers: Vec<Soldier>,
    pub ground: Ground,
    pub curr_time: f64,
    pub frame_time: f64,
    winner: Option<bool>,
    winner_timer: f32,
    pub flag_position: Vector2<f32>,
}

pub struct GameState {
    pub bf: Battlefield,
    ai: AiState,
}

impl GameState {
    pub fn new(d: glium::Display) -> GameState {
        let ground = init_ground();
        let fx = 200.0;
        let fy = 200.0;
        let cpx = fx;
        let cpz = fy - 200.0;
        let cpy = f32::max(0.0, get_height_at(&ground,
                                              cpx / TILE_SIZE,
                                              cpz / TILE_SIZE)) + 200.0;
        let mut gs = {
            let bf = Battlefield {
                display: d,
                camera: Camera {
                    position:  Vector3::new(cpx, cpy, cpz),
                    direction: Vector3::new(0.0, -0.707, 0.707),
                    upvec:     Vector3::new(0.0, 1.0, 0.0),
                    speed:     Vector3::new(0.0, 0.0, 0.0),
                },
                mouse_look: false,
                prev_mouse_position: None,
                soldiers: vec![],
                ground: init_ground(),
                curr_time: 0.0,
                frame_time: 0.0,
                winner: None,
                winner_timer: WINNER_TIMER,
                flag_position: Vector2::new(fx, fy),
            };
            let ai = AiState {
                soldier_ai: vec![],
            };

            GameState {
                bf: bf,
                ai: ai
            }
        };
        init_soldiers(&mut gs);
        gs
    }
}

pub fn won(game_state: &GameState) -> Option<bool> {
    return game_state.bf.winner;
}

pub fn update_game_state(game_state: &mut GameState, frame_time: f64) -> bool {
    game_state.bf.frame_time = frame_time;
    game_state.bf.curr_time += frame_time;
    update_soldiers(game_state);
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
                                                                      &mut game_state.ai.soldier_ai, true),
                    glium::glutin::VirtualKeyCode::B => spawn_soldier(game_state.bf.camera.position,
                                                                      &mut game_state.bf.soldiers,
                                                                      &mut game_state.ai.soldier_ai, false),
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

fn check_winner(game_state: &mut GameState) -> () {
    let mut holding = [false, false];
    for sold in game_state.bf.soldiers.iter() {
        if sold.alive {
            let dist = (game_state.bf.flag_position - Vector2::<f32>::new(sold.position.x, sold.position.z)).norm();
            if dist < 20.0 {
                holding[if sold.side { 0 } else { 1 }] = true;
            }
        }
    }
    if holding[0] ^ holding[1] {
        game_state.bf.winner_timer -= game_state.bf.frame_time as f32;
        if game_state.bf.winner_timer <= 0.0 {
            game_state.bf.winner = Some(if holding[0] { true } else { false });
        }
    } else {
        game_state.bf.winner_timer = WINNER_TIMER;
    }
}

fn update_soldiers(mut game_state: &mut GameState) -> () {
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
    if bf.soldiers[from].shot_timer <= 0.0 {
        bf.soldiers[from].shot_timer = 1.0;
        let hit_num = rand::random::<f32>();
        if hit_num < 0.9 {
            bf.soldiers[to].alive = false;
        }
        println!("{} shoots at {}! {}", from, to, hit_num);
    }
}

fn init_soldiers(gs: &mut GameState) -> () {
    for side in [false, true].iter() {
        for i in 0..10 {
            let xp = if *side { 10.0 } else { gs.bf.flag_position.x * 2.0 - 10.0 };
            let yp = 0.0;
            let zp = i as f32 * 10.0 + gs.bf.flag_position.y;
            spawn_soldier(Vector3::new(xp, yp, zp),
                          &mut gs.bf.soldiers,
                          &mut gs.ai.soldier_ai, *side);
        }
    }
}

fn spawn_soldier(pos: Vector3<f32>, soldiers: &mut Vec<Soldier>, soldier_ai: &mut Vec<ai::SoldierAI>, side: bool) -> () {
    let s = Soldier {
        position: pos,
        direction: 0.0,
        alive: true,
        side: side,
        id: soldiers.len(),
        shot_timer: 0.0,
        reap_timer: 10.0,
    };
    soldiers.push(s);
    soldier_ai.push(ai::SoldierAI::new());
}

