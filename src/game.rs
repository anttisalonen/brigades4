extern crate glium;
extern crate rand;
extern crate nalgebra as na;
extern crate noise;
extern crate rustc_serialize;

use std;
use std::collections::HashMap;

use na::{Vector3, Norm, Rotation3, Cross, Matrix2, Determinant};

use noise::{Brownian2, Seed};

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

#[derive(PartialEq, Eq, Copy, Clone, Hash)]
pub struct SoldierID {
    pub id: usize,
}

pub struct Soldier {
    pub position: Vector3<f64>,
    pub direction: f64,
    pub alive: bool,
    pub side: Side,
    pub id: SoldierID,
    pub shot_timer: f64,
    pub reap_timer: f64,
    pub ammo: i32,
    pub food: i32,
    pub eat_timer: f64,
}

#[derive(PartialEq, Eq, Copy, Clone, Hash)]
pub struct TruckID {
    pub id: usize,
}

pub struct Truck {
    pub position: Vector3<f64>,
    pub speed: f64,
    pub direction: Vector3<f64>,
    pub alive: bool,
    pub side: Side,
    pub id: TruckID,
    pub reap_timer: f64,
    pub fuel: f64,
}

#[derive(PartialEq, Eq, Copy, Clone)]
pub enum BoardRole {
    Driver,
    Passenger,
}

struct Boarded {
    sid: SoldierID,
    role: BoardRole,
}

pub struct BoardedMap {
    map: HashMap<TruckID, Vec<Boarded>>,
}

pub const GROUND_NUM_TILES: i32 = 128;
pub const TILE_SIZE:   f64 = 1024.0;

pub const DIM:  f64 = GROUND_NUM_TILES as f64 * TILE_SIZE;
pub const HDIM: f64 = DIM * 0.5;

const CAM_SPEED_FACTOR_TAC: f32 = 20.0;
const CAM_SPEED_FACTOR_STR: f32 = 150.0;
const CAM_SPEED: f32            = 100.0;

// times in seconds
pub const TIME_MULTIPLIER: i32 = 60;
const REINFORCEMENT_TIME: i32  = TIME_MULTIPLIER * 60;
pub const EAT_TIME: f64        = TIME_MULTIPLIER as f64 * 479.0;
const DAY_TIME: f64            = TIME_MULTIPLIER as f64 * 60.0 * 24.0;
const SUPPLY_TIME: i32         = TIME_MULTIPLIER * 1; // how often are supplies picked up

const MAX_SOLDIERS_PER_SIDE: i32 = 40;
pub const SOLDIER_SPEED: f64 = 1.3; // m/s

pub const SUPPLY_DISTANCE: f64 = 5.0; // distance where supply can be picked up
const SUPPLY_MAX_FOOD: i32 = 800;
const SUPPLY_MAX_AMMO: i32 = 4000;

const SOLDIER_MAX_FOOD: i32 = 8;
const SOLDIER_MAX_AMMO: i32 = 40;

const MAX_TRUCKS_PER_SIDE: i32 = 8;
const TRUCK_NUM_PASSENGERS: i32 = 8;
const MAX_TRUCK_SPEED_GRASS:  f64 = 18.0;
const MAX_TRUCK_SPEED_FOREST: f64 = 2.0;

pub trait Locatable {
    fn pos(&self) -> Vector3<f64>;
}

impl Locatable for Soldier {
    fn pos(&self) -> Vector3<f64> {
        self.position
    }
}

impl Locatable for SupplyPoint {
    fn pos(&self) -> Vector3<f64> {
        self.position
    }
}

impl Locatable for Truck {
    fn pos(&self) -> Vector3<f64> {
        self.position
    }
}

impl Locatable for Flag {
    fn pos(&self) -> Vector3<f64> {
        self.position
    }
}

pub struct Ground {
    height: [[f64; GROUND_NUM_TILES as usize]; GROUND_NUM_TILES as usize],
    forest: [[f64; GROUND_NUM_TILES as usize]; GROUND_NUM_TILES as usize],
}

fn init_ground(param: &GroundParams) -> Ground {
    let ground_seed = Seed::new(param.ground.seed as u32);
    let forest_seed = Seed::new(param.forest.seed as u32);

    let ground_noise = Brownian2::new(noise::perlin2, param.ground.octaves as usize).wavelength(param.ground.wavelength).persistence(param.ground.persistence);
    let forest_noise = Brownian2::new(noise::perlin2, param.forest.octaves as usize).wavelength(param.forest.wavelength).persistence(param.forest.persistence);

    build_ground(|x, y| (ground_noise.apply(&ground_seed,
                                           &[param.loc_x + x / param.frequency_x,
                                             param.loc_y + y / param.frequency_y]) + param.ground.bias) * param.ground.scale,
                 |x, y| (forest_noise.apply(&forest_seed,
                                           &[param.loc_x + x / param.frequency_x,
                                             param.loc_y + y / param.frequency_y]) + param.forest.bias) * param.forest.scale)
}

fn build_ground<F, G>(height: F, forest: G) -> Ground
    where F: Fn(f64, f64) -> f64,
          G: Fn(f64, f64) -> f64 {
    let mut g: Ground = Ground {
        height: [[0.0; GROUND_NUM_TILES as usize]; GROUND_NUM_TILES as usize],
        forest: [[0.0; GROUND_NUM_TILES as usize]; GROUND_NUM_TILES as usize],
    };

    for j in 0..GROUND_NUM_TILES as usize {
        for i in 0..GROUND_NUM_TILES as usize {
            g.height[i][j] = gameutil::clamp(-500.0, 4000.0, height(i as f64, j as f64));
            g.forest[i][j] = gameutil::clamp(0.0, 1.0, forest(i as f64, j as f64));
        }
    }
    g
}

pub fn get_landscape_geometry<F, G>(num_tiles: i32, scale: f64, height: F, color: G) -> geom::Geom
    where F : Fn(i32, i32) -> f64,
          G : Fn(f32, f32) -> (f32, f32, f32) {
    let gu = num_tiles as usize;
    let mut geo = geom::new_geom(gu * gu, (gu - 1) * (gu - 1) * 6);
    for j in 0..gu {
        for i in 0..gu {
            let rx = (i as f64 * scale - num_tiles as f64 * scale * 0.5) as f32;
            let rz = (j as f64 * scale - num_tiles as f64 * scale * 0.5) as f32;
            geo.vertices[j * gu + i] = geom::Vertex{position:
                (rx, height(i as i32, j as i32) as f32, rz)
            };

            let dy_x = height(i as i32 + 1, j as i32)     - height(i as i32 - 1, j as i32);
            let dy_z = height(i as i32    , j as i32 + 1) - height(i as i32,     j as i32 - 1);
            let norm_x = Vector3::new((2.0 * scale) as f32, dy_x as f32, 0.0);
            let norm_z = Vector3::new(0.0,                  dy_z as f32, (2.0 * scale) as f32);
            let norm = norm_z.cross(&norm_x);
            let norm = norm.normalize();

            let col = color(rx, rz);

            geo.normals[j * gu + i] = geom::Normal{normal: (norm.x, norm.y, norm.z)};
            geo.colors[j * gu + i] = geom::Color{color: (col.0, col.1, col.2)};
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

pub fn get_ground_geometry(ground: &Ground) -> geom::Geom {
    get_landscape_geometry(GROUND_NUM_TILES, TILE_SIZE, |x, y| get_height_at_i(ground, x, y),
        |x, y| get_ground_color_at(ground, x, y))
}

fn get_ground_color_at(ground: &Ground, x: f32, y: f32) -> (f32, f32, f32) {
    let forest = get_forest_at(ground, x as f64, y as f64);
    let r = gameutil::mix(0.2, 0.05, forest) as f32;
    let g = gameutil::mix(0.8, 0.15, forest) as f32;
    let b = gameutil::mix(0.2, 0.05, forest) as f32;
    (r, g, b)
}

pub fn get_water_geometry() -> geom::Geom {
    get_landscape_geometry(GROUND_NUM_TILES / 4, TILE_SIZE * 4.0, |_, _| 0.0,
                           |_, _| (0.0, 0.0, 0.9))
}

pub fn get_height_at_i(ground: &Ground, x: i32, y: i32) -> f64 {
    let ix = gameutil::clamp_i(0, GROUND_NUM_TILES - 1, x) as usize;
    let iy = gameutil::clamp_i(0, GROUND_NUM_TILES - 1, y) as usize;
    return ground.height[ix][iy];
}

macro_rules! interpolate_at {
    ( $ground:expr, $xp:expr, $yp:expr, $field:ident ) => {
        {
            let x = ($xp + HDIM) / TILE_SIZE;
            let y = ($yp + HDIM) / TILE_SIZE;
            let ix = gameutil::clamp_i(0, GROUND_NUM_TILES - 2, x as i32) as usize;
            let iy = gameutil::clamp_i(0, GROUND_NUM_TILES - 2, y as i32) as usize;
            let fx = f64::fract(x);
            let fy = f64::fract(y);
            let h1 = $ground.$field[ix + 0][iy + 0];
            let h2 = $ground.$field[ix + 1][iy + 0];
            let h3 = $ground.$field[ix + 0][iy + 1];
            let h4 = $ground.$field[ix + 1][iy + 1];
            // barycentric coordinates
            let ((x1, y1), (x2, y2), (x3, y3), (nh1, nh2, nh3)) = {
                if fy < 1.0 - fx {
                    ((0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (h1, h3, h2))
                } else {
                    ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (h2, h3, h4))
                }
            };
            let t = Matrix2::new(x1 - x3, x2 - x3,
                                 y1 - y3, y2 - y3);
            let det = t.determinant();
            let lam1 = ((y2 - y3) * (fx - x3) + (x3 - x2) * (fy - y3)) / det;
            let lam2 = ((y3 - y1) * (fx - x3) + (x1 - x3) * (fy - y3)) / det;
            let lam3 = 1.0 - lam1 - lam2;
            lam1 * nh1 + lam2 * nh2 + lam3 * nh3
        }
    };
}

pub fn get_height_at(ground: &Ground, x: f64, y: f64) -> f64 {
    interpolate_at!(ground, x, y, height)
}

pub fn get_forest_at(ground: &Ground, x: f64, y: f64) -> f64 {
    interpolate_at!(ground, x, y, forest)
}

struct AiState {
    soldier_ai: Vec<ai::SoldierAI>,
}

const FLAG_TIMER: f64 = 10.0;

pub enum FlagState {
    Free,
    Transition(Side),
    Owned(Side),
}

pub struct Flag {
    pub position: Vector3<f64>,
    pub flag_state: FlagState,
    flag_timer: f64,
}

pub struct SupplyPoint {
    pub position: Vector3<f64>,
    pub amount_food: i32,
    pub amount_ammo: i32,
}

impl SupplyPoint {
    fn add_food(&mut self, i: i32) {
        self.amount_food += i;
        self.amount_food = std::cmp::min(self.amount_food, SUPPLY_MAX_FOOD);
    }

    fn add_ammo(&mut self, i: i32) {
        self.amount_ammo += i;
        self.amount_ammo = std::cmp::min(self.amount_ammo, SUPPLY_MAX_AMMO);
    }
}

#[derive(PartialEq, Eq, Copy, Clone)]
pub enum ViewMode {
    Normal,
    Tactical,
    Strategic,
}

pub struct Battlefield {
    pub display: glium::Display,
    pub camera: Camera,
    pub view_mode: ViewMode,
    pub mouse_look: bool,
    pub prev_mouse_position: Option<(i32, i32)>,
    pub soldiers: Vec<Soldier>,
    pub ground: Ground,
    pub curr_time: f64,
    pub frame_time: f64,
    winner: Option<Side>,
    pub flags: Vec<Flag>,
    time_accel: i32,
    rng: rand::StdRng,
    base_position: [Vector3<f64>; 2],
    pub supply_points: Vec<SupplyPoint>,
    pub trucks: Vec<Truck>,
    pub boarded_map: BoardedMap,
    pause: bool,
}

pub struct GameState {
    pub bf: Battlefield,
    ai: AiState,
}

#[derive(RustcDecodable, RustcEncodable)]
pub struct GroundItemParams {
    seed: u64,
    octaves: u64,
    wavelength: f64,
    persistence: f64,
    bias: f64,
    scale: f64,
}

#[derive(RustcDecodable, RustcEncodable)]
pub struct GroundParams {
    ground: GroundItemParams,
    forest: GroundItemParams,
    frequency_x: f64,
    frequency_y: f64,
    loc_x: f64,
    loc_y: f64,
}

#[derive(RustcDecodable, RustcEncodable)]
pub struct GameParams {
    seed: u64,
    ground_params: GroundParams,
}

impl GameState {
    pub fn new(d: glium::Display, game_params: &GameParams) -> GameState {
        let seed = game_params.seed;
        let mut rng = rand::StdRng::from_seed(&[seed as usize]);
        let ground = init_ground(&game_params.ground_params);

        let mut flag_positions = Vec::new();
        loop {
            let xp = (rng.gen::<f64>() * 0.8 + 0.1) * DIM - HDIM;
            let zp = (rng.gen::<f64>() * 0.8 + 0.1) * DIM - HDIM;
            let yp = get_height_at(&ground, xp, zp);
            if yp > 1.0 {
                flag_positions.push(Vector3::new(xp, yp, zp));
            }
            if flag_positions.len() == 10 {
                break;
            }
        }
        let flags = flag_positions.into_iter().map(|p| Flag {
            position: p,
            flag_state: FlagState::Free,
            flag_timer: FLAG_TIMER,
        }).collect();

        let bx0 = -HDIM + 20.0;
        let bx1 =  HDIM - 20.0;
        let bz  = 0.0;
        let by0 = get_height_at(&ground, bx0, bz);
        let by1 = get_height_at(&ground, bx1, bz);
        let base_positions = [
            Vector3::new(bx0, by0, bz),
            Vector3::new(bx1, by1, bz),
        ];

        let mut gs = {
            let bf = Battlefield {
                display: d,
                camera: Camera {
                    position:  Vector3::new(-HDIM as f32,
                                            (get_height_at(&ground, -HDIM, 0.0) + 70.0) as f32,
                                            0.0),
                    direction: Vector3::new(1.0, 0.0, 0.0),
                    upvec:     Vector3::new(0.0, 1.0, 0.0),
                    speed:     Vector3::new(0.0, 0.0, 0.0),
                },
                view_mode: ViewMode::Normal,
                mouse_look: false,
                prev_mouse_position: None,
                soldiers: vec![],
                ground: ground,
                curr_time: TIME_MULTIPLIER as f64 * 360.0,
                frame_time: 0.0,
                winner: None,
                flags: flags,
                time_accel: 1,
                rng: rng,
                base_position: base_positions,
                supply_points: base_positions.iter().map(|p| SupplyPoint {
                    position: *p,
                    amount_food: 0,
                    amount_ammo: 0,
                }).collect(),
                trucks: vec![],
                boarded_map: BoardedMap{map: HashMap::new()},
                pause: false,
            };
            let ai = AiState {
                soldier_ai: vec![],
            };

            GameState {
                bf: bf,
                ai: ai
            }
        };
        spawn_soldiers(&mut gs, 10);
        spawn_trucks(&mut gs, 1);
        gs
    }
}

pub fn won(game_state: &GameState) -> Option<Side> {
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

    for ev in game_state.bf.display.poll_events() {
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
                    glium::glutin::VirtualKeyCode::Key1 => game_state.bf.view_mode = ViewMode::Normal,
                    glium::glutin::VirtualKeyCode::Key2 => game_state.bf.view_mode = ViewMode::Tactical,
                    glium::glutin::VirtualKeyCode::Key3 => game_state.bf.view_mode = ViewMode::Strategic,
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
                let dist = gameutil::dist(&sold, &flag.position);
                if dist < 20.0 {
                    holding[if sold.side == Side::Blue { 0 } else { 1 }] = true;
                }
            }
        }

        if holding[0] ^ holding[1] {
            flag.flag_timer -= gs.bf.frame_time as f64;
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
    if game_state.bf.winner != None {
        return;
    }

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
        execute_action(&action, &mut game_state.bf, prev_curr_time);
    }
    for ref mut sold in &mut game_state.bf.soldiers {
        sold.position.y = f64::max(0.0, get_height_at(&game_state.bf.ground, sold.position.x, sold.position.z));
    }
    for ref mut truck in &mut game_state.bf.trucks {
        truck.position.y = f64::max(0.0, get_height_at(&game_state.bf.ground, truck.position.x, truck.position.z));
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
                game_state.bf.soldiers[i].eat_timer += EAT_TIME;
                game_state.bf.soldiers[i].food -= 1;
                if game_state.bf.soldiers[i].food < 0 {
                    println!("Soldier starved!");
                    kill_soldier(&mut game_state.bf, SoldierID{id: i});
                }
            }
        }
    }
    if reaped {
        for i in 0..game_state.bf.soldiers.len() {
            let old_id = game_state.bf.soldiers[i].id;
            let id = SoldierID{id: i};
            game_state.bf.soldiers[i].id = id;
            update_boarded(&mut game_state.bf.boarded_map, old_id, id);
        }
    }
}

fn update_boarded(boarded_map: &mut BoardedMap, old_id: SoldierID, new_id: SoldierID) -> () {
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

fn execute_action(action: &ai::Action, bf: &mut Battlefield, prev_curr_time: f64) -> () {
    match action {
        &ai::Action::NoAction(s)            => idle_soldier(bf, s, prev_curr_time),
        &ai::Action::MoveAction(s, diff)    => move_soldier(bf, s, diff),
        &ai::Action::ShootAction(from, to)  => shoot_soldier(from, to, bf),
        &ai::Action::BoardAction(s, tr)     => board_truck(bf, s, tr),
        &ai::Action::DriveAction(s, st, ga) => drive_truck(bf, s, st, ga),
        &ai::Action::DisembarkAction(s)     => disembark_truck(bf, s),
    }
}

fn move_soldier(bf: &mut Battlefield, sid: SoldierID, diff: Vector3<f64>) -> () {
    if soldier_boarded(bf, sid) != None {
        return;
    }
    let ref mut s = bf.soldiers[sid.id];
    s.position += gameutil::truncate(diff, SOLDIER_SPEED * bf.frame_time as f64);
}

fn idle_soldier(bf: &mut Battlefield, sid: SoldierID, prev_curr_time: f64) -> () {
    let check_supply = has_tick(bf, prev_curr_time, SUPPLY_TIME);
    let ref mut soldier = bf.soldiers[sid.id];
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

fn kill_soldier(bf: &mut Battlefield, sid: SoldierID) -> () {
    bf.soldiers[sid.id].alive = false;
    disembark_truck(bf, sid);
}

fn shoot_soldier(from: SoldierID, to: SoldierID, mut bf: &mut Battlefield) {
    if soldier_boarded(bf, from) != None {
        return;
    }

    if bf.soldiers[from.id].ammo <= 0 {
        return;
    }

    if bf.soldiers[from.id].shot_timer <= 0.0 {
        bf.soldiers[from.id].shot_timer = 1.0;
        bf.soldiers[from.id].ammo -= 1;
        let dist = gameutil::dist(&bf.soldiers[from.id], &bf.soldiers[to.id].position);
        let threshold = if dist > 100.0 { 0.0 } else { -dist * 0.005 + 1.0 };
        let hit_num = bf.rng.gen::<f64>();
        if hit_num < threshold {
            kill_soldier(&mut bf, to);
        }
        println!("{} shoots at {}! {} ({} - threshold was {})",
        from.id, to.id, hit_num, !bf.soldiers[to.id].alive, threshold);
    }
}

pub fn truck_free_by_id(bf: &Battlefield, truck: TruckID) -> bool {
    if let Some(ref bds) = bf.boarded_map.map.get(&truck) {
        if bds.len() < TRUCK_NUM_PASSENGERS as usize + 1 {
            return true;
        } else {
            return false;
        }
    }

    return true;
}

pub fn truck_free(bf: &Battlefield, truck: &Truck) -> bool {
    truck_free_by_id(bf, truck.id)
}

fn board_truck(bf: &mut Battlefield, sid: SoldierID, tid: TruckID) -> () {
    if truck_free_by_id(bf, tid) &&
        gameutil::dist(&bf.soldiers[sid.id], &bf.trucks[tid.id].position) < 3.0 {
        set_boarded(bf, sid, tid);
    }
}

fn drive_truck(bf: &mut Battlefield, sid: SoldierID, steering: f64, gas: f64) -> () {
    let mtid = soldier_boarded(bf, sid);
    if mtid == None {
        return;
    }

    let ref mut truck = bf.trucks[mtid.unwrap().0.id];
    truck.direction = na::rotate(
        &Rotation3::new(Vector3::new(0.0, 0.2 * gameutil::clamp(-1.0, 1.0, steering) * bf.frame_time, 0.0)),
        &truck.direction);
    truck.speed += gameutil::clamp(0.0,  1.0, gas) * 2.0  * bf.frame_time;
    truck.speed += gameutil::clamp(-1.0, 0.0, gas) * 10.0 * bf.frame_time;
    let forest = get_forest_at(&bf.ground, truck.position.x, truck.position.z);
    let max_speed = gameutil::mix(MAX_TRUCK_SPEED_GRASS, MAX_TRUCK_SPEED_FOREST, forest);
    truck.speed = gameutil::clamp(0.0, max_speed, truck.speed);
    truck.position += truck.direction * truck.speed * bf.frame_time;
}

fn disembark_truck(bf: &mut Battlefield, sid: SoldierID) -> () {
    let mid = soldier_boarded(bf, sid);
    match mid {
        None => (),
        Some((tid, role)) => {
            unset_boarded(bf, sid);
            if role == BoardRole::Driver {
                let boarded = bf.boarded_map.map.get_mut(&tid).unwrap();
                if boarded.len() > 0 {
                    boarded[0].role = BoardRole::Driver;
                }
            }
        },
    }
}

pub fn soldier_boarded(bf: &Battlefield, s: SoldierID) -> Option<(TruckID, BoardRole)> {
    for (tid, bds) in &bf.boarded_map.map {
        for bd in bds {
            if bd.sid == s {
                return Some((*tid, bd.role));
            }
        }
    }
    return None;
}

fn set_boarded(mut bf: &mut Battlefield, sid: SoldierID, tid: TruckID) -> () {
    if soldier_boarded(bf, sid) != None {
        println!("{} tried to board but is already boarded", sid.id);
        return;
    }

    match bf.boarded_map.map.get_mut(&tid) {
        Some(bds) => {
            let ln = bds.len();
            let role = if ln == 0 {
                BoardRole::Driver
            } else {
                BoardRole::Passenger
            };
            if ln < TRUCK_NUM_PASSENGERS as usize + 1 {
                bds.push(Boarded{sid: sid, role: role });
                println!("{} embarked truck {}!", sid.id, tid.id);
            } else {
                println!("{} tried to board but failed", sid.id);
            }
            return;
        },
        None => (),
    }
    bf.boarded_map.map.insert(tid, vec![Boarded{sid: sid, role: BoardRole::Driver}]);
    println!("{} embarked truck {}!", sid.id, tid.id);
}

fn unset_boarded(mut bf: &mut Battlefield, sid: SoldierID) -> () {
    for (_, ref mut bds) in &mut bf.boarded_map.map {
        bds.retain(|ref mut bd| bd.sid != sid);
    }
}

fn add_supplies(gs: &mut GameState) -> () {
    for i in 0..2 {
        gs.bf.supply_points[i].add_food(10);
        gs.bf.supply_points[i].add_ammo(100);
    }
}

fn spawn_soldiers(gs: &mut GameState, num: i32) -> () {
    for side in [Side::Blue, Side::Red].iter() {
        let num_soldiers = gs.bf.soldiers.iter().filter(|s| s.side == *side).count() as i32;
        let mut pos = gs.bf.base_position[if *side == Side::Red { 1 } else { 0 }];
        for _ in 0..(std::cmp::min(num, MAX_SOLDIERS_PER_SIDE - num_soldiers)) {
            pos.z += 10.0;
            spawn_soldier(pos,
                          &mut gs.bf.soldiers,
                          &mut gs.ai.soldier_ai, *side);
        }
    }
}

fn spawn_trucks(gs: &mut GameState, num: i32) -> () {
    for side in [Side::Blue, Side::Red].iter() {
        let num_trucks = gs.bf.trucks.iter().filter(|s| s.side == *side).count() as i32;
        let mut pos = gs.bf.base_position[if *side == Side::Red { 1 } else { 0 }];
        pos.x += 100.0;
        for _ in 0..(std::cmp::min(num, MAX_TRUCKS_PER_SIDE - num_trucks)) {
            pos.z += 20.0;
            pos.y = get_height_at(&gs.bf.ground, pos.x, pos.z);
            spawn_truck(pos,
                        &mut gs.bf.trucks,
                        *side);
        }
    }
}

fn spawn_soldier(pos: Vector3<f64>, soldiers: &mut Vec<Soldier>, soldier_ai: &mut Vec<ai::SoldierAI>, side: Side) -> () {
    let s = Soldier {
        position: pos,
        direction: 0.0,
        alive: true,
        side: side,
        id: SoldierID{id: soldiers.len()},
        shot_timer: 0.0,
        reap_timer: 10.0,
        ammo: 40,
        food: 8,
        eat_timer: EAT_TIME,
    };
    soldiers.push(s);
    soldier_ai.push(ai::SoldierAI::new());
}

fn spawn_truck(pos: Vector3<f64>, trucks: &mut Vec<Truck>, side: Side) -> () {
    let s = Truck {
        position: pos,
        speed: 0.0,
        direction: Vector3::new(0.0, 0.0, 1.0),
        alive: true,
        side: side,
        id: TruckID{id: trucks.len()},
        reap_timer: 10.0,
        fuel: 40.0,
    };
    trucks.push(s);
}

fn change_time_accel(time_accel: i32, incr: bool) -> i32 {
    if time_accel <= 1 && !incr {
        return time_accel;
    }
    if time_accel > 80 * TIME_MULTIPLIER && incr {
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

// true every <tick> seconds
fn has_tick(bf: &Battlefield, prev_curr_time: f64, tick: i32) -> bool {
    let pt = prev_curr_time as u64 / tick as u64;
    let ct = bf.curr_time   as u64 / tick as u64;
    pt != ct
}

fn spawn_reinforcements(mut gs: &mut GameState, prev_curr_time: f64) -> () {
    if has_tick(&gs.bf, prev_curr_time, REINFORCEMENT_TIME) {
        spawn_soldiers(&mut gs, 2);
        spawn_trucks(&mut gs, 1);
        add_supplies(&mut gs);
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

fn cam_speed(vm: ViewMode) -> f32 {
    match vm {
        ViewMode::Normal    => CAM_SPEED,
        ViewMode::Tactical  => CAM_SPEED * CAM_SPEED_FACTOR_TAC,
        ViewMode::Strategic => CAM_SPEED * CAM_SPEED_FACTOR_STR,
    }
}

pub fn num_passengers(bf: &Battlefield, truck: &Truck) -> i32 {
    match bf.boarded_map.map.get(&truck.id) {
        None => 0,
        Some(b) => std::cmp::max(0, b.len() as i32 - 1),
    }
}

pub fn get_base_position(bf: &Battlefield, side: Side) -> Vector3<f64> {
    bf.base_position[if side == Side::Red { 1 } else { 0 }]
}
