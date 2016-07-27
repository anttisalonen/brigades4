extern crate rand;
extern crate rustc_serialize;
extern crate noise;

use noise::{Brownian2, Seed};
use na::{Vector3, Norm, Cross, Determinant, Matrix2};

use geom;
use gameutil;
use prim;

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

pub struct Ground {
    pub height: [[f64; prim::GROUND_NUM_TILES as usize]; prim::GROUND_NUM_TILES as usize],
    pub forest: [[f64; prim::GROUND_NUM_TILES as usize]; prim::GROUND_NUM_TILES as usize],
}


pub fn init_ground(param: &GroundParams) -> Ground {
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
        height: [[0.0; prim::GROUND_NUM_TILES as usize]; prim::GROUND_NUM_TILES as usize],
        forest: [[0.0; prim::GROUND_NUM_TILES as usize]; prim::GROUND_NUM_TILES as usize],
    };

    for j in 0..prim::GROUND_NUM_TILES as usize {
        for i in 0..prim::GROUND_NUM_TILES as usize {
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
    get_landscape_geometry(prim::GROUND_NUM_TILES, prim::TILE_SIZE, |x, y| get_height_at_i(ground, x, y),
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
    get_landscape_geometry(prim::GROUND_NUM_TILES / 4, prim::TILE_SIZE * 4.0, |_, _| 0.0,
                           |_, _| (0.0, 0.0, 0.9))
}

fn get_height_at_i(ground: &Ground, x: i32, y: i32) -> f64 {
    let ix = gameutil::clamp_i(0, prim::GROUND_NUM_TILES - 1, x) as usize;
    let iy = gameutil::clamp_i(0, prim::GROUND_NUM_TILES - 1, y) as usize;
    return ground.height[ix][iy];
}

macro_rules! interpolate_at {
    ( $ground:expr, $xp:expr, $yp:expr, $field:ident ) => {
        {
            let x = ($xp + prim::HDIM) / prim::TILE_SIZE;
            let y = ($yp + prim::HDIM) / prim::TILE_SIZE;
            let ix = gameutil::clamp_i(0, prim::GROUND_NUM_TILES - 2, x as i32) as usize;
            let iy = gameutil::clamp_i(0, prim::GROUND_NUM_TILES - 2, y as i32) as usize;
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


