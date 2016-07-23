use std;

use na::{Vector3, Norm};
use game::{Soldier};

pub fn to_vec_on_map(s1: &Soldier, tgt: &Vector3<f32>) -> Vector3<f32> {
    let diff_vec = *tgt - s1.position;
    Vector3::new(diff_vec.x, 0.0, diff_vec.z)
}

/*
pub fn dist_on_map(s1: &Soldier, tgt: &Vector3<f32>) -> f32 {
    to_vec_on_map(s1, tgt).norm()
}
*/

pub fn to_vec(s1: &Soldier, tgt: &Vector3<f32>) -> Vector3<f32> {
    *tgt - s1.position
}

pub fn dist(s1: &Soldier, tgt: &Vector3<f32>) -> f32 {
    to_vec(s1, tgt).norm()
}

pub fn clamp_i(a: i32, b: i32, x: i32) -> i32 {
    return std::cmp::max(std::cmp::min(b, x), a);
}

pub fn clamp(a: f32, b: f32, x: f32) -> f32 {
    return f32::max(f32::min(b, x), a);
}

pub fn truncate(v: Vector3<f32>, len: f32) -> Vector3<f32> {
    let vl = v.norm();
    if vl < len {
        v
    } else {
        v.normalize() * len
    }
}

