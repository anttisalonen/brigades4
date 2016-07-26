use std;

use na::{Vector3, Norm, BaseFloat};
use game::{Soldier};

pub fn to_vec_on_map(s1: &Soldier, tgt: &Vector3<f64>) -> Vector3<f64> {
    let diff_vec = *tgt - s1.position;
    Vector3::new(diff_vec.x, 0.0, diff_vec.z)
}

/*
pub fn dist_on_map(s1: &Soldier, tgt: &Vector3<f64>) -> f64 {
    to_vec_on_map(s1, tgt).norm()
}
*/

pub fn to_vec(s1: &Soldier, tgt: &Vector3<f64>) -> Vector3<f64> {
    *tgt - s1.position
}

pub fn dist(s1: &Soldier, tgt: &Vector3<f64>) -> f64 {
    to_vec(s1, tgt).norm()
}

pub fn clamp_i(a: i32, b: i32, x: i32) -> i32 {
    return std::cmp::max(std::cmp::min(b, x), a);
}

pub fn clamp<T: BaseFloat>(a: T, b: T, x: T) -> T {
    return T::max(T::min(b, x), a);
}

pub fn truncate<T>(v: Vector3<T>, len: T) -> Vector3<T>
    where T: BaseFloat {
    let vl = v.norm();
    if vl < len {
        v
    } else {
        v.normalize() * len
    }
}

pub fn mix(x: f64, y: f64, a: f64) -> f64 {
    x * (1.0 - a) + y * a
}


