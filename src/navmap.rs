extern crate rand;

use na::{Vector2, Vector3};
use std::collections::HashSet;
use std::collections::HashMap;
use std::hash::Hash;
use std::collections::BinaryHeap;
use std::cmp::Ordering;

use self::rand::{Rng,StdRng};

use terrain;
use prim;
use gameutil;
use contmap::*;

const STEP: i64 = prim::TILE_SIZE as i64;

#[derive(PartialEq, Eq, Copy, Clone, Hash)]
pub enum SearchProfile {
    Land,
    Sea,
}

#[derive(RustcDecodable, RustcEncodable)]
pub struct Navmap {
    cmap: ContinentMap,
}

impl Navmap {
    pub fn new(ground: &terrain::Ground) -> Navmap {
        let cm = ContinentMap::new(&ground);

        Navmap {
            cmap: cm,
        }
    }

    pub fn find_config(&self, ground: &terrain::Ground, rand: &mut StdRng) -> NavConfig {
        let flags = self.find_flag_positions(ground, rand);

        assert!(flags.len() > 0);
        let flags_mass: Vec<MassID> = flags.iter().map(|ref f| self.cmap.get_mass(f.position.x, f.position.z)).collect();
        let all_flags_on_same = flags_mass.iter().all(|&m| m == flags_mass[0]);

        let bases = self.find_base_positions(ground, all_flags_on_same, flags_mass, rand);
        let naval_bases = if all_flags_on_same {
            None
        } else if bases.is_some() {
            self.find_naval_positions(bases.unwrap())
        } else {
            None
        };

        NavConfig {
            flags: flags,
            base_positions: bases,
            naval_positions: naval_bases,
        }
    }

    fn find_flag_positions(&self, ground: &terrain::Ground, rand: &mut StdRng) -> Vec<prim::Flag> {
        let mut flag_positions = Vec::new();
        loop {
            let xp = (rand.gen::<f64>() * 0.9 + 0.05) * prim::DIM - prim::HDIM;
            let zp = (rand.gen::<f64>() * 0.9 + 0.05) * prim::DIM - prim::HDIM;
            let yp = terrain::get_height_at(&ground, xp, zp);
            if yp > 10.0 {
                let this_mass = self.cmap.get_mass(xp, zp);
                if this_mass.is_land() {
                    let mreachable = self.cmap.get_reachables(this_mass);
                    match mreachable {
                        None    => (),
                        Some(r) => {
                            if r.iter().position(|&n| n == self.cmap.largest_water).is_some() {
                                flag_positions.push(Vector3::new(xp, yp, zp));
                                if flag_positions.len() == 10 {
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }

        flag_positions.into_iter().map(|p| prim::Flag {
            position: p,
            flag_state: prim::FlagState::Free,
            flag_timer: prim::FLAG_TIMER,
        }).collect()
    }

    fn find_base_positions(&self, ground: &terrain::Ground, all_flags_on_same: bool, flags_mass: Vec<MassID>, rand: &mut StdRng) -> Option<[Vector3<f64>; 2]> {
        if !all_flags_on_same {
            if self.cmap.get_neighbors(self.cmap.largest_water).unwrap_or(&vec![]).len() == 0 {
                return None;
            }
        }
        for _ in 0..200 {
            let (x1, x2, z1, z2) = if all_flags_on_same {
                let x1 = (rand.gen::<f64>() * 0.9 + 0.05) * prim::DIM - prim::HDIM;
                let x2 = (rand.gen::<f64>() * 0.9 + 0.05) * prim::DIM - prim::HDIM;
                let z1 = (rand.gen::<f64>() * 0.9 + 0.05) * prim::DIM - prim::HDIM;
                let z2 = (rand.gen::<f64>() * 0.9 + 0.05) * prim::DIM - prim::HDIM;
                (x1, x2, z1, z2)
            } else {
                let &(px, py) = rand.choose(&self.cmap.get_neighbors(self.cmap.largest_water).unwrap()).unwrap();
                let x1 = terrain::undiscretize(px);
                let z1 = terrain::undiscretize(py);
                let &(px, py) = rand.choose(&self.cmap.get_neighbors(self.cmap.largest_water).unwrap()).unwrap();
                let x2 = terrain::undiscretize(px);
                let z2 = terrain::undiscretize(py);
                (x1, x2, z1, z2)
            };
            let cm1 = self.cmap.get_mass(x1, z1);
            let cm2 = self.cmap.get_mass(x2, z2);

            if all_flags_on_same && cm1 != cm2 {
                continue;
            }
            if all_flags_on_same && cm1 != flags_mass[0] {
                continue;
            }
            if !cm1.is_land() {
                continue;
            }
            if (x2 - x1) + (z2 - z1).abs() < prim::DIM * 0.5 {
                continue;
            }
            if !all_flags_on_same {
                // ensure base island has at least one flag
                if flags_mass.iter().position(|&n| n == cm1).is_none() {
                    continue;
                }
                if flags_mass.iter().position(|&n| n == cm2).is_none() {
                    continue;
                }
            }

            let y1 = terrain::get_height_at(&ground, x1, z1);
            let y2 = terrain::get_height_at(&ground, x2, z2);
            if y1 < 10.0 || y2 < 10.0 {
                continue;
            }

            let bp = [
                        Vector3::new(x1, y1, z1),
                        Vector3::new(x2, y2, z2),
            ];
            if !all_flags_on_same {
                return Some(bp);
            } else {
                let path = self.find_path(&ground, bp[0], bp[1], "base_position", 20000, SearchProfile::Land);

                match path {
                    None    => (),
                    Some(_) => return Some(bp),
                }
            }
        }
        None
    }

    fn find_naval_positions(&self, bases: [Vector3<f64>; 2]) -> Option<[Vector3<f64>; 2]> {
        let mut ps = [None, None];
        for _ in 0..200 {
            ps[0] = None;
            ps[1] = None;
            for i in 0..2 {
                let x = terrain::discretize(bases[i].x);
                let y = terrain::discretize(bases[i].z);
                for spot in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)].iter() {
                    let &(sx, sy) = spot;
                    let fx = terrain::undiscretize(sx);
                    let fy = terrain::undiscretize(sy);
                    let cm = self.cmap.get_mass(fx, fy);
                    if cm == self.cmap.largest_water {
                        ps[i] = Some((fx, fy));
                    }
                }
            }
            if ps[0].is_some() && ps[1].is_some() {
                return Some([
                            Vector3::new(ps[0].unwrap().0, 2.0, ps[0].unwrap().1),
                            Vector3::new(ps[1].unwrap().0, 2.0, ps[1].unwrap().1),
                ]);
            }
        }
        None // TODO
    }

    pub fn find_path(&self, ground: &terrain::Ground, p1: Vector3<f64>, p2: Vector3<f64>, user: &str, limit: usize, prof: SearchProfile) -> Option<Path> {
        if prof == SearchProfile::Land {
            let mass1 = self.cmap.get_mass(p1.x, p1.z);
            let mass2 = self.cmap.get_mass(p2.x, p2.z);
            if mass1 != mass2 || !mass1.is_land() {
                return None;
            }
        }

        let pn1 = vec_to_grid(p1);
        let pn2 = vec_to_grid(p2);

        let graph = |n: &Vector2<i64>| {
            let mut ret = Vec::new();
            let mut j = -STEP;
            while j <= STEP {
                let mut i = -STEP;
                while i <= STEP {
                    if i == 0 && j == 0 {
                        i += STEP;
                        continue;
                    }

                    let pos = Vector2::new(n.x + i as i64, n.y + j as i64);
                    i += STEP;
                    let hgt = terrain::get_height_at(ground, pos.x as f64, pos.y as f64);
                    match prof {
                        SearchProfile::Land => 
                            if hgt < 50.0 {
                                continue;
                            },
                        SearchProfile::Sea => 
                            if hgt > -50.0 {
                                continue;
                            },
                    }
                    if pos.x <= -prim::HDIM as i64 || pos.x >= prim::HDIM as i64 ||
                       pos.y <= -prim::HDIM as i64 || pos.y >= prim::HDIM as i64 {
                           continue;
                    }

                    ret.push(pos);
                }
                j += STEP;
            }
            ret
        };

        let calc_dist = |n1: &Vector2<i64>, n2: &Vector2<i64>| {
            let dx = n2.x - n1.x;
            let dy = n2.y - n1.y;
            ((dx * dx + dy * dy) as f64).sqrt() * 0.001
        };

        let distance = |n1: &Vector2<i64>, n2: &Vector2<i64>| {
            let hgt = terrain::get_height_at(ground, n2.x as f64, n2.y as f64);
            let risk_to_coast = 100.0 - gameutil::clamp(0.0, 100.0, f64::abs(hgt));
            let mul = 1.0 + risk_to_coast * 0.1;
            (calc_dist(n1, n2) * mul) as u64
        };

        let heuristic = |n1: &Vector2<i64>| {
            calc_dist(n1, &pn2) as u64
        };

        let goal = |n1: &Vector2<i64>| {
            let dx = pn2.x - n1.x;
            let dy = pn2.y - n1.y;
            dx.abs() < STEP / 2 && dy.abs() < STEP / 2
        };

        let mut mp = astar(graph, distance, heuristic, goal, pn1, limit);
        match mp {
            None            => (),
            Some(ref mut p) => {
                p.remove(0); // remove start node
                p.push(Vector2::new(p2.x as i64, p2.z as i64));
            }
        }

        match mp {
            None        => println!("[{}] No path found", user),
            Some(ref p) => {
                println!("[{}] Path found with {} nodes", user, p.len());
            }
        }
        mp
    }

}

pub type Path = Vec<Vector2<i64>>;

pub fn vec_to_grid(v: Vector3<f64>) -> Vector2<i64> {
    fn round(n: f64) -> i64 {
        (prim::TILE_SIZE * (n / prim::TILE_SIZE).round()) as i64
    };

    Vector2::new(round(v.x), round(v.z))
}

fn astar<G, D, H, Goal, A>(graph: G, distance: D, heuristic: H, goal: Goal, start: A, limit: usize) -> Option<Vec<A>>
    where G: Fn(&A) -> Vec<A>,
          D: Fn(&A, &A) -> u64,
          Goal: Fn(&A) -> bool,
          H: Fn(&A) -> u64,
          A: Eq + Hash + Copy {
    struct State<B> {
        cost: u64,
        node: B
    }

    impl<B> Eq for State<B> {}
    impl<B> PartialEq for State<B> {
        fn eq(&self, other: &State<B>) -> bool {
            other.cost == self.cost
        }
    }

    impl<B> Ord for State<B> {
        fn cmp(&self, other: &State<B>) -> Ordering {
            other.cost.cmp(&self.cost)
        }
    }

    impl<B> PartialOrd for State<B> {
        fn partial_cmp(&self, other: &State<B>) -> Option<Ordering> {
            Some(self.cmp(&other))
        }
    }

    let mut visited = HashSet::new();
    let mut open = BinaryHeap::new();
    let mut cost_here = HashMap::new();
    let mut parents = HashMap::new();
    let mut ret: Vec<A> = Vec::new();
    let mut curr_backtrack_node = None;

    open.push(State{cost:0, node: start});
    while !open.is_empty() {
        let current = open.pop().unwrap().node;
        if visited.len() > limit {
            break;
        }

        if visited.contains(&current) {
            continue;
        }

        if goal(&current) {
            curr_backtrack_node = Some(current);
            break;
        }
        visited.insert(current);

        let children = graph(&current);
        for child in children.into_iter() {
            if visited.contains(&child) {
                continue;
            }

            let edge_cost = distance(&current, &child);
            let this_g_cost = cost_here.get(&current).unwrap_or(&0) + edge_cost;
            let mut add_this_as_parent = true;
            {
                let already_open_child = cost_here.get(&child);
                match already_open_child {
                    Some(&cost_prev) => if cost_prev <= this_g_cost { add_this_as_parent = false; },
                    None             => (),
                }
            }
            if add_this_as_parent {
                let this_f_cost = this_g_cost + heuristic(&child);
                parents.insert(child, current);
                open.push(State{cost: this_f_cost, node: child});
                cost_here.insert(child, this_g_cost);
            }
        }
    }

    if curr_backtrack_node.is_none() {
        return None;
    }

    loop {
        match curr_backtrack_node {
            None    => break,
            Some(n) => {
                ret.push(n);
                let parent = parents.get(&n);
                match parent {
                    None     => break,
                    Some(n2) => curr_backtrack_node = Some(*n2),
                }
            }
        }
    }
    ret.reverse();
    Some(ret)
}
