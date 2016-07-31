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

const STEP: i64 = prim::TILE_SIZE as i64;

pub fn find_flag_positions(ground: &terrain::Ground, rand: &mut StdRng) -> Vec<prim::Flag> {
    let mut flag_positions = Vec::new();
    loop {
        let xp = (rand.gen::<f64>() * 0.8 + 0.1) * prim::DIM - prim::HDIM;
        let zp = (rand.gen::<f64>() * 0.8 + 0.1) * prim::DIM - prim::HDIM;
        let yp = terrain::get_height_at(&ground, xp, zp);
        if yp > 10.0 {
            flag_positions.push(Vector3::new(xp, yp, zp));
        }
        if flag_positions.len() == 10 {
            break;
        }
    }

    flag_positions.into_iter().map(|p| prim::Flag {
        position: p,
        flag_state: prim::FlagState::Free,
        flag_timer: prim::FLAG_TIMER,
    }).collect()
}

pub fn find_base_positions(ground: &terrain::Ground, rand: &mut StdRng) -> [Vector3<f64>; 2] {
    loop {
        let x1 = -prim::HDIM + prim::TILE_SIZE * 2.0;
        let x2 =  prim::HDIM - prim::TILE_SIZE * 2.0;
        let z1 = (rand.gen::<f64>() * 0.8 + 0.1) * prim::DIM - prim::HDIM;
        let z2 = (rand.gen::<f64>() * 0.8 + 0.1) * prim::DIM - prim::HDIM;
        let y1 = terrain::get_height_at(&ground, x1, z1);
        let y2 = terrain::get_height_at(&ground, x2, z2);
        if y1 < 10.0 || y2 < 10.0 {
            continue;
        }

        let bp = [
                    Vector3::new(x1, y1, z1),
                    Vector3::new(x2, y2, z2),
        ];
        let path = find_path(&ground, bp[0], bp[1]);

        match path {
            None    => (),
            Some(_) => return bp,
        }
    }
}

pub fn vec_to_grid(v: Vector3<f64>) -> Vector2<i64> {
    fn round(n: f64) -> i64 {
        (prim::TILE_SIZE * (n / prim::TILE_SIZE).round()) as i64
    };

    Vector2::new(round(v.x), round(v.z))
}

pub fn find_path(ground: &terrain::Ground, p1: Vector3<f64>, p2: Vector3<f64>) -> Option<Vec<Vector2<i64>>> {
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
                if terrain::get_height_at(ground, pos.x as f64, pos.y as f64) < 1.0 {
                    continue;
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

    let distance = |n1: &Vector2<i64>, n2: &Vector2<i64>| {
        let dx = n2.x - n1.x;
        let dy = n2.y - n1.y;
        ((dx * dx + dy * dy) as f64).sqrt() as u64
    };

    let heuristic = |n1: &Vector2<i64>| {
        let dx = pn2.x - n1.x;
        let dy = pn2.y - n1.y;
        (dx.abs() + dy.abs()) as u64
    };

    let goal = |n1: &Vector2<i64>| {
        let dx = pn2.x - n1.x;
        let dy = pn2.y - n1.y;
        dx.abs() < STEP / 2 && dy.abs() < STEP / 2
    };

    let mut mp = astar(graph, distance, heuristic, goal, pn1);
    match mp {
        None            => (),
        Some(ref mut p) => {
            p.push(Vector2::new(p2.x as i64, p2.z as i64));
        }
    }

    match mp {
        None        => println!("No path found"),
        Some(ref p) => {
            println!("Path found with {} nodes", p.len());
        }
    }
    mp
}

fn astar<G, D, H, Goal, A>(graph: G, distance: D, heuristic: H, goal: Goal, start: A) -> Option<Vec<A>>
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
