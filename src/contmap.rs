extern crate rand;

use na::Vector3;
use std::collections::HashMap;

use terrain;
use prim;

// 1. put flags on land masses which are reachable from the
//    largest water mass
// 2. if all flags are on the same land mass:
//      try to put the bases on the same land mass
//      no naval bases needed
//    else:
//      find coastline on largest water mass and land mass
//      that has at least one flag on it
//      put bases and naval bases near that coast

#[derive(PartialEq, Eq, Copy, Clone)]
#[derive(RustcDecodable, RustcEncodable)]
pub struct MassID {
    id: i32,
}

impl MassID {
    fn new(i: i32) -> MassID {
        MassID {
            id: i
        }
    }

    pub fn is_land(self: &MassID) -> bool {
        self.id > 0
    }
}

#[derive(RustcDecodable, RustcEncodable)]
pub struct ContinentMap {
    mass: Vec<Vec<MassID>>,
    neighbors: HashMap<i32, Vec<(usize, usize)>>,
    sizes: HashMap<i32, i32>,
    reachable_from: HashMap<i32, Vec<MassID>>,
    pub largest_water: MassID,
}

pub struct NavConfig {
    pub flags: Vec<prim::Flag>,
    pub base_positions: Option<[Vector3<f64>; 2]>,
    pub naval_positions: Option<[Vector3<f64>; 2]>,
}

impl ContinentMap {
    pub fn new(ground: &terrain::Ground) -> ContinentMap {
        let mut cm: ContinentMap = ContinentMap {
            mass: vec![vec![MassID::new(0); prim::GROUND_NUM_TILES as usize]; prim::GROUND_NUM_TILES as usize],
            neighbors: HashMap::new(),
            sizes: HashMap::new(),
            reachable_from: HashMap::new(),
            largest_water: MassID::new(0),
        };

        cm.set_up_mass(&ground);
        cm.find_neighbors();

        let mut largest_land = 0;
        let mut largest_land_size = 0;
        for (&id, &size) in &cm.sizes {
            if id > 0 && size > largest_land_size {
                largest_land_size = size;
                largest_land = id;
            }
        }
        println!("Largest land is {} with size {}", largest_land, largest_land_size);
        let mut largest_water = MassID::new(0);
        let mut largest_water_size = 0;
        for (&id, &size) in &cm.sizes {
            if id < 0 && size > largest_water_size {
                largest_water_size = size;
                largest_water = MassID::new(id);
            }
        }
        cm.largest_water = largest_water;
        println!("Largest water is {} with size {}", largest_water.id, largest_water_size);

        cm
    }

    pub fn get_reachables(&self, m: MassID) -> Option<&Vec<MassID>> {
        self.reachable_from.get(&m.id)
    }

    pub fn get_neighbors(&self, m: MassID) -> Option<&Vec<(usize, usize)>> {
        self.neighbors.get(&m.id)
    }

    fn get_mass_i(&self, i: usize, j: usize) -> MassID {
        let r = self.mass[i][j];
        assert!(r.id != 0);
        r
    }

    pub fn get_mass(&self, i: f64, j: f64) -> MassID {
        let x = terrain::discretize(i);
        let y = terrain::discretize(j);
        self.get_mass_i(x, y)
    }

    fn fill<F>(&mut self, a: usize, b: usize, id: MassID, check: F) -> ()
        where F: Fn(usize, usize) -> bool {
        let mut x = a;
        let mut y = b;
        let mut next = Vec::new();
        loop {
            self.mass[x][y] = id;
            let counter = self.sizes.entry(id.id).or_insert(0);
            *counter += 1;
            if x > 0                                   && check(x - 1, y) && self.mass[x - 1][y].id == 0 {
                next.push((x - 1, y));
            }
            if x < prim::GROUND_NUM_TILES as usize - 1 && check(x + 1, y) && self.mass[x + 1][y].id == 0 {
                next.push((x + 1, y));
            }
            if y > 0                                   && check(x, y - 1) && self.mass[x][y - 1].id == 0 {
                next.push((x, y - 1));
            }
            if y < prim::GROUND_NUM_TILES as usize - 1 && check(x, y + 1) && self.mass[x][y + 1].id == 0 {
                next.push((x, y + 1));
            }
            let mnext = next.pop();
            match mnext {
                None         => return,
                Some((i, j)) => { x = i; y = j; }
            }
        }
    }

    fn set_up_mass(&mut self, ground: &terrain::Ground) -> () {
        let mut next_land = 1;
        let mut next_sea  = -1;
        for j in 0..prim::GROUND_NUM_TILES as usize {
            for i in 0..prim::GROUND_NUM_TILES as usize {
                if self.mass[i][j].id == 0 {
                    let h = terrain::get_height_at_i(&ground, i as i32, j as i32);
                    if h > 0.0 {
                        self.fill(i, j, MassID::new(next_land), |x, y| terrain::get_height_at_i(&ground, x as i32, y as i32) > 0.0);
                        next_land += 1;
                    } else {
                        self.fill(i, j, MassID::new(next_sea), |x, y| terrain::get_height_at_i(&ground, x as i32, y as i32) <= 0.0);
                        next_sea -= 1;
                    }
                }
            }
        }
    }

    fn find_neighbors(&mut self) -> () {
        for j in 1..(prim::GROUND_NUM_TILES - 1) as usize {
            for i in 1..(prim::GROUND_NUM_TILES - 1) as usize {
                let this = self.mass[i][j];
                assert!(this.id != 0);
                let tests = [self.mass[i][j + 1], self.mass[i][j - 1], self.mass[i - 1][j], self.mass[i + 1][j]];
                for t in tests.iter() {
                    assert!(t.id != 0);
                    if this != *t {
                        {
                            let vn = self.neighbors.entry(t.id).or_insert(vec![]);
                            (*vn).push((i, j));
                        }
                        {
                            let vn = self.reachable_from.entry(t.id).or_insert(vec![]);
                            (*vn).push(this);
                        }
                    }
                }
            }
        }
    }
}


