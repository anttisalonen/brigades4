extern crate rand;

use bf_info::*;
use terrain;
use actions;

const DROWN_DEPTH: f64 = -2.0;

impl Battlefield {
    pub fn update_soldiers(&mut self) {
        for ref mut sold in &mut self.movers.soldiers {
            sold.position.y = terrain::get_height_at(&self.ground, sold.position.x, sold.position.z);
        }

        self.movers.check_drowning();
    }

    pub fn update_vehicles(&mut self) {
        for ref mut vehicle in &mut self.movers.vehicles {
            vehicle.position.y = terrain::get_height_at(&self.ground, vehicle.position.x, vehicle.position.z);
            vehicle.speed = vehicle.speed * (1.0 - 0.10 * self.frame_time);
        }

        self.movers.check_underwater_vehicles();
    }
}

impl Movers {
    fn check_drowning(&mut self) {
        for ref mut sold in &mut self.soldiers {
            if sold.position.y < DROWN_DEPTH {
                actions::kill_soldier(&mut self.boarded_map, sold);
            }
        }
    }

    fn check_underwater_vehicles(&mut self) {
        for ref mut vehicle in &mut self.vehicles {
            if vehicle.position.y < DROWN_DEPTH {
                actions::destroy_vehicle(&mut self.boarded_map, vehicle);
            }
        }
    }
}
