extern crate rand;

use bf_info::*;
use terrain;
use actions;

const DROWN_DEPTH: f64 = -2.0;

impl Battlefield {
    pub fn update_soldiers(&mut self) {
        for ref mut sold in &mut self.movers.soldiers {
            match soldier_boarded(&self.movers.boarded_map, sold.id) {
                None           => {
                    sold.position.y = terrain::get_height_at(&self.ground, sold.position.x, sold.position.z);
                },
                Some((vid, _)) => {
                    let ref veh = self.movers.vehicles[vid.id];
                    sold.position.y = veh.position.y;
                }
            }
        }

        self.movers.check_drowning();
    }

    pub fn update_vehicles(&mut self) {
        {
            let types: Vec<VehicleType> = self.movers.vehicles.iter().map(|v| self.movers.vehicle_type(v.id)).collect();
            for (ref mut vehicle, typ) in &mut self.movers.vehicles.iter_mut().zip(types) {
                match typ {
                    VehicleType::Land => {
                        vehicle.position.y = terrain::get_height_at(&self.ground, vehicle.position.x, vehicle.position.z);
                    },
                    VehicleType::Sea  => {
                        vehicle.position.y = f64::max(0.0, terrain::get_height_at(&self.ground, vehicle.position.x, vehicle.position.z));
                    }
                }
                vehicle.speed = vehicle.speed * (1.0 - 0.10 * self.frame_time);
            }
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
        let mut to_destroy = vec![];
        for (i, ref vehicle) in self.vehicles.iter().enumerate() {
            match self.vehicle_type(vehicle.id) {
                VehicleType::Land => {
                    if vehicle.position.y < DROWN_DEPTH {
                        to_destroy.push(i);
                    }
                },
                VehicleType::Sea  => {
                    if vehicle.position.y > -DROWN_DEPTH {
                        to_destroy.push(i);
                    }
                },
            }
        }
        for i in to_destroy.iter() {
            actions::destroy_vehicle(&mut self.boarded_map, &mut self.vehicles[*i]);
        }
    }
}
