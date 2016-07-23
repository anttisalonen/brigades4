extern crate nalgebra as na;
extern crate core;
extern crate rand;

use self::core::iter::FromIterator;

use na::{Vector3, Norm};

use game;
use game::{Soldier, Battlefield, FlagState};

use gameutil;

const SHOOT_DISTANCE: f32 = 100.0;
const REPLAN_TIME: f64       = game::TIME_MULTIPLIER as f64 * 120.0; // seconds
const FOOD_FETCH_BUFFER: f32 = game::TIME_MULTIPLIER as f32 * 480.0; // seconds

// data structures
pub struct SoldierAI {
    tasks: Vec<AiTask>,
    replan_timer: f64,
}

pub enum Action {
    NoAction(usize),
    MoveAction(usize, Vector3<f32>),
    ShootAction(usize, usize),
}

// core code
trait Task {
    fn update(&self, soldier: &Soldier, bf: &Battlefield) -> Option<Action>;
}

impl Task for AiTask {
    fn update(&self, soldier: &Soldier, bf: &Battlefield) -> Option<Action> {
        match self {
            &AiTask::Goto(ref g) => g.update(soldier, bf),
        }
    }
}

impl SoldierAI {
    pub fn new() -> SoldierAI {
        SoldierAI {
            tasks: vec![],
            replan_timer: rand::random::<f64>() * REPLAN_TIME,
        }
    }
}

pub fn soldier_ai_update(sai: &mut SoldierAI, soldier: &Soldier, bf: &Battlefield) -> Action {
    sai.replan_timer -= bf.frame_time;
    if sai.replan_timer <= 0.0 {
        sai.replan_timer += REPLAN_TIME * (rand::random::<f64>() * 2.0);
        sai.tasks = vec![];
    }
    if sai.tasks.len() == 0 {
        sai.tasks.push(ai_arbitrate_task(soldier, bf));
    }
    let num_tasks = sai.tasks.len();
    if num_tasks > 0 {
        let act = sai.tasks[num_tasks - 1].update(soldier, bf);
        match act {
            None    => sai.tasks.truncate(num_tasks - 1),
            Some(a) => return a,
        }
    }
    return Action::NoAction(soldier.id);
}

fn ai_arbitrate_task(s: &Soldier, bf: &Battlefield) -> AiTask {
    let supply = if s.side == game::Side::Blue {
        &bf.supply_points[0]
    } else {
        &bf.supply_points[1]
    };

    let dist = gameutil::dist(s, &supply.position);
    let time_to_travel = dist / game::SOLDIER_SPEED;
    let time_with_food = game::EAT_TIME * (s.food - 1) as f32;
    if time_with_food - FOOD_FETCH_BUFFER < time_to_travel || s.ammo < 5 {
        AiTask::Goto(AiGoto::new(supply.position))
    } else {
        AiTask::Goto(AiGoto::new(flag_target_position(s, bf)))
    }
}

// generic helpers
fn flag_target_position(sold: &Soldier, bf: &Battlefield) -> Vector3<f32> {
    let side = sold.side;
    let distances = bf.flags.iter().map(|f| gameutil::dist(sold, &f.position));
    let zm = distances.zip(bf.flags.iter());
    let mut zm = Vec::from_iter(zm);
    zm.sort_by(|&(d1, _), &(d2, _)| d1.partial_cmp(&d2).unwrap());
    for (_, flag) in zm {
        match flag.flag_state {
            FlagState::Free                       => return flag.position,
            FlagState::Transition(s) if s == side => (),
            FlagState::Transition(_)              => return flag.position,
            FlagState::Owned(s) if s == side      => if flag_lone_holder(sold, bf, &flag.position) { return flag.position; } else { () },
            FlagState::Owned(_)                   => return flag.position,
        }
    }
    bf.flags[0].position
}

fn flag_lone_holder(sold: &Soldier, bf: &Battlefield, pos: &Vector3<f32>) -> bool {
    let mut num_guards = 3i16;
    if gameutil::dist(sold, pos) > 5.0 {
        return false;
    }

    let side = sold.side;
    for sn in bf.soldiers.iter() {
        if sn.side == side && gameutil::dist(sn, pos) < 5.0 {
            num_guards -= 1;
            if num_guards == 0 {
                return false;
            }
        }
    }
    return true;
}


// task listing
enum AiTask {
    Goto(AiGoto),
}

// goto: if enemy is within shooting range, stop to shoot.
// else go to given point and stay there.
struct AiGoto {
    targetpos: Vector3<f32>,
}

// goto task - constructor
impl AiGoto {
    fn new(tgt: Vector3<f32>) -> AiGoto {
        AiGoto {
            targetpos: tgt
        }
    }
}

// goto task - update function
impl Task for AiGoto {
    fn update(&self, soldier: &Soldier, bf: &Battlefield) -> Option<Action> {
        let enemy_tgt = find_enemy(soldier, bf);
        match enemy_tgt {
            None    => Some(ai_goto(&self.targetpos, soldier)),
            Some(e) => Some(attack(e, soldier, &bf)),
        }
    }
}

fn find_enemy(soldier: &Soldier, bf: &Battlefield) -> Option<usize> {
    if soldier.ammo <= 0 {
        return None;
    }

    for i in 0..bf.soldiers.len() {
        if bf.soldiers[i].alive && bf.soldiers[i].side != soldier.side && gameutil::dist(soldier, &bf.soldiers[i].position) < SHOOT_DISTANCE {
            return Some(i)
        }
    }
    None
}

fn ai_goto(targetpos: &Vector3<f32>, soldier: &Soldier) -> Action {
    let tgt_vec = gameutil::to_vec_on_map(soldier, targetpos);
    let dist_to_tgt = tgt_vec.norm();
    if dist_to_tgt >= game::SUPPLY_DISTANCE * 0.5 {
        Action::MoveAction(soldier.id, tgt_vec.normalize() * game::SOLDIER_SPEED)
    } else {
        Action::NoAction(soldier.id)
    }
}

fn attack(e: usize, soldier: &Soldier, bf: &Battlefield) -> Action {
    assert!(soldier.ammo > 0);
    let dist = (bf.soldiers[e].position - soldier.position).norm();
    if dist > SHOOT_DISTANCE {
        ai_goto(&bf.soldiers[e].position, soldier)
    } else {
        if soldier.shot_timer <= 0.0 {
            Action::ShootAction(soldier.id, e)
        } else {
            Action::NoAction(soldier.id)
        }
    }
}


