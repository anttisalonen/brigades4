extern crate nalgebra as na;
extern crate core;

use self::core::iter::FromIterator;

use na::{Vector3, Norm};

use game::{Soldier, Battlefield, FlagState};

const SHOOT_DISTANCE: f32 = 100.0;

pub struct SoldierAI {
    tasks: Vec<AiTask>
}

pub enum Action {
    NoAction(usize),
    MoveAction(usize, Vector3<f32>),
    ShootAction(usize, usize),
}

enum AiTask {
    Goto(AiGoto),
}

trait Task {
    fn update(&self, soldier: &Soldier, bf: &Battlefield) -> Option<Action>;
}

struct AiGoto {
    targetpos: Vector3<f32>,
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
        }
    }
}

impl AiGoto {
    fn new(soldier: &Soldier, bf: &Battlefield) -> AiGoto {
        AiGoto {
            targetpos: find_goto_target(soldier, bf)
        }
    }
}

fn find_goto_target(s: &Soldier, bf: &Battlefield) -> Vector3<f32> {
    let side = s.side;
    let distances = bf.flags.iter().map(|f| dist(s, &Vector3::new(f.position.x, s.position.y, f.position.y)));
    let zm = distances.zip(bf.flags.iter());
    let mut zm = Vec::from_iter(zm);
    zm.sort_by(|&(d1, _), &(d2, _)| d1.partial_cmp(&d2).unwrap());
    for (_, flag) in zm {
        match flag.flag_state {
            FlagState::Free                       => return Vector3::new(flag.position.x, 0.0, flag.position.y),
            FlagState::Transition(s) if s == side => (),
            FlagState::Transition(_)              => return Vector3::new(flag.position.x, 0.0, flag.position.y),
            FlagState::Owned(s) if s == side      => (),
            FlagState::Owned(_)                   => return Vector3::new(flag.position.x, 0.0, flag.position.y),
        }
    }
    Vector3::new(bf.flags[0].position.x, 0.0, bf.flags[0].position.y)
}

impl Task for AiGoto {
    fn update(&self, soldier: &Soldier, bf: &Battlefield) -> Option<Action> {
        let enemy_tgt = find_enemy(soldier, bf);
        match enemy_tgt {
            None    => Some(ai_goto(&self.targetpos, soldier, &bf)),
            Some(e) => Some(attack(e, soldier, &bf)),
        }
    }
}

fn ai_goto(targetpos: &Vector3<f32>, soldier: &Soldier, bf: &Battlefield) -> Action {
    let tgt_vec = to_vec(soldier, targetpos);
    let dist_to_tgt = tgt_vec.norm();
    if dist_to_tgt > 10.0 {
        let vel = 1.3f32;
        Action::MoveAction(soldier.id, tgt_vec.normalize() * (vel * bf.frame_time as f32))
    } else {
        Action::NoAction(soldier.id)
    }
}

fn to_vec(s1: &Soldier, tgt: &Vector3<f32>) -> Vector3<f32> {
    let diff_vec = *tgt - s1.position;
    Vector3::new(diff_vec.x, 0.0, diff_vec.z)
}

fn dist(s1: &Soldier, tgt: &Vector3<f32>) -> f32 {
    to_vec(s1, tgt).norm()
}

fn find_enemy(soldier: &Soldier, bf: &Battlefield) -> Option<usize> {
    for i in 0..bf.soldiers.len() {
        if bf.soldiers[i].alive && bf.soldiers[i].side != soldier.side && dist(soldier, &bf.soldiers[i].position) < SHOOT_DISTANCE {
            return Some(i)
        }
    }
    None
}

fn attack(e: usize, soldier: &Soldier, bf: &Battlefield) -> Action {
    let dist = (bf.soldiers[e].position - soldier.position).norm();
    if dist > SHOOT_DISTANCE {
        ai_goto(&bf.soldiers[e].position, soldier, &bf)
    } else {
        if soldier.shot_timer <= 0.0 {
            Action::ShootAction(soldier.id, e)
        } else {
            Action::NoAction(soldier.id)
        }
    }
}

pub fn soldier_ai_update(sai: &mut SoldierAI, soldier: &Soldier, bf: &Battlefield) -> Action {
    let num_tasks = sai.tasks.len();
    if num_tasks > 0 {
        let act = sai.tasks[num_tasks - 1].update(soldier, bf);
        match act {
            None    => sai.tasks.truncate(num_tasks - 1),
            Some(a) => return a,
        }
    }
    if num_tasks == 0 {
        sai.tasks.push(AiTask::Goto(AiGoto::new(soldier, bf)));
    }
    return Action::NoAction(soldier.id);
}


