extern crate nalgebra as na;

use na::{Vector3, Norm};

use game::{Soldier, Battlefield};

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

impl AiGoto {
    fn new(bf: &Battlefield) -> AiGoto {
        AiGoto {
            targetpos: Vector3::new(bf.flags[0].flag_position.x, 0.0, bf.flags[0].flag_position.y)
        }
    }
}

fn ai_goto(targetpos: &Vector3<f32>, soldier: &Soldier, bf: &Battlefield) -> Action {
    let diff_vec = *targetpos - soldier.position;
    let tgt_vec = Vector3::new(diff_vec.x, 0.0, diff_vec.z);
    let dist_to_tgt = tgt_vec.norm();
    if dist_to_tgt > 10.0 {
        let vel = 1.3f32;
        Action::MoveAction(soldier.id, tgt_vec.normalize() * (vel * bf.frame_time as f32))
    } else {
        Action::NoAction(soldier.id)
    }
}

fn find_enemy(soldier: &Soldier, bf: &Battlefield) -> Option<usize> {
    for i in 0..bf.soldiers.len() {
        if bf.soldiers[i].alive && bf.soldiers[i].side != soldier.side {
            return Some(i)
        }
    }
    None
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

fn attack(e: usize, soldier: &Soldier, bf: &Battlefield) -> Action {
    let dist = (bf.soldiers[e].position - soldier.position).norm();
    if dist > 100.0 {
        ai_goto(&bf.soldiers[e].position, soldier, &bf)
    } else {
        if soldier.shot_timer <= 0.0 {
            Action::ShootAction(soldier.id, e)
        } else {
            Action::NoAction(soldier.id)
        }
    }
}

impl Task for AiTask {
    fn update(&self, soldier: &Soldier, bf: &Battlefield) -> Option<Action> {
        match self {
            &AiTask::Goto(ref g) => g.update(soldier, bf),
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
        sai.tasks.push(AiTask::Goto(AiGoto::new(bf)));
    }
    return Action::NoAction(soldier.id);
}

impl SoldierAI {
    pub fn new() -> SoldierAI {
        SoldierAI {
            tasks: vec![],
        }
    }
}

