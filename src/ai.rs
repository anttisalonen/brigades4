extern crate nalgebra as na;
extern crate core;
extern crate rand;

use std::collections::VecDeque;

use bf_info::*;

use ai_tree;
use ai_task::*;
use ai_prim::*;

pub fn soldier_ai_update(aicfg: &ai_tree::AiConfig, sideai: &mut SideAI, sai: &mut SoldierAI, soldier: &Soldier, bf: &Battlefield) -> Action {
    sai.replan_timer -= bf.frame_time;
    if sai.replan_timer <= 0.0 {
        sai.replan_timer += REPLAN_TIME * 0.5 + REPLAN_TIME * (rand::random::<f64>());
        sai.tasks = VecDeque::new();
    }
    if sai.tasks.len() == 0 {
        let mut tasks = ai_tree::ai_arbitrate_task(aicfg, sideai, sai, soldier, bf);
        assert!(!tasks.is_empty());
        sai.tasks.append(&mut tasks);
    }
    let num_tasks = sai.tasks.len();
    if num_tasks > 0 {
        let act = sai.tasks.front_mut().unwrap().update(soldier, bf);
        match act {
            None    => { sai.tasks.pop_front(); (); }
            Some(a) => return a,
        }
    }
    return Action::NoAction(soldier.id);
}



