extern crate nalgebra as na;
extern crate core;
extern crate rand;

use std::collections::VecDeque;
use std::collections::HashMap;

use bf_info::*;
use ai_task::*;
use gameutil;
use ai_prim::*;
use ai_act;
use ai_dec;

macro_rules! map(
    { $($key:expr => $value:expr),+ } => {
        {
            let mut m = ::std::collections::HashMap::new();
            $(
                m.insert($key, $value);
            )+
            m
        }
     };
);

type DecActionMap = HashMap<String, DecAction>;
fn get_action_function_name_map() -> DecActionMap {
    const ACTFUNVEC: [DecAction; 8] = [
        ai_act::drive_to_food,
        ai_act::walk_to_supply,
        ai_act::drive_to_flag,
        ai_act::drive_to_base,
        ai_act::get_driven_to_destination,
        ai_act::walk_to_nearest_flag,
        ai_act::board_nearby_vehicle,
        ai_act::rest,
    ];
    map!{
        "drive_to_food".to_string()             => ACTFUNVEC[0],
        "walk_to_supply".to_string()            => ACTFUNVEC[1],
        "drive_to_flag".to_string()             => ACTFUNVEC[2],
        "drive_to_base".to_string()             => ACTFUNVEC[3],
        "get_driven_to_destination".to_string() => ACTFUNVEC[4],
        "walk_to_nearest_flag".to_string()      => ACTFUNVEC[5],
        "board_nearby_vehicle".to_string()      => ACTFUNVEC[6],
        "rest".to_string()                      => ACTFUNVEC[7]
    }
}

type ConditionMap = HashMap<String, Condition>;
fn get_condition_function_name_map() -> ConditionMap {
    const CONDFUNVEC: [Condition; 7] = [
        ai_dec::need_food,
        ai_dec::am_driving,
        ai_dec::have_enough_passengers,
        ai_dec::am_boarded,
        ai_dec::flag_within_walking_distance,
        ai_dec::vehicle_within_walking_distance,
        ai_dec::flag_within_days_march,
    ];
    map!{
        "need_food".to_string()                       => CONDFUNVEC[0],
        "am_driving".to_string()                      => CONDFUNVEC[1],
        "have_enough_passengers".to_string()          => CONDFUNVEC[2],
        "am_boarded".to_string()                      => CONDFUNVEC[3],
        "flag_within_walking_distance".to_string()    => CONDFUNVEC[4],
        "vehicle_within_walking_distance".to_string() => CONDFUNVEC[5],
        "flag_within_days_march".to_string()          => CONDFUNVEC[6]
    }
}

#[derive(RustcDecodable, RustcEncodable)]
enum DecisionResultSer {
    DecisionSer(DecisionNodeSer),
    ActionSer(String),
}

#[derive(RustcDecodable, RustcEncodable)]
struct DecisionNodeSer {
    func: String,
    yes: Box<DecisionResultSer>,
    no: Box<DecisionResultSer>,
}

fn setup_decision_tree(root_ser: DecisionNodeSer) -> DecisionNode {
    let dec_name_map = get_condition_function_name_map();
    let act_name_map = get_action_function_name_map();
    let my_func = *dec_name_map.get(&root_ser.func).expect(&format!("Could not find decision \"{}\"", root_ser.func));
    let my_yes = match *root_ser.yes {
        DecisionResultSer::DecisionSer(node) => Box::new(DecisionResult::Decision(setup_decision_tree(node))),
        DecisionResultSer::ActionSer(act)    => Box::new(DecisionResult::Action(*act_name_map.get(&act).expect(&format!("Could not find action \"{}\"", act)))),
    };
    let my_no = match *root_ser.no {
        DecisionResultSer::DecisionSer(node) => Box::new(DecisionResult::Decision(setup_decision_tree(node))),
        DecisionResultSer::ActionSer(act)    => Box::new(DecisionResult::Action(*act_name_map.get(&act).expect(&format!("Could not find action \"{}\"", act)))),
    };
    DecisionNode {
        func: my_func,
        yes: my_yes,
        no: my_no,
    }
}

enum DecisionResult {
    Decision(DecisionNode),
    Action(DecAction),
}

struct DecisionNode {
    func: Condition,
    yes: Box<DecisionResult>,
    no: Box<DecisionResult>,
}

type DecAction = fn(sideai: &mut SideAI, sai: &mut SoldierAI, &Soldier, &Battlefield) -> VecDeque<AiTask>;
type Condition = fn(sideai: &mut SideAI, sai: &mut SoldierAI, &Soldier, &Battlefield) -> bool;

fn eval_node<'a>(node: &'a DecisionNode, sideai: &mut SideAI, sai: &mut SoldierAI, s: &Soldier, bf: &Battlefield) -> &'a Box<DecisionResult> {
    if (node.func)(sideai, sai, s, bf) {
        &node.yes
    } else {
        &node.no
    }
}

fn eval_decision_tree(root_decision: &DecisionNode, sideai: &mut SideAI, sai: &mut SoldierAI, s: &Soldier, bf: &Battlefield) -> VecDeque<AiTask> {
    let mut node = root_decision;
    loop {
        let dec = eval_node(node, sideai, sai, s, bf);
        match **dec {
            DecisionResult::Action(ref act)    => return (act)(sideai, sai, s, bf),
            DecisionResult::Decision(ref next) => { node = &next; ()},
        }
    }
}

pub fn ai_arbitrate_task(aicfg: &AiConfig, sideai: &mut SideAI, sai: &mut SoldierAI, s: &Soldier, bf: &Battlefield) -> VecDeque<AiTask> {
    let root_decision = &aicfg.root_decision;
    let st = get_status(bf, s);
    sai.status = st;
    match st {
        Status::Driving(tid) => sai.my_vehicle = Some(tid),
        Status::Boarded(tid) => sai.my_vehicle = Some(tid),
        _                    => sai.my_vehicle = None,
    }
    sai.nearby_flag = flag_nearby(s, bf, 30000.0);
    sai.nearby_vehicle = free_vehicle_nearby(s, bf, 5000.0);

    eval_decision_tree(root_decision, sideai, sai, s, bf)
}

pub struct AiConfig {
    root_decision: DecisionNode,
}

impl AiConfig {
    pub fn new(filename: &str) -> AiConfig {
        let root_ser = gameutil::read_json(filename);
        let root = setup_decision_tree(root_ser);
        AiConfig {
            root_decision: root,
        }
    }
}


