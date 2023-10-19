import re
import sys
import math

def findvariables(x):
    findv = re.compile('R[0-9]+')
    variables = findv.findall(x)

    return variables

def parse_subgoals(x):
    pos_subgoal_1 = x.find("Subgoal 1")
    subgoal_plan = x[pos_subgoal_1:]
    subgoal_w_action_pairs = subgoal_plan.split("\n\n")
    subgoals = [subgoal_w_action_pair.split('\n')[0] for subgoal_w_action_pair in subgoal_w_action_pairs]
    actions = [subgoal_w_action_pair.split('\n')[1:] for subgoal_w_action_pair in subgoal_w_action_pairs]

    return subgoals, actions

def parse_actions(x):
    pos_action = x.find(" = ") + len(" = ")
    pos_parenthesis = x.find('(')
    action_variable = x[: pos_action - len(" = ")].split(", ")
    action_name = x[pos_action: pos_parenthesis]

    tot = 0
    pos_right_parenthesis = pos_parenthesis - 1
    for ch in x[pos_parenthesis:]:
        pos_right_parenthesis += 1
        if ch == '(':
            tot += 1
        elif ch == ')':
            tot -= 1
        if tot == 0:
            break
    action_args = x[pos_parenthesis + 1: pos_right_parenthesis]
    
    return action_variable, action_name, action_args

def parse_args(x):
    pos_args_st = x.find("def Code(") + len("def Code(")
    pos_args_ed = x.find('):')
    
    return x[pos_args_st: pos_args_ed].split(', ')