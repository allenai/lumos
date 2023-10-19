import os
import re
import sys
import math
import wolframalpha
from eval.svamp.math_tool import *

MATH_OP = ['*', '/', '+', '-', '^']

wolfram_app_id = os.environ["WOLFRAM_API_KEY"]
client = wolframalpha.Client(wolfram_app_id)

def WA_result(question):
    tmp = next(client.query(question).results)
    if "plaintext" not in tmp.subpod:
        return [i["plaintext"] for i in tmp.subpod]
    else:
        return tmp.subpod["plaintext"]

def isalphain(x):
    isletter = re.compile('[A-Za-z]+')
    match = isletter.findall(x)

    return True if match else False

def parse_func_args(x):
    pos_right_parenthesis = x.rfind(')')

    tot = 0
    for i in range(pos_right_parenthesis):
        ch = x[pos_right_parenthesis-i]
        if ch == ')':
            tot += 1
        elif ch == '(':
            tot -= 1
        if tot == 0:
            pos_parenthesis = pos_right_parenthesis-i
            break
    func_args = x[pos_parenthesis+1: pos_right_parenthesis]
    func_exp = x[:pos_parenthesis]

    return func_args, func_exp

def parse_equation(x):
    iseq = re.compile(':\((.*?)\):')
    matches = iseq.findall(x)
    common_op = x
    for match in matches:
        common_op = common_op.replace(match, "")
    common_op = common_op.replace(' ', '').split(":():")

    return [match.split('=')[0] for match in matches], [match.split('=')[1] for match in matches], [op for op in common_op if op]

def combine_eqs(x):
    eqs_left, eqs_right, common_op = parse_equation(x)
    
    if len(eqs_left) == 1:
        return MathActions.Calculator(common_op[0] + f"({eqs_left[0]})", False) + " = " + MathActions.Calculator(common_op[0] + f"({eqs_right[0]})", False)
    else:
        final_left = ""
        for i in range(len(eqs_left)-1):
            final_left += f"({eqs_left[i]})" + common_op[i]
        final_left += f"({eqs_left[-1]})"

        final_right = ""
        for i in range(len(eqs_right)-1):
            final_right += f"({eqs_right[i]})" + common_op[i]
        final_right += f"({eqs_right[-1]})"

        return MathActions.Calculator(final_left, False) + " = " + MathActions.Calculator(final_right, False)

def examine_ops(action_args):
    for op in MATH_OP:
        if op in action_args:
            return True
    return False

def findvariables(x):
    findv = re.compile('R[0-9]+')
    variables = findv.findall(x)

    return variables

def process_frac(x):
    if isinstance(x, int) or isinstance(x, float):
        return x
    if '/' in x and x.replace('/', '').replace(' ', '').isdigit():
        numerator = float(x.split('/')[0].strip())
        denominator = float(x.split('/')[1].strip())
        return numerator / denominator

    return x

def isfunc(x):
    findfunc = re.compile("R[0-9]+\(")
    func = findfunc.findall(x)

    return True if func else False

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
    if pos_right_parenthesis + 2 >= len(x):
        action_args = x[pos_parenthesis + 1: x.rfind(")")]
    else:
        if pos_right_parenthesis == pos_parenthesis:    # Output = R9 = 40
            action_name = ''
            action_args = x[pos_action: x.rfind(" = ")].strip()
        else:
            if x[pos_right_parenthesis + 2] != '=':       # parenthesis doesn't match
                action_args = x[pos_parenthesis + 1: x.rfind(" = ") - 1]
            else:
                action_args = x[pos_parenthesis + 1: pos_right_parenthesis]
    
    return action_variable, action_name, action_args

def parse_args(x):
    pos_args_st = x.find("def Code(") + len("def Code(")
    pos_args_ed = x.find('):')
    
    return x[pos_args_st: pos_args_ed].split(', ')