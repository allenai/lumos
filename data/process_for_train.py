import json
import re
import argparse
from data.prompt_convertion import *


def parse_subgoals(x):
    pos_subgoal_1 = x.find("Subgoal 1")
    subgoal_plan = x[pos_subgoal_1:]
    subgoal_w_action_pairs = subgoal_plan.split("\n\n")
    subgoals = [subgoal_w_action_pair.split('\n')[0] for subgoal_w_action_pair in subgoal_w_action_pairs]
    actions = [subgoal_w_action_pair.split('\n')[1:] for subgoal_w_action_pair in subgoal_w_action_pairs]

    return subgoals, actions

def parse_actions(x, domain="maths"):
    pos_action = x.find(" = ") + len(" = ")
    pos_parenthesis = x.find('(')
    action_variable = x[x.find(':') + 2: pos_action - len(" = ")].split(", ")
    action_name = x[pos_action: pos_parenthesis]

    if domain != "web_agent":
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
        if pos_right_parenthesis+2 >= len(x):
            action_args = x[pos_parenthesis+1: x.rfind(")")]
            action_results = action_args
        else:
            if pos_right_parenthesis == pos_parenthesis:    # Output = R9 = 40
                action_name = ''
                action_args = x[pos_action: x.rfind(" = ")].strip()
                action_results = x[x.rfind(" = ")+3: ].strip()
            else:
                if x[pos_right_parenthesis+2] != '=':       # parenthesis doesn't match
                    action_args = x[pos_parenthesis+1: x.rfind(" = ")-1]
                    action_results = x[x.rfind(" = ")+3: ].strip()
                else:
                    action_args = x[pos_parenthesis+1: pos_right_parenthesis]
                    action_results = x[pos_right_parenthesis+3: ].strip()
    else:
        action_args = x[x.find('(')+1: x.rfind(')')]
        action_results = ""
    
    return action_variable, action_name, action_args, action_results


def aggregate_multi_datasets(domain):
    data = list()
    for fn in os.listdir(os.path.join("data/train", domain, "converted")):
        with open(os.path.join("data/train", domain, "converted", fn), 'r') as f:
            data += [json.loads(d) for d in f]
    return data


def collect_cot_data(args):
    if args.domain == "maths":
        data = list()
        for fn in os.listdir("data/train/maths/raw_data"):
            with open(os.path.join("data/train/maths/raw_data", fn), 'r') as f:
                fn_data = [json.loads(d) for d in f]
                if "prm" in fn:
                    fn_data = fn_data[:10000]
                for d in fn_data:
                    if "gsm" in fn:
                        d["solu"] = d["answer"].split("#### ")[0].replace("\n", " ").strip() + f" Answer is {d['answer'].split('####')[1].strip()}."
                    elif "prm" in fn:
                        if d["question"]["ground_truth_solution"] and d["question"]["ground_truth_answer"]:
                            d["solu"] = d["question"]["ground_truth_solution"] + f" Answer is {d['question']['ground_truth_answer']}."
                        else:
                            continue
                        d["question"] = d["question"]["problem"]
                    elif "asdiv" in fn:
                        d["solu"] = d["solution"] + f". Answer is {d['answer'].split(' ')[0]}."
                    data.append(d)
    elif args.domain == "complex_qa":
        data = aggregate_multi_datasets("complex_qa")
    
    if not os.path.exists(f"data/train/{args.domain}/train_annots"):
        os.mkdir(f"data/train/{args.domain}/train_annots")

    with open(f"data/train/{args.domain}/train_annots/{args.domain}_cot.jsonl", 'w') as f:
        if args.domain == "maths":
            for i, d in enumerate(data):
                messages = [{
                    "role": "user",
                    "content": d["question"]
                }, 
                {
                    "role": "assistant",
                    "content": d["solu"]
                }]

                f.write(json.dumps({
                    "dataset": "maths",
                    "id": f"maths_{i}",
                    "messages": messages
                }) + "\n")
        elif args.domain == "complex_qa":
            for i, d in enumerate(data):
                pos_answer_part = d["natural_language_plan"].find("Based on these evidences and decomposed questions")
                answer_part = d["natural_language_plan"][pos_answer_part:]
                messages = [{
                    "role": "user",
                    "content": d["task"]
                }, 
                {
                    "role": "assistant",
                    "content": d["natural_language_plan"].split("We need to answer these questions:")[0].strip() + ' ' + answer_part
                }]

                f.write(json.dumps({
                    "dataset": "complex_qa",
                    "id": f"complex_qa_{i}",
                    "messages": messages
                }) + "\n")


def collect_direct_data(args):
    if args.domain == "maths":
        data = list()
        for fn in os.listdir("data/train/maths/raw_data"):
            with open(os.path.join("data/train/maths/raw_data", fn), 'r') as f:
                fn_data = [json.loads(d) for d in f]
                if "prm" in fn:
                    fn_data = fn_data[:10000]
                for d in fn_data:
                    if "gsm" in fn:
                        d["answer"] = d["answer"].split('####')[1].strip()
                    elif "prm" in fn:
                        d["answer"] = d["question"]["ground_truth_answer"]
                        d["question"] = d["question"]["problem"]
                    elif "asdiv" in fn:
                        d["answer"] = d["answer"].split(' ')[0]
                    data.append(d)
    if args.domain == "complex_qa":
        with open("data/train/complex_qa/raw_data/musique_full_v1.0_train.jsonl", 'r') as f:
            musique_data = [json.loads(d) for d in f]
        with open("data/train/complex_qa/raw_data/strategyqa_train.json", 'r') as f:
            strategyqa_data = json.load(f)
        data = musique_data + strategyqa_data

    if not os.path.exists(f"data/train/{args.domain}/train_annots"):
        os.mkdir(f"data/train/{args.domain}/train_annots")

    with open(f"data/train/{args.domain}/train_annots/{args.domain}_direct.jsonl", 'w') as f:
        for i, d in enumerate(data):
            messages = [{
                "role": "user",
                "content": d["question"]
            }, 
            {
                "role": "assistant",
                "content": str(d["answer"])
            }]

            f.write(json.dumps({
                "dataset": args.domain,
                "id": f"{args.domain}_{i}",
                "messages": messages
            }) + "\n")


def collect_plan_data(args):
    data = aggregate_multi_datasets(args.domain)

    if not os.path.exists(f"data/train/{args.domain}/train_annots"):
        os.mkdir(f"data/train/{args.domain}/train_annots")

    if "iterative" in args.formulation:
        output_fn = f"data/train/{args.domain}/train_annots/lumos_{args.domain}_plan_iterative.jsonl"
    else:
        output_fn = f"data/train/{args.domain}/train_annots/lumos_{args.domain}_plan_onetime.jsonl"

    all_messages = list()
    for i, d in enumerate(data):
        subgoal_plan = d["subgoal_plan"]
        subgoals, actions = parse_subgoals(subgoal_plan)
        user = f"Please provide a reasonable subgoal-based plan to solve the given task.\nTask: {d['task']}; Initial Environment Description: None."

        unavailable = False
        messages = []
        messages.append({
            "role": "user",
            "content": user
        })

        if "iterative" not in args.formulation:
            messages.append({
                "role": "assistant",
                "content": "; ".join(subgoals)
            })
        else:
            for j, (subgoal, action) in enumerate(zip(subgoals, actions)):
                if action == []:
                    unavailable = True
                    break
                if args.domain != "web_agent":
                    user = f"The executed result for Subgoal {str(j+1)} is "
                    last_act = action[-1]
                    user += last_act[last_act.rfind('=')+2:].strip().strip('.')
                else:
                    last_subgoal = subgoal[subgoal.find(": ")+2:]
                    last_subgoal = last_subgoal[0].lower() + last_subgoal[1:]
                    user = f"We have already {last_subgoal}".strip('.')       # we won't put the html results here (too long..)
                user = user.strip() + ". Should we stop planning?"
                
                try:
                    current_variable = action[-1].split(" = ")[0].split(':')[1].strip().split(", ")[-1]
                    if current_variable[0] == "R":
                        current_variable_idx = int(current_variable[1:])
                    else:
                        user = user.replace("Output", "R" + str(current_variable_idx + 1))

                    if j != 0:
                        messages.append({
                            "role": "assistant",
                            "content": "No, I will keep planning. " + subgoal
                        })
                    else:
                        messages.append({
                            "role": "assistant",
                            "content": subgoal
                        })
                    
                    messages.append({
                        "role": "user",
                        "content": user
                    })
                except:
                    unavailable = True
                    break
            
            messages.append({
                "role": "assistant",
                "content": "Yes, I will stop planning."
            })

        if unavailable:
            continue
        else:
            all_messages.append(messages)
    
    with open(output_fn, 'w') as f:
        for i, messages in enumerate(all_messages):
            f.write(json.dumps({
                "dataset": f"{args.domain}",
                "id": f"{args.domain}_{i}",
                "messages": messages
            }) + "\n")


def collect_ground_data(args):
    data = aggregate_multi_datasets(args.domain)

    if not os.path.exists(f"data/train/{args.domain}/train_annots"):
        os.mkdir(f"data/train/{args.domain}/train_annots")

    if "iterative" in args.formulation:
        output_fn = f"data/train/{args.domain}/train_annots/lumos_{args.domain}_ground_iterative.jsonl"
    else:
        output_fn = f"data/train/{args.domain}/train_annots/lumos_{args.domain}_ground_onetime.jsonl"

    all_messages = list()
    for i, d in enumerate(data):
        subgoal_plan = d["subgoal_plan"]
        subgoals, actions = parse_subgoals(subgoal_plan)
        
        if args.domain == "maths":
            user_inst = "Please ground the given subgoal to corresponding executable actions for solving the given task. The grounded actions must be the one in available action list.\n\n" \
                        "The available action list is 'Calculator', 'SetEquation', 'SolveEquation', 'Count', 'SolveInequality', 'Code', and 'Define'.\n" \
                        "Calculator(formula): Calculate the input formula; SetEquation(equation): Set up an equation to be solved; SolveEquation(equation): Solve the previous set equation; Count(list): Count the number of elements in the given list; SolveInequality(inequality): Solve the previous set inequality; Code(pseudo_code): Generate a Python function that corresponds to the pseudo code; Define(variable/number): Define a variable or a number for latter usage.\n\n" \
                        f"Task: {d['task']} \n"
        elif args.domain == "complex_qa":
            user_inst = "Please ground the given subgoal to corresponding executable actions for solving the given task. The grounded actions must be the one in available action list.\n\n" \
                        "The available action list is 'KnowledgeQuery', 'ParagraphRetrieve', 'QA', 'Calculator', and 'Code'.\n" \
                        "Calculator(formula): Calculate the input formula; KnowledgeQuery(query): Capture the relevant webpages according to the query; ParagraphRetrieve(context, query): Given a query, retrieve the most relevant paragraphs from the given context; QA(context, query): Given context, answer the given query; Calculator(formula): Calculate the input formula; Code(pseudo_code): Generate a Python function that corresponds to the pseudo code.\n\n" \
                        f"Task: {d['task']} \n"
        elif args.domain == "web_agent":
            user_inst = "Please ground the given subgoal to corresponding executable actions for solving the given task. The grounded actions must be the one in available action list.\n\n" \
                        "The available action list is 'CLICK', 'TYPE', 'SELECT'.\n" \
                        "CLICK(Env, Query): Click the relevant html region in Env according to Query; TYPE(Env, Query, Text): Type Text into the relevant html region in Env according to Query; SELECT(Env, Query, Text): Select the value Text of the relevant selection box in Env according to Query.\n\n" \
                        f"Task: {d['task']} \n"

        unavailable = False
        messages = []

        if "iterative" not in args.formulation:
            user = user_inst + f"Subgoals to be grounded: {'; '.join(subgoals)}\n"
                
            assistant = ""
            for action in actions:
                for act in action:
                    action_variable, action_name, action_args, action_results = parse_actions(act, args.domain)
                    assistant += f"{', '.join(action_variable)} = {action_name}({action_args})" + '; '
            assistant = assistant[:-2]

            messages.append({
                "role": "user",
                "content": user
            })
            messages.append({
                "role": "assistant",
                "content": assistant
            })
        else:
            subgoal_result = dict()
            for j, (subgoal, action) in enumerate(zip(subgoals, actions)):
                if action == []:
                    unavailable = True
                    break
                if j == 0:
                    user = user_inst + f"Subgoal to be grounded: {subgoal}\n"
                else:
                    if args.domain == "complex_qa":
                        user = f"Subgoal to be grounded: {subgoal} The grounding could be based on the following results:"
                        for subgoal_idx in sorted(list(subgoal_result.keys())):
                            user += f" The execution result for {subgoal_idx} is {subgoal_result[subgoal_idx]}"
                        user += "\n"
                    else:
                        user = f"Subgoal to be grounded: {subgoal}\n"
                
                assistant = ""
                cur_subgoal_idx = subgoal[: subgoal.find(':')]
                for act in action:
                    action_variable, action_name, action_args, action_results = parse_actions(act, args.domain)
                    assistant += f"{', '.join(action_variable)} = {action_name}({action_args})" + '; '
                subgoal_result[cur_subgoal_idx] = action_results
                assistant = assistant[:-2]

                try:
                    current_variable = action_variable[-1]
                    if current_variable[0] == "R":
                        current_variable_idx = int(current_variable[1:])
                    else:
                        assistant = assistant.replace("Output", "R" + str(current_variable_idx + 1))
                except:
                    unavailable = True
                    break

                messages.append({
                    "role": "user",
                    "content": user
                })
                messages.append({
                    "role": "assistant",
                    "content": assistant
                })

        if unavailable:
            continue
        else:
            all_messages.append(messages)
    
    with open(output_fn, 'w') as f:
        for i, messages in enumerate(all_messages):
            f.write(json.dumps({
                "dataset": f"{args.domain}",
                "id": f"{args.domain}_{i}",
                "messages": messages
            }) + "\n")


def collect_unified_data(domains):
    unified_data = {"plan": list(), "ground": list()}
    for domain in domains:
        for module in ["plan", "ground"]:
            with open(f"data/train/{domain}/train_annots/lumos_{domain}_{module}_iterative.jsonl", "r") as f:
                unified_data[module] += [json.loads(d) for d in f]

    for module in ["plan", "ground"]:
        with open(f"data/train/unified/train_annots/lumos_unified_{module}_iterative.jsonl", "w") as f:
            for d in unified_data[module]:
                f.write(json.dumps(d)+'\n')


def main(args):
    if args.unified:
        domains = args.unified.split(",")
        collect_unified_data(domains)
        return
    
    if args.formulation == "cot":
        collect_cot_data(args)
    elif args.formulation == "direct":
        collect_direct_data(args)
    elif "lumos" in args.formulation:
        collect_plan_data(args)
        collect_ground_data(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--formulation',
        dest='formulation',
        type=str,
        default="lumos_iterative"
    )

    parser.add_argument(
        '--domain',
        dest='domain',
        type=str,
        default="maths"
    )

    parser.add_argument(
        '--unified',
        dest='unified',
        type=str,
    )

    args = parser.parse_args()
    
    main(args)
