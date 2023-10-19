import argparse
import os
import re
import json
import random
import evaluate
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from eval.mind2web.model import CrossEncoder
from eval.mind2web.dataloader import get_data_split
from eval.utils import generate_completions, load_hf_lm_and_tokenizer


def process_tag(x):
    x = x.split("target: ")[-1].strip()
    cleaned_tag = ""
    for w in x.split(' ')[2:]:
        if '(' not in w and ')' not in w:
            cleaned_tag += w + " "
    return cleaned_tag.strip()


def extract_key_info(env, query, previous_actions, rank, action_name=None, op=None, text=None):
    global tot_tags, corr_em
    query_reformulate = (
        f'task is: {query}\n'
        f'Previous actions: {previous_actions}'
    )

    tag_dict = dict()
    tag_mapping, unique_tags = list(), list()
    for tag in env:
        tag = tag[1]
        if tag not in tag_dict:
            tag_dict[tag] = len(tag_dict)
            unique_tags.append(tag)
        tag_mapping.append(tag_dict[tag])

    model_input = [[query_reformulate, tag] for tag in unique_tags]
    pred_scores = query_model.predict(
        model_input,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=64,
    )
    pred_scores = np.array(
        [pred_scores[tag_idx] for tag_idx in tag_mapping]
    )
    
    pred_scores_argsort = np.argsort(
        -pred_scores
    )  # Sort in decreasing order
    if rank == 1:
        for i in pred_scores_argsort[:rank]:
            if env[i][1].replace('\n', ' ') == env[0][1].replace('\n', ' '):
                if action_name == op["op"]:
                    if action_name == "CLICK" or ((action_name == "TYPE" or action_name == "SELECT") and text.lower() == op["value"].lower()):
                        corr_em += 1
            tot_tags += 1

        model_input, orig_tags = list(), list()
        for i in pred_scores_argsort[:10]:
            model_input.append([query, process_tag(env[i][1].replace('\n', ' '))])
            orig_tags.append(env[i][1].replace('\n', ' '))
    if rank == 50:
        return env[:rank]
    else:
        return env[i][1]


def execute(action_name, env, query, previous_actions, op, text=None):
    if action_name == "CLICK":
        relevant_tag = extract_key_info(env, query, previous_actions, 1, action_name, op)
        return {"tag": relevant_tag, "text": query}
    elif action_name == "TYPE":
        relevant_tag = extract_key_info(env, query, previous_actions, 1, action_name, op, text)
        return {"tag": relevant_tag, "text": text}
    else:
        relevant_tag = extract_key_info(env, query + f" The option contents should be close to '{text}'.", previous_actions, 1, action_name, op, text)
        return {"tag": relevant_tag, "text": text}

def translate_action(action):
    action = action.split(" -> ")
    tag_type = action[0][action[0].find('[')+1: action[0].find(']')]
    tag_contents = action[0][action[0].find(']')+1: ].strip()

    if "CLICK" in action[1] or "HOVER" in action[1]:
        phrases_for_click = ["Visit", "Click", "Find and go into", "Enter"]
        return f"{phrases_for_click[random.randint(0, 3)]} {tag_type} {tag_contents}"
    elif "TYPE" in action[1]:
        phrases_for_type = ["Type", "Input", "Put text"]
        type_text = action[1].split("TYPE:")[-1].strip()
        return f"{phrases_for_type[random.randint(0, 2)]} {type_text} in the {tag_type} {tag_contents}"
    elif "SELECT" in action[1]:
        phrases_for_select = ["Select", "Choose", "Pick up"]
        type_text = action[1].split("SELECT:")[-1].strip()
        return f"{phrases_for_select[random.randint(0, 2)]} {type_text} in the {tag_type} {tag_contents}"
    else:
        action_text = action[1].split(":")[0].strip().lower()
        return f"{action_text} {tag_type} {tag_contents}."


def lumos_iterative(args):
    print("Loading data...")
    test_data, subgoals, actions, plan_prompts, ground_prompts = list(), list(), list(), list(), list()
    all_subgoals_actions, solved_idx, previous_syn_subgoals, previous_syn_actions = list(), list(), list(), list()

    for fn in os.listdir(os.path.join(args.data_dir, "test_domain")):
        with open(os.path.join(args.data_dir, "test_domain", fn)) as fin:
            test_data += json.load(fin)

    if args.max_num_examples and len(test_data) > args.max_num_examples:
        test_data = test_data[:args.max_num_examples]
    test_data = get_data_split(test_data)
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    tot_tags = 0
    for i in tqdm(range(15)):
        if len(solved_idx) == len(test_data):
            break
        prompt_plan_prefix = "Please provide a reasonable subgoal-based plan to solve the given task.\n"
        prompt_ground_prefix = "Please ground the given subgoal to corresponding executable actions for solving the given task. The grounded actions must be the one in available action list.\n\n" \
                                "The available action list is 'CLICK', 'TYPE', 'SELECT'.\nCLICK(Env, Query): Click the relevant html region in Env according to Query; TYPE(Env, Query, Text): Type Text into the relevant html region in Env according to Query; SELECT(Env, Query, Text): Select the value Text of the relevant selection box in Env according to Query.\n\n"
        
        assert not subgoals or len(subgoals) == len(test_data)
        assert not actions or len(actions) == len(test_data)
        
        for j, example in enumerate(test_data):
            if j in solved_idx:
                continue
            if i == 0:
                all_subgoals_actions.append({"question": example["confirmed_task"].strip(), "answer": example["actions"][i]["pos_candidates"], "subgoals": [], "actions": [], "results": dict()})
                plan_prompt = "<|user|>\n" + prompt_plan_prefix + "Task: " + example["confirmed_task"].strip() + "; Initial Environment Description: None.\n<|assistant|>\n"
                plan_prompts.append(plan_prompt)
            else:
                syn_subgoal = translate_action(example["action_reprs"][i-1])
                plan_prompts[j] += f"Subgoal {str(i)}: " + syn_subgoal
                syn_subgoal = syn_subgoal[0].lower() + syn_subgoal[1:]
                plan_prompts[j] += f"\n\n<|user|>\nWe have already {syn_subgoal}. Should we stop planning?\n<|assistant|>\n"

        if args.model_name_or_path:
            print("Loading model and tokenizer...")
            model_name_or_path = os.path.join(args.model_name_or_path, "web_agent_plan_091300_llama-2-7b")
            
            model, tokenizer = load_hf_lm_and_tokenizer(
                model_name_or_path=model_name_or_path, 
                tokenizer_name_or_path=model_name_or_path, 
                load_in_8bit=args.load_in_8bit, 
                load_in_half=True,
                gptq_model=args.gptq
            )
            new_line_token = tokenizer.encode("\n", add_special_tokens=False)[-1] # get the last token because the tokenizer may add space tokens at the start.
            
            subgoals = generate_completions(
                model=model,
                tokenizer=tokenizer,
                prompts=plan_prompts,
                max_new_tokens=512,
                batch_size=args.eval_batch_size,
                stop_id_sequences=[[new_line_token]]
            )

        for j, example in enumerate(test_data):
            if i == 0:
                ground_prompt = "<|user|>\n" + prompt_ground_prefix + "Task: " + example["confirmed_task"].strip() + f" \nSubgoal to be grounded: {subgoals[j].strip()}\n<|assistant|>\n"
                ground_prompts.append(ground_prompt)
            else:
                ground_prompts[j] += actions[j].strip() + f"\n\n<|user|>\nSubgoal to be grounded: {subgoals[j].split('No, I will keep planning.')[-1].strip()}\n<|assistant|>\n"

        if args.model_name_or_path:
            print("Loading model and tokenizer...")
            model_name_or_path = os.path.join(args.model_name_or_path, "web_agent_ground_091300_llama-2-7b")
            
            model, tokenizer = load_hf_lm_and_tokenizer(
                model_name_or_path=model_name_or_path, 
                tokenizer_name_or_path=model_name_or_path, 
                load_in_8bit=args.load_in_8bit, 
                load_in_half=True,
                gptq_model=args.gptq
            )
            new_line_token = tokenizer.encode("\n", add_special_tokens=False)[-1] # get the last token because the tokenizer may add space tokens at the start.

            actions = generate_completions(
                model=model,
                tokenizer=tokenizer,
                prompts=ground_prompts,
                max_new_tokens=512,
                batch_size=args.eval_batch_size,
                stop_id_sequences=[[new_line_token]]
            )

        for j in range(len(test_data)):
            if j in solved_idx:
                continue
            for action in actions[j].strip().split('; '):
                pos_action = action.find(" = ") + len(" = ")
                pos_parenthesis = action.find('(')
                pos_right_parenthesis = action.rfind(')')

                action_variable = action[: pos_action - len(" = ")].split(", ")[0].strip()
                action_name = action[pos_action: pos_parenthesis]
                action_args = action[pos_parenthesis+1: pos_right_parenthesis]
                arg_variable = action_args[:action_args.find(", ")]

                left_bound = max(i-3, 0)
                previous_actions = "; ".join(test_data[j]["action_reprs"][left_bound: i])
                if i == 0:
                    env = extract_key_info(test_data[j]["actions"][i]["pos_candidates"] + test_data[j]["actions"][i]["neg_candidates"], test_data[j]["confirmed_task"].strip(), previous_actions, 50)
                    all_subgoals_actions[j]["results"]["Env"] = env
                else:
                    env = all_subgoals_actions[j]["results"][arg_variable]

                if action_name == "CLICK":
                    query = action_args[action_args.find(", QUERY:") + len(", QUERY:"): ].strip()
                    execution_results = execute(action_name, env, query, previous_actions, test_data[j]["actions"][i]["operation"])
                elif action_name == "SELECT" or action_name == "TYPE":
                    query = action_args[action_args.find(", QUERY:") + len(", QUERY:"): action_args.rfind(", TEXT:")].strip()
                    text = action_args[action_args.find(", TEXT:") + len(", TEXT:"): ].strip()
                    execution_results = execute(action_name, env, query, previous_actions, test_data[j]["actions"][i]["operation"], text)
                
                if i < len(test_data[j]["actions"]) - 1:
                    all_subgoals_actions[j]["results"][action_variable] = extract_key_info(test_data[j]["actions"][i+1]["pos_candidates"] + test_data[j]["actions"][i+1]["neg_candidates"], test_data[j]["confirmed_task"].strip(), previous_actions + f"; {test_data[j]['action_reprs'][i]}", 50)
                else:
                    if j not in solved_idx:
                        solved_idx.append(j)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--max_num_examples", type=int, default=None, help="maximum number of examples to evaluate.")
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--model_name_or_path", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="batch size for evaluation.")
    parser.add_argument("--formulation", type=str, default='lumos_iterative', help="considered formulation.")
    parser.add_argument("--load_in_8bit", action="store_true", help="load model in 8bit mode, which will reduce memory and speed up inference.")
    parser.add_argument("--gptq", action="store_true", help="If given, we're evaluating a 4-bit quantized GPTQ model.")
    args = parser.parse_args()

    global query_model 
    query_model = CrossEncoder(
        "osunlp/MindAct_CandidateGeneration_deberta-v3-base",
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        num_labels=1,
        max_length=512,
    )

    tot_tags, corr_em = 0, 0
    lumos_iterative(args)

    print("Step success rate:", corr_em*1.0/tot_tags)

