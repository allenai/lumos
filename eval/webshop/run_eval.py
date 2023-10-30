import argparse
import os
import re
import json
import random
import torch
import gym
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from rich import print
from rich.markup import escape
from eval.webshop.shop_tool import *
from eval.webshop.utils import *
from eval.webshop.examplars import PLAN_EXAMPLARS, GROUND_EXAMPLARS
from eval.webshop.web_agent_site.envs import WebAgentTextEnv
from eval.webshop.web_agent_site.models import RandomPolicy
from eval.webshop.web_agent_site.utils import DEBUG_PROD_SIZE
from eval.utils import generate_completions, load_hf_lm_and_tokenizer


def execute(env, action, inter_results, idx=None):
    action_variable, action_name, action_args = parse_actions(action)
    variables = findvariables(action_args)

    if action_name in ["", "Search", "Click", "FeatureRetrieve", "Pick"]:
        if action_name == "Search":
            previous_actions[idx].append(f"search[{action_args}]")

            return action_variable[0], ShopActions.Search(env, action_args)
        elif action_name == "Click":
            for action in previous_actions[idx]:    # finish all the previous actions
                observation, reward, done, _ = env.step(action)
            if not variables:
                curr_action = f"click[{action_args}]"
            else:
                curr_action = f"click[{inter_results[action_args]}]"
            previous_actions[idx].append(curr_action)

            return action_variable[0], ShopActions.Click(env, curr_action)
        elif action_name == "FeatureRetrieve":
            if "QUERY" in action_args:
                query_sep = "QUERY"
            else:
                query_sep = "Query"
            query = action_args[action_args.find(f"{query_sep}: ")+len(f"{query_sep}: "): -1]

            return action_variable[0], ShopActions.FeatureRetrieve(query, inter_results[variables[0]][6:])
        elif action_name == "Pick":
            if "QUERY" in action_args:
                query_sep = "QUERY"
            else:
                query_sep = "Query"
            item_names = action_args.split(", Item_features: ")[0].split("Item_names: ")[-1].strip()
            item_features = action_args.split(", Item_features: ")[-1].split(f", {query_sep}: ")[0][1:-1].split(", ")
            query = action_args.split(f", {query_sep}: ")[-1].strip()
            items = inter_results[item_names].split(" ** ")

            items_w_desp = list()
            for i, variable_idx in enumerate(sorted(list([int(k[1:]) for k in item_features]))):
                if f"R{str(variable_idx)}" in inter_results and isinstance(inter_results[f"R{str(variable_idx)}"], str):
                    items_w_desp.append(items[i] + ". Available features: " + inter_results[f"R{str(variable_idx)}"])
            if not items_w_desp:
                return action_variable[0], items[0].split(' - ')[0].strip()

            return action_variable[0], ShopActions.Pick(query, items_w_desp)


def lumos_iterative(args):
    print("Loading data...")
    test_data, subgoals, actions, plan_prompts, ground_prompts = list(), list(), list(), list(), list()
    all_subgoals_actions, solved_idx, subgoal_results = list(), list(), list()

    env = args.env
    for instruction_idx in range(args.max_num_examples):
        env.reset(instruction_idx=instruction_idx)
        observation = env.observation
        instruction = observation.split(" [SEP] ")[-2].strip()
        test_data.append(instruction)
    
    prompt_plan_prefix, prompt_ground_prefix = "", ""
    for plan_example, ground_example in zip(PLAN_EXAMPLARS, GROUND_EXAMPLARS):
        tmp_subgoals = list()
        plan_task_prefix = "Please provide a reasonable subgoal-based plan to solve the given task.\n"
        prompt_plan_prefix += "<|user|>\n" + plan_task_prefix + "Task: " + plan_example["instruction"].strip() + "; Initial Environment Description: None.\n<|assistant|>\n"
        for i, subgoal in enumerate(plan_example["subgoals"].split("; Subgoal ")):
            if i == 0:
                results = subgoal.split(" || Results: ")[1]
                subgoal = subgoal.split(" || Results: ")[0]
                prompt_plan_prefix += subgoal.strip() + f"\n\n<|user|>\nThe execution result for Subgoal {str(i+1)} is {results.strip('.')}.\n<|assistant|>\n"
            else:
                results = f"Subgoal {subgoal}".split(" || Results: ")[1]
                subgoal = f"Subgoal {subgoal}".split(" || Results: ")[0]
                prompt_plan_prefix += subgoal.strip() + f"\n\n<|user|>\nThe execution result for Subgoal {str(i+1)} is {results.strip('.')}.\n<|assistant|>\n"
            tmp_subgoals.append(subgoal)
        prompt_plan_prefix += "\n\n"

        ground_task_prefix = "Please ground the given subgoal to corresponding executable actions for solving the given task. The grounded actions must be the one in available action list.\n\n" \
                            "The available action list is 'Search', 'FeatureRetrieve', 'Pick' and 'Click'.\nSearch(Query): Search related items according to the Query; FeatureRetrieve(Feature_list, Query): Select the most relevant features from Feature_list according to Query; Pick(Item_names, Item_features, Query): Pick the most relevant item from Item_names according to Query, Item_names and Item_features; Click(Item): Click the Item to check more information.\n\n"
        prompt_ground_prefix += "<|user|>\n" + ground_task_prefix + "Task: " + ground_example["instruction"].strip()
        for i, action in enumerate(ground_example["actions"].split("; Action ")):
            action = f"Action {action}"
            if i == 0:
                prompt_ground_prefix += f" \nSubgoal to be grounded: {tmp_subgoals[i]}\n<|assistant|>\n"
            else:
                prompt_ground_prefix += f"\n\n<|user|>\nSubgoal to be grounded: {tmp_subgoals[i]}\n<|assistant|>\n"
            for act in action.split(" || "):
                act = act[act.find(":")+1: ].strip()
                prompt_ground_prefix += act + "; "
            prompt_ground_prefix = prompt_ground_prefix[:-2]
        prompt_ground_prefix += "\n\n"

    tot_reward = 0
    for i in tqdm(range(10)):
        if i == 0:
            prompt_plan_prefix += "<|user|>\nPlease provide a reasonable subgoal-based plan to solve the given task.\n"
        elif i == 1:
            prompt_ground_prefix += "<|user|>\nPlease ground the given subgoal to corresponding executable actions for solving the given task. The grounded actions must be the one in available action list.\n\n" \
                                    "The available action list is 'Search', 'FeatureRetrieve', 'Pick' and 'Click'.\nSearch(Query): Search related items according to the Query; FeatureRetrieve(Feature_list, Query): Select the most relevant features from Feature_list according to Query; Pick(Item_names, Item_features, Query): Pick the most relevant item from Item_names according to Query, Item_names and Item_features; Click(Item): Click the Item to check more information.\n\n"
        
        assert not subgoals or len(subgoals) == len(test_data)
        assert not actions or len(actions) == len(test_data)
        
        if i % 2 == 0:
            for j, example in enumerate(test_data):
                if j in solved_idx:
                    continue
                if i == 0:
                    all_subgoals_actions.append({"question": test_data[j].strip(), "subgoals": [], "actions": [], "results": dict()})
                    plan_prompt = prompt_plan_prefix + "Task: " + test_data[j].strip() + "; Initial Environment Description: None.\n<|assistant|>\n"
                    plan_prompts.append(plan_prompt)
                else:
                    try:
                        previous_subgoals = sorted(list([int(k.split("Subgoal ")[-1]) for k in subgoal_results[j].keys()]))
                        last_subgoal = 'Subgoal ' + str(previous_subgoals[-1])
                        plan_prompts[j] += subgoals[j].strip() + f"\n\n<|user|>\nThe executed result for Subgoal {str(i // 2)} is {subgoal_results[j][last_subgoal]}\n<|assistant|>\n"
                    except:
                        solved_idx.append(j)
        else:
            for j, example in enumerate(test_data):
                if j in solved_idx:
                    continue
                if i == 1:
                    ground_prompt = prompt_ground_prefix + "Task: " + test_data[j].strip() + f" \nSubgoal to be grounded: {subgoals[j].split('No, I will keep planning.')[-1].strip()}\n<|assistant|>\n" + "R1 = "
                    ground_prompts.append(ground_prompt)
                else:
                    ground_prompts[j] += actions[j].strip() + f"\n\n<|user|>\nSubgoal to be grounded: {subgoals[j].split('No, I will keep planning.')[-1].strip()}\n<|assistant|>\n"

        if args.model_name_or_path:
            print("Loading model and tokenizer...")
            if i % 2 == 0:
                model_name_or_path = os.path.join(args.model_name_or_path, "lumos_unified_plan_iterative")
            else:
                model_name_or_path = os.path.join(args.model_name_or_path, "lumos_unified_ground_iterative")
            
            model, tokenizer = load_hf_lm_and_tokenizer(
                model_name_or_path=model_name_or_path, 
                tokenizer_name_or_path=model_name_or_path, 
                load_in_8bit=args.load_in_8bit, 
                load_in_half=True,
                gptq_model=args.gptq
            )
            new_line_token = tokenizer.encode("\n", add_special_tokens=False)[-1] # get the last token because the tokenizer may add space tokens at the start.

            if i % 2 == 0:
                subgoals = generate_completions(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=plan_prompts,
                    max_new_tokens=512,
                    batch_size=args.eval_batch_size,
                    stop_id_sequences=[[new_line_token]]
                )
            else:
                actions = generate_completions(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=ground_prompts,
                    max_new_tokens=512,
                    batch_size=args.eval_batch_size,
                    stop_id_sequences=[[new_line_token]]
                )
        
        if i % 2 == 1:
            for j in range(len(test_data)):
                if j in solved_idx:
                    continue
                env.reset(instruction_idx=j)
                all_subgoals_actions[j]["subgoals"].append(subgoals[j].strip())
                cur_subgoal_idx = subgoals[j].strip().split(':')[0].split("No, I will keep planning.")[-1].strip()
                if i == 1:
                    previous_actions.append(list())
                    actions = ["R1 = " + action.strip("R1 = ") for action in actions]
                    subgoal_results.append({cur_subgoal_idx: None})
                
                try:
                    for action in actions[j].strip().split('; '):
                        results_variable, execution_results = execute(env, action, all_subgoals_actions[j]["results"], j)
                        if execution_results == "":
                            execution_results = "buy now"
                        all_subgoals_actions[j]["results"][results_variable] = execution_results
                    subgoal_results[j][cur_subgoal_idx] = execution_results
                    
                    if isinstance(execution_results, float):
                        solved_idx.append(j)
                        tot_reward += execution_results
                except:
                    solved_idx.append(j)
    print(tot_reward, len(solved_idx), "Average Reward:", tot_reward / len(solved_idx))
    env.close()
    

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
    
    previous_actions = list()
    args.env = gym.make('WebAgentTextEnv-v0', observation_mode='text', num_products=DEBUG_PROD_SIZE)

    lumos_iterative(args)
