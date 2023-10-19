import openai
import json
import os
import argparse
from tqdm import tqdm
from zeno_build.models import lm_config
from zeno_build.prompts import chat_prompt
from data.incontext import *
from data.fast_prompt import *


def maths_convertion(args):
    system_info = "You are a helpful assist to convert natural language plans or structural action list" \
                    "into subgoal-based plans and their corresponding structured actions."
    model_config = lm_config.LMConfig(provider="openai_chat", model="gpt-4")
    converted_num = 0

    with open(args.data_fn, "r") as f:
        solus = [json.loads(line) for line in f]

    if not os.path.exists("data/train/maths/converted"):
        os.mkdir("data/train/maths/converted")

    if not args.convert_all:
        solus = solus[args.st_idx: args.ed_idx]
        output_fn = f"data/train/maths/converted/{args.data_fn[args.data_fn.rfind('/')+1: args.data_fn.find('.')]}_converted_{args.st_idx}_{args.ed_idx}.json"
        data_num = args.ed_idx - args.st_idx
    else:
        if "prm" in args.data_fn:
            output_fn = "data/train/maths/converted/prm800k_converted_10000.json"
            data_num = 10000
        else:
            output_fn = f"data/train/maths/converted/{args.data_fn[args.data_fn.rfind('/')+1: args.data_fn.find('.')]}_converted.json"
            data_num = len(solus)

    while converted_num < data_num:
        full_contexts = list()
        for solu in solus:
            if "prm" in args.data_fn:
                final_prompt = instruction_maths_prm + f"Task: {solu['question']['problem']}\n\n"
                final_prompt += f"Natural language plan: {solu['question']['ground_truth_solution']}\n\n"
            elif "gsm" in args.data_fn:
                final_prompt = instruction_maths_gsm + f"Task: {solu['question']}\n\n"
                human_solu = solu['answer'].split("####")[0].split(" ** ")
                final_prompt += f"Natural language plan: {' '.join(human_solu[1:])}\n\n"
            elif "asdiv" in args.data_fn:
                final_prompt = instruction_maths_gsm + f"Task: {solu['question']}\n\n"
                final_prompt += f"Natural language plan: {solu['solution']}\n\n"
            full_contexts.append(
                chat_prompt.ChatMessages([{"role": "system", "content": system_info}, {"role": "user", "content": final_prompt}])
            )

        predictions = asyncio.run(
            generate_from_openai_chat_completion(
                full_contexts=full_contexts,
                model_config=model_config,
                temperature=0,
                max_tokens=500
            )
        )

        new_solus = list()
        with open(output_fn, "a+") as f:
            for i, (solu, pred) in enumerate(zip(solus, predictions)):
                inst = dict()
                if "prm" in args.data_fn:
                    inst["task"] = solu['question']['problem']
                    inst["natural_language_plan"] = solu['question']['ground_truth_solution']
                elif "gsm" in args.data_fn:
                    inst["task"] = solu['question']
                    human_solu = solu['answer'].split("####")[0].split(" ** ")
                    inst["ans"] = solu['answer'].split("####")[1].strip()
                    inst["natural_language_plan"] = ' '.join(human_solu[1:])
                elif "asdiv" in args.data_fn:
                    inst["task"] = solu['question']
                    inst["natural_language_plan"] = solu['solution']
                inst["subgoal_plan"] = pred

                if not inst["subgoal_plan"]:
                    new_solus.append(solu)
                else:
                    converted_num += 1
                    f.write(json.dumps(inst)+'\n')
                    if converted_num >= data_num:
                        break
        solus = new_solus


def convert_qa_to_human_solution(x, data_fn):
    if "strategyqa" in data_fn:
        prompt = "We find relevant facts: "
        prompt += ' '.join(x['facts'])
        prompt += " We need to answer these questions: "

        for evidence in x["evidence"]:
            useful = True
            all_ref = []
            for ref in evidence:
                if "no_evidence" in ref:
                    useful = False
                    break
                else:
                    islist = False
                    for r in ref:
                        if isinstance(r, list):
                            islist = True
                            all_ref.append(r)
                    if not islist:
                        all_ref.append([])
            if useful:
                break

        if not useful:
            return None
        
        for i, q in enumerate(x['decomposition']):
            ref = all_ref[i]
            ref = [f"'{r}'" for r in ref]
            prompt += str(i+1) + ". " + q + ' '
            if ref:
                prompt += f"(Can be answered based on paragraph {', '.join(ref)}) "
        prompt += f"Based on these evidences and decomposed questions, the answer is {str(x['answer'])}."
    elif "musique" in data_fn:
        prompt = "We find relevant facts: "
        paras = dict()
        for para in x["paragraphs"]:
            paras[para["idx"]] = para
        for q in x["question_decomposition"]:
            prompt += ' ' + paras[q["paragraph_support_idx"]]["paragraph_text"]
        
        prompt += " We need to answer these questions:"
        for i, q in enumerate(x["question_decomposition"]):
            prompt += ' ' + str(i+1) + ". " + q["question"] + f" (Can be answered based on paragraph '{paras[q['paragraph_support_idx']]['title']}')"
        prompt += f" Based on these evidences and decomposed questions, the answer is {x['answer']}."

    return prompt


def complex_qa_convertion(args):
    system_info = "You are a helpful assist to convert natural language plans or structural action list" \
                    "into subgoal-based plans and their corresponding structured actions."
    model_config = lm_config.LMConfig(provider="openai_chat", model="gpt-4")
    converted_num = 0

    if "strategyqa" in args.data_fn:
        with open(args.data_fn, "r") as f:
            solus = json.load(f)
    elif "musique" in args.data_fn:
        with open(args.data_fn, "r") as f:
            solus = [json.loads(line) for line in f]

    if not os.path.exists("data/train/complex_qa/converted"):
        os.mkdir("data/train/complex_qa/converted")

    if not args.convert_all:
        solus = solus[args.st_idx: args.ed_idx]
        if "strategyqa" in args.data_fn:
            output_fn = f"data/train/complex_qa/converted/strategyqa_converted_{args.st_idx}_{args.ed_idx}.json"
        else:
            output_fn = f"data/train/complex_qa/converted/musique_converted_{args.st_idx}_{args.ed_idx}.json"
        data_num = args.ed_idx - args.st_idx
    else:
        if "strategyqa" in args.data_fn:
            output_fn = f"data/train/complex_qa/converted/strategyqa_converted.json"
        else:
            output_fn = f"data/train/complex_qa/converted/musique_converted.json"
        data_num = len(solus)

    available_solus = list()
    if "strategyqa" in args.data_fn:
        for solu in solus:
            converted_test_case = convert_qa_to_human_solution(solu, args.data_fn)
            if converted_test_case:
                available_solus.append(solu)
    elif "musique" in args.data_fn:
        for solu in solus:
            is_support = True
            for q in solu['question_decomposition']:
                if not q['paragraph_support_idx']:
                    is_support = False
                    break
            if is_support:
                available_solus.append(solu)
    
    solus = available_solus.copy()
    available_solus_num = len(solus)
        
    while converted_num < data_num and converted_num < available_solus_num:
        full_contexts = list()
        for solu in solus:
            final_prompt = instruction_qa + f"Task: {solu['question']}\n\n"
            if convert_qa_to_human_solution(solu, args.data_fn):
                final_prompt += f"Natural language plan: {convert_qa_to_human_solution(solu, args.data_fn)}\n\n"
            full_contexts.append(
                chat_prompt.ChatMessages([{"role": "system", "content": system_info}, {"role": "user", "content": final_prompt}])
            )

        predictions = asyncio.run(
            generate_from_openai_chat_completion(
                full_contexts=full_contexts,
                model_config=model_config,
                temperature=0,
                max_tokens=500
            )
        )

        new_solus = list()
        with open(output_fn, "a+") as f:
            for i, (solu, pred) in enumerate(zip(solus, predictions)):
                inst = dict()
                inst["task"] = solu['question']
                inst["natural_language_plan"] = convert_qa_to_human_solution(solu, args.data_fn)
                inst["subgoal_plan"] = pred

                if not inst["subgoal_plan"]:
                    new_solus.append(solu)
                else:
                    if inst["natural_language_plan"]:
                        converted_num += 1
                        f.write(json.dumps(inst)+'\n')
                    if converted_num >= data_num or converted_num >= available_solus_num:
                        break
        solus = new_solus


def web_agent_convertion(args):
    system_info = "You are a helpful assist to convert natural language plans or structural action list" \
                    "into subgoal-based plans and their corresponding structured actions."
    model_config = lm_config.LMConfig(provider="openai_chat", model="gpt-4")
    converted_num = 0

    data_fns = os.listdir("data/train/web_agent/raw_data")
    solus = list()
    for fn in data_fns:
        if "train_" in fn:
            with open(os.path.join("data/train/web_agent/raw_data", fn), "r") as f:
                solus += json.load(f)

    if not os.path.exists("data/train/web_agent/converted"):
        os.mkdir("data/train/web_agent/converted")

    if not args.convert_all:
        solus = solus[args.st_idx: args.ed_idx]
        output_fn = f"data/train/web_agent/converted/mind2web_converted_{args.st_idx}_{args.ed_idx}.json"
        data_num = args.ed_idx - args.st_idx
    else:
        output_fn = "data/train/web_agent/converted/mind2web_converted.json"
        data_num = len(solus)
    
    while converted_num < data_num:
        full_contexts = list()
        for solu in solus:
            final_prompt = instruction_web_agent + f"Task: {solu['confirmed_task']}\n\n"
            final_prompt += f"Natural language plan: {'; '.join(solu['action_reprs'])}\n\n"
            full_contexts.append(
                chat_prompt.ChatMessages([{"role": "system", "content": system_info}, {"role": "user", "content": final_prompt}])
            )

        predictions = asyncio.run(
            generate_from_openai_chat_completion(
                full_contexts=full_contexts,
                model_config=model_config,
                temperature=0,
                max_tokens=800
            )
        )

        new_solus = list()
        with open(output_fn, "a+") as f:
            for solu, pred in zip(solus, predictions):
                inst = dict()
                inst["task"] = solu['confirmed_task']
                inst["natural_language_plan"] = solu['action_reprs']
                inst["subgoal_plan"] = pred

                if not inst["subgoal_plan"]:
                    new_solus.append(solu)
                else:
                    if inst["natural_language_plan"]:
                        converted_num += 1
                        f.write(json.dumps(inst)+'\n')
                    if converted_num >= data_num:
                        break
        solus = new_solus


def main(args):
    if args.domain == "maths":
        maths_convertion(args)
    elif args.domain == "complex_qa":
        complex_qa_convertion(args)
    elif args.domain == "web_agent":
        web_agent_convertion(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--domain',
        dest='domain',
        type=str,
    )

    parser.add_argument(
        '--data_fn',
        dest='data_fn',
        type=str,
    )

    parser.add_argument(
        '--convert_all',
        dest='convert_all',
        action="store_true"
    )

    parser.add_argument(
        '--st_idx',
        dest='st_idx',
        type=int,
        default=0
    )

    parser.add_argument(
        '--ed_idx',
        dest='ed_idx',
        type=int,
        default=0
    )

    args = parser.parse_args()
    
    main(args)
