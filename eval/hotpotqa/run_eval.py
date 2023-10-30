import argparse
import os
import re
import json
import random
import string
import evaluate
from googlesearch import search
from collections import Counter
from tqdm import tqdm
from eval.hotpotqa.qa_tool import *
from eval.hotpotqa.utils import *
from eval.utils import generate_completions, load_hf_lm_and_tokenizer


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return 0
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return 0

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def llm_accuracy_score(question, prediction, ground_truth):
    predictions = asyncio.run(
        generate_from_openai_chat_completion(
            full_contexts=[chat_prompt.ChatMessages(
                [{"role": "user", "content": f"Please judge whether the model prediction can be regarded as correct according to the given question and its ground-truth answer. Please answer 'CORRECT' or 'INCORRECT'. No more details and explanations.\n\nQuestion: {question}\nModel prediction: {prediction}\nGround-truth answer: {ground_truth}\n"}]
            )],
            model_config=eval_model_config,
            temperature=0,
            max_tokens=100,
            tqdm=False
        )
    )

    return 1 if predictions[0].strip() == 'CORRECT' else 0


def decide_final_ans(inter_results):
    context = ""
    for variable in inter_results["results"]:
        if isinstance(inter_results["results"][variable], str) and "Question: " in inter_results["results"][variable]:
            context += inter_results["results"][variable] + '; '
    context = context.strip()
    orig_question = inter_results["question"]

    return QAActions.QA({"context": context, "question": orig_question}, model_config)


def execute(action, inter_results, subgoals=None, idx=None, orig_question=None):
    action_variable, action_name, action_args = parse_actions(action)
    variables = findvariables(action_args)
    title_text_sep = "***"

    if action_name in ["", "KnowledgeQuery", "ParagraphRetrieve", "QA", "Calculator", "Code"]:
        try:
            if action_name == "KnowledgeQuery":
                updated_action_args = action_args
                for variable in variables:
                    updated_action_args = updated_action_args.replace(variable, str(inter_results[variable]))
                knowledge = QAActions.KnowledgeQuery(updated_action_args)

                if isinstance(knowledge, str):
                    if table_text_sep not in knowledge:
                        return action_variable[0], [f"{updated_action_args} {title_text_sep} "+e.strip() for e in knowledge.split(entry_knowledge_sep)]
                    else:
                        table_knowledge = [f"Reference Table: {title_text_sep} " + knowledge.split(table_text_sep)[0].strip()]
                        entry_knowledge = [f"{updated_action_args} {title_text_sep} "+e.strip() for e in knowledge.split(table_text_sep)[1].strip().split(entry_knowledge_sep)]
                        return action_variable[0], table_knowledge + entry_knowledge
                else:
                    all_knowledge = list()
                    for entry in knowledge:
                        if len(knowledge) == 5:
                            entry_knowledge = QAActions.KnowledgeQuery(entry, "failure")
                        else:
                            entry_knowledge = QAActions.KnowledgeQuery(entry, "ambiguity")
                        if isinstance(entry_knowledge, str):
                            if f" {table_text_sep} " not in entry_knowledge:
                                all_knowledge += [f"{entry} {title_text_sep} "+e.strip() for e in entry_knowledge.split(entry_knowledge_sep)]
                            else:
                                all_table_knowledge = [f"Reference Table: {title_text_sep} " + entry_knowledge.split(table_text_sep)[0].strip()]
                                all_entry_knowledge = [f"{entry} {title_text_sep} "+e.strip() for e in entry_knowledge.split(table_text_sep)[1].strip().split(entry_knowledge_sep)]
                                all_knowledge += all_table_knowledge + all_entry_knowledge
                    return action_variable[0], all_knowledge
            elif action_name == "ParagraphRetrieve":
                updated_action_args = dict()
                updated_action_args["question"] = action_args.split(", Query: ")[1]
                updated_action_args["titles"] = [text.split(title_text_sep)[0].strip() for text in inter_results[variables[0]]]
                updated_action_args["texts"] = [text.split(title_text_sep)[1].strip() for text in inter_results[variables[0]]]

                if "Reference Table:" not in updated_action_args["titles"][0]:
                    return action_variable[0], QAActions.ParagraphRetrieve(updated_action_args)
                else:
                    return action_variable[0], "Reference Table: " + updated_action_args["texts"][0] + '\n' + QAActions.ParagraphRetrieve(updated_action_args)
            elif action_name == "QA":
                updated_action_args = dict()
                if orig_question:
                    updated_action_args["question"] = orig_question
                else:
                    updated_action_args["question"] = action_args.split(", Question: ")[1]
                updated_action_args["context"] = ""
                for variable in action_args.split(", Question: ")[0][1: -1].split(", "):
                    updated_action_args["context"] += inter_results[variable] + '\n'
                updated_action_args["context"] = updated_action_args["context"].strip()
                
                llm_pred = QAActions.QA(updated_action_args, model_config)
                if "no answer" not in llm_pred.lower():
                    return action_variable[0], llm_pred
                else:
                    if orig_question:
                        updated_action_args["question"] = action_args.split(", Question: ")[1]
                        llm_pred = QAActions.QA(updated_action_args, model_config)
                        if "no answer" not in llm_pred.lower():
                            return action_variable[0], llm_pred
                    
                    for page in search(updated_action_args["question"] + " Wikipedia", tld="co.in", num=10, stop=3, pause=2):
                        if "en.wikipedia.org/wiki" in page:
                            entity = page[len("https://en.wikipedia.org/wiki/"):].replace('_', ' ')
                            knowledge = QAActions.KnowledgeQuery(page)
                            if isinstance(knowledge, str):
                                if table_text_sep not in knowledge:
                                    candidate_paras = [f"{entity} {title_text_sep} "+e for e in knowledge.split(entry_knowledge_sep)]
                                else:
                                    table_paras = [f"Reference Table: {title_text_sep} " + knowledge.split(table_text_sep)[0].strip()]
                                    entity_paras = [f"{entity} {title_text_sep} "+e.strip() for e in knowledge.split(table_text_sep)[1].strip().split(entry_knowledge_sep)]
                                    candidate_paras = table_paras + entity_paras
                            else:
                                continue
                            qa_args = dict()
                            qa_args["question"] = updated_action_args["question"]
                            qa_args["titles"] = [text.split(title_text_sep)[0].strip() for text in candidate_paras]
                            qa_args["texts"] = [text.split(title_text_sep)[1].strip() for text in candidate_paras]
                            paras = QAActions.ParagraphRetrieve(qa_args)

                            updated_action_args["context"] = paras
                            llm_pred = QAActions.QA(updated_action_args, model_config)
                            if "no answer" not in llm_pred.lower():
                                return action_variable[0], llm_pred
                    return action_variable[0], "No answer."
            elif action_name == "Calculator":
                updated_action_args = action_args
                for variable in variables:
                    updated_action_args = updated_action_args.replace(variable, str(inter_results[variable])+'.')
                MATH_OP = ['*', '/', '+', '-', '^', '=', '<', '>']
                for op in MATH_OP:
                    updated_action_args = updated_action_args.replace(op, '')

                if subgoals:
                    subgoal = subgoals[-1]  # usually appear in the last subgoal
                    subgoal = subgoal[subgoal.find(":") + 2:]
                    llm_pred = QAActions.QA({"context": updated_action_args, "question": subgoal}, model_config)
                    if "no answer" not in llm_pred.lower():
                        return action_variable[0], llm_pred
                    
                if orig_question:
                    llm_pred = QAActions.QA({"context": updated_action_args, "question": orig_question}, model_config)
                    if "no answer" not in llm_pred.lower():
                        return action_variable[0], llm_pred
                    else:
                        return action_variable[0], "No answer."
            elif action_name == "":
                return action_variable[0], updated_action_args
        except:
            return "", ""


def lumos_iterative(args):
    random.seed(42)

    print("Loading data...")
    test_data, subgoals, actions, plan_prompts, ground_prompts, all_subgoals_actions, subgoal_results, solved_idx = list(), list(), list(), list(), list(), list(), list(), list()
    with open(os.path.join(args.data_dir, f"test.json")) as fin:
        data = json.load(fin)
        for example in data:
            test_data.append({
                "question": example["question"],
                "answer": example["answer"]
            })

    if args.max_num_examples and len(test_data) > args.max_num_examples:
        test_data = test_data[:args.max_num_examples]
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    for i in tqdm(range(10)):
        if i % 2 == 0:
            prompt_prefix = "Please provide a reasonable subgoal-based plan to solve the given task.\n"
        else:
            prompt_prefix = "Please ground the given subgoal to corresponding executable actions for solving the given task. The grounded actions must be the one in available action list.\n\n" \
                            "The available action list is 'KnowledgeQuery', 'ParagraphRetrieve', 'QA', 'Calculator', and 'Code'.\n" \
                            "Calculator(formula): Calculate the input formula; KnowledgeQuery(query): Capture the relevant webpages according to the query; ParagraphRetrieve(context, query): Given a query, retrieve the most relevant paragraphs from the given context; QA(context, query): Given context, answer the given query; Calculator(formula): Calculate the input formula; Code(pseudo_code): Generate a Python function that corresponds to the pseudo code.\n\n"
        
        assert not subgoals or len(subgoals) == len(test_data)
        assert not actions or len(actions) == len(test_data)
        
        if i % 2 == 0:
            for j, example in enumerate(test_data):
                if i == 0:
                    all_subgoals_actions.append({"question": example["question"].strip(), "answer": example["answer"], "subgoals": [], "actions": [], "results": dict()})
                    plan_prompt = "<|user|>\n" + prompt_prefix + "Task: " + example["question"].strip() + "; Initial Environment Description: None.\n<|assistant|>\n"
                    plan_prompts.append(plan_prompt)
                else:
                    plan_prompts[j] += subgoals[j].strip() + f"\n\n<|user|>\nThe executed result for Subgoal {str(i // 2)} is {all_subgoals_actions[j]['actions'][-1].split('=')[-1].strip()}. Should we stop planning?\n<|assistant|>\n"
        else:
            for j, example in enumerate(test_data):
                if i == 1:
                    ground_prompt = "<|user|>\n" + prompt_prefix + "Task: " + example["question"].strip() + f" \nSubgoal to be grounded: {subgoals[j].strip()}\n<|assistant|>\n"
                    ground_prompts.append(ground_prompt)
                else:
                    ground_prompts[j] += actions[j].strip() + f"\n\n<|user|>\nSubgoal to be grounded: {subgoals[j].split('No, I will keep planning.')[-1].strip()} The grounding could be based on the following results:"
                    for subgoal_idx in sorted(list(subgoal_results[j].keys())):
                        if "Answer: " in subgoal_results[j][subgoal_idx]:
                            ground_prompts[j] += f" The execution result for {subgoal_idx} is {subgoal_results[j][subgoal_idx].split('Answer: ')[1]}"
                    ground_prompts[j] += "\n<|assistant|>\n"

        if args.model_name_or_path:
            print("Loading model and tokenizer...")
            if i % 2 == 0:
                model_name_or_path = os.path.join(args.model_name_or_path, "lumos_complex_qa_plan_iterative")
            else:
                model_name_or_path = os.path.join(args.model_name_or_path, "lumos_complex_qa_ground_iterative")
            
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
        
        if i % 2 == 0:
            for j in range(len(test_data)):
                if j in solved_idx:
                    continue
                if "Yes, I will stop planning" in subgoals[j].strip():
                    solved_idx.append(j)
                    if all_subgoals_actions[j]["results"]:
                        all_subgoals_actions[j]["results"]["final_ans"] = decide_final_ans(all_subgoals_actions[j])
        else:
            for j in range(len(test_data)):
                if j in solved_idx or all_subgoals_actions[j]["results"] is None:
                    continue
                all_subgoals_actions[j]["subgoals"].append(subgoals[j].strip())
                cur_subgoal_idx = subgoals[j].strip().split(':')[0].split("No, I will keep planning.")[-1].strip()
                for k, action in enumerate(actions[j].strip().split('; ')):
                    try:
                        results_variable, execution_results = execute(action, all_subgoals_actions[j]["results"], subgoals=subgoals[j].strip().split("; "))
                        all_subgoals_actions[j]["results"][results_variable] = execution_results
                    except:
                        all_subgoals_actions[j]["results"] = None
                if i == 1:
                    subgoal_results.append({cur_subgoal_idx: execution_results})
                else:
                    subgoal_results[j][cur_subgoal_idx] = execution_results
                all_subgoals_actions[j]["actions"].append(action + " = " + execution_results)
    
    em, llm_acc = 0, 0
    with open(os.path.join(args.save_dir, f"predictions_iterative.jsonl"), "w") as f:
        for i, subgoal_action in enumerate(tqdm(all_subgoals_actions)):
            if subgoal_action["results"] and "final_ans" in subgoal_action["results"]:
                if "Answer: " in subgoal_action["results"]["final_ans"]:
                    pred = subgoal_action["results"]["final_ans"].split("Answer: ")[1]
                else:
                    pred = subgoal_action["results"]["final_ans"]

                norm_pred = normalize_answer(pred)
                norm_ans = normalize_answer(subgoal_action["answer"])

                f.write(json.dumps({"question": subgoal_action["question"], 
                                    "pred": pred, 
                                    "answer": subgoal_action["answer"]
                                    })+'\n')

                if norm_ans == norm_pred:
                    em += 1
                llm_acc += llm_accuracy_score(subgoal_action["question"], pred, subgoal_action["answer"])
    print("EM:", em, "LLM_Acc:", llm_acc)


def lumos_onetime(args):
    random.seed(42)

    print("Loading data...")
    test_data, subgoals, actions, plan_prompts, ground_prompts, all_subgoals_actions = list(), list(), list(), list(), list(), list()
    with open(os.path.join(args.data_dir, f"test.json")) as fin:
        data = json.load(fin)
        for example in data:
            test_data.append({
                "question": example["question"],
                "answer": example["answer"]
            })

    if args.max_num_examples and len(test_data) > args.max_num_examples:
        test_data = test_data[:args.max_num_examples]
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    for i in range(2):
        if i % 2 == 0:
            prompt_prefix = "Please provide a reasonable subgoal-based plan to solve the given task.\n"
        else:
            prompt_prefix = "Please ground the given subgoal to corresponding executable actions for solving the given task. The grounded actions must be the one in available action list.\n\n" \
                            "The available action list is 'KnowledgeQuery', 'ParagraphRetrieve', 'QA', 'Calculator', and 'Code'.\n" \
                            "Calculator(formula): Calculate the input formula; KnowledgeQuery(query): Capture the relevant webpages according to the query; ParagraphRetrieve(context, query): Given a query, retrieve the most relevant paragraphs from the given context; QA(context, query): Given context, answer the given query; Calculator(formula): Calculate the input formula; Code(pseudo_code): Generate a Python function that corresponds to the pseudo code.\n\n"
        
        assert not subgoals or len(subgoals) == len(test_data)
        assert not actions or len(actions) == len(test_data)
        
        if i == 0:
            for j, example in enumerate(test_data):
                all_subgoals_actions.append({"question": example["question"].strip(), "answer": example["answer"], "subgoals": [], "actions": [], "results": dict()})
                plan_prompt = "<|user|>\n" + prompt_prefix + "Task: " + example["question"].strip() + "; Initial Environment Description: None.\n<|assistant|>\n"
                plan_prompts.append(plan_prompt)
        else:
            for j, example in enumerate(test_data):
                ground_prompt = "<|user|>\n" + prompt_prefix + "Task: " + example["question"].strip() + f" \nSubgoals to be grounded: {subgoals[j].strip()}\n<|assistant|>\n"
                ground_prompts.append(ground_prompt)

        if args.model_name_or_path:
            print("Loading model and tokenizer...")
            if i == 0:
                model_name_or_path = os.path.join(args.model_name_or_path, "lumos_complex_qa_plan_onetime")
            else:
                model_name_or_path = os.path.join(args.model_name_or_path, "lumos_complex_qa_ground_onetime")
            
            model, tokenizer = load_hf_lm_and_tokenizer(
                model_name_or_path=model_name_or_path, 
                tokenizer_name_or_path=model_name_or_path, 
                load_in_8bit=args.load_in_8bit, 
                load_in_half=True,
                gptq_model=args.gptq
            )
            new_line_token = tokenizer.encode("\n", add_special_tokens=False)[-1] # get the last token because the tokenizer may add space tokens at the start.
            if i == 0:
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
        
        if i == 1:
            for j in tqdm(range(len(test_data))):
                all_subgoals_actions[j]["subgoals"].append(subgoals[j].strip())
                for k, action in enumerate(actions[j].strip().split('; ')):
                    try:
                        if k == len(actions[j].strip().split('; ')) - 1:
                            results_variable, execution_results = execute(action, all_subgoals_actions[j]["results"], subgoals=subgoals[j].strip().split("; "), idx=k, orig_question=all_subgoals_actions[j]["question"])
                        else:
                            results_variable, execution_results = execute(action, all_subgoals_actions[j]["results"], subgoals=subgoals[j].strip().split("; "))
                        all_subgoals_actions[j]["results"][results_variable] = execution_results
                    except:
                        all_subgoals_actions[j]["results"] = None
                        break
    
    em, llm_acc = 0, 0
    with open(os.path.join(args.save_dir, f"predictions_onetime.jsonl"), "w") as f:
        for i, subgoal_action in enumerate(all_subgoals_actions):
            if subgoal_action["results"]:
                try:
                    variables = sorted(list([int(k[1:]) for k in subgoal_action["results"].keys()]))
                    final_variable = 'R' + str(variables[-1])
                    if "Answer: " in subgoal_action["results"][final_variable]:
                        pred = subgoal_action["results"][final_variable].split("Answer: ")[1]
                    else:
                        pred = subgoal_action["results"][final_variable]
                    
                    norm_pred = normalize_answer(pred)
                    norm_ans = normalize_answer(subgoal_action["answer"])

                    f.write(json.dumps({"question": subgoal_action["question"], 
                                        "pred": pred, 
                                        "answer": subgoal_action["answer"]
                                        })+'\n')

                    if norm_ans == norm_pred:
                        em += 1
                    llm_acc += llm_accuracy_score(subgoal_action["question"], pred, subgoal_action["answer"])
                except:
                    continue
    print("EM:", em, "LLM_Acc:", llm_acc)


def cot(args):
    random.seed(42)

    print("Loading data...")
    test_data, prompts, all_outputs = list(), list(), list()
    with open(os.path.join(args.data_dir, f"test.json")) as fin:
        data = json.load(fin)
        for example in data:
            test_data.append({
                "question": example["question"],
                "answer": example["answer"]
            })

    if args.max_num_examples and len(test_data) > args.max_num_examples:
        test_data = test_data[:args.max_num_examples]
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    for i, example in enumerate(test_data):
        all_outputs.append({"question": example["question"].strip(), "answer": example["answer"], "responses": None})
        prompt = "<|user|>\n" + example["question"].strip() + "\n<|assistant|>\n"
        prompts.append(prompt)

    if args.model_name_or_path:
        print("Loading model and tokenizer...")
        model_name_or_path = os.path.join(args.model_name_or_path, "complex_qa_cot")
        
        model, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=model_name_or_path, 
            tokenizer_name_or_path=model_name_or_path, 
            load_in_8bit=args.load_in_8bit, 
            load_in_half=True,
            gptq_model=args.gptq
        )
        new_line_token = tokenizer.encode("\n", add_special_tokens=False)[-1] # get the last token because the tokenizer may add space tokens at the start.
        outputs = generate_completions(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_new_tokens=512,
            batch_size=args.eval_batch_size,
            stop_id_sequences=[[new_line_token]]
        )
    
    with open(os.path.join(args.save_dir, f"predictions_cot.jsonl"), "w") as f:
        em, llm_acc = 0, 0
        for i in range(len(all_outputs)):
            try:
                pred = outputs[i].split("decomposed questions, the answer is")[1].strip()[:-1]
                if "false" == pred.lower() or "false." == pred.lower():
                    pred = "no"
                elif "true" == pred.lower() or "true." == pred.lower():
                    pred = "yes"
                
                norm_pred = normalize_answer(pred)
                norm_ans = normalize_answer(all_outputs[i]["answer"])

                f.write(json.dumps({"question": all_outputs[i]["question"], "pred": pred, 
                                    "answer": all_outputs[i]["answer"]
                                    })+'\n')

                if norm_ans == norm_pred:
                    em += 1
                llm_acc += llm_accuracy_score(all_outputs[i]["question"], pred, all_outputs[i]["answer"])
            except:
                continue
    print("EM:", em, "LLM_Acc:", llm_acc)


def direct(args):
    random.seed(42)

    print("Loading data...")
    test_data, prompts, all_outputs = list(), list(), list()
    with open(os.path.join(args.data_dir, f"test.json")) as fin:
        data = json.load(fin)
        for example in data:
            test_data.append({
                "question": example["question"],
                "answer": example["answer"]
            })

    if args.max_num_examples and len(test_data) > args.max_num_examples:
        test_data = test_data[:args.max_num_examples]
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    for i, example in enumerate(test_data):
        all_outputs.append({"question": example["question"].strip(), "answer": example["answer"], "responses": None})
        prompt = "<|user|>\n" + example["question"].strip() + "\n<|assistant|>\n"
        prompts.append(prompt)

    if args.model_name_or_path:
        print("Loading model and tokenizer...")
        model_name_or_path = os.path.join(args.model_name_or_path, "complex_qa_direct")
        
        model, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=model_name_or_path, 
            tokenizer_name_or_path=model_name_or_path, 
            load_in_8bit=args.load_in_8bit, 
            load_in_half=True,
            gptq_model=args.gptq
        )
        new_line_token = tokenizer.encode("\n", add_special_tokens=False)[-1] # get the last token because the tokenizer may add space tokens at the start.
        outputs = generate_completions(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_new_tokens=512,
            batch_size=args.eval_batch_size,
            stop_id_sequences=[[new_line_token]]
        )
    
    em, llm_acc = 0, 0
    with open(os.path.join(args.save_dir, f"predictions_direct.jsonl"), "w") as f:
        for i in range(len(all_outputs)):
            pred = outputs[i].strip()
            if "false" == pred.lower() or "false." == pred.lower():
                pred = "no"
            elif "true" == pred.lower() or "true." == pred.lower():
                pred = "yes"

            norm_pred = normalize_answer(pred)
            norm_ans = normalize_answer(all_outputs[i]["answer"])

            f.write(json.dumps({"question": all_outputs[i]["question"], 
                                "pred": pred, 
                                "answer": all_outputs[i]["answer"]})+'\n')

            if norm_ans == norm_pred:
                em += 1
            llm_acc += llm_accuracy_score(all_outputs[i]["question"], pred, all_outputs[i]["answer"])
    print("EM:", em, "LLM_Acc:", llm_acc)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--max_num_examples", type=int, default=None, help="maximum number of examples to evaluate.")
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--model_name_or_path", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="batch size for evaluation.")
    parser.add_argument("--formulation", type=str, default='lumos-iterative', help="considered formulation.")
    parser.add_argument("--load_in_8bit", action="store_true", help="load model in 8bit mode, which will reduce memory and speed up inference.")
    parser.add_argument("--gptq", action="store_true", help="If given, we're evaluating a 4-bit quantized GPTQ model.")
    args = parser.parse_args()

    model_config = lm_config.LMConfig(provider="openai_chat", model="gpt-3.5-turbo")
    eval_model_config = lm_config.LMConfig(provider="openai_chat", model="gpt-4")

    if args.formulation == "lumos_iterative":
        lumos_iterative(args)
    elif args.formulation == "lumos_onetime":
        lumos_onetime(args)
    elif args.formulation == "cot":
        cot(args)
    elif args.formulation == "direct":
        direct(args)
