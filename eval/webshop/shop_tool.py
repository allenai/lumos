import re
import sys
import math
import requests
import numpy as np
import torch
import asyncio
from zeno_build.models import lm_config
from zeno_build.prompts import chat_prompt
from sentence_transformers import CrossEncoder

semantic_match_model = CrossEncoder(
    "cross-encoder/stsb-roberta-large",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    max_length=512,
)

def get_max_idx(x, k):
    max_num_idx = np.argsort(x)[-k:]
    max_num_idx = max_num_idx[::-1]
    return max_num_idx

class ShopActions:
    def Search(env, action_args):
        observation, _, _, _ = env.step(f"search[{action_args}]")
        available_actions = env.get_available_actions()

        items = available_actions['clickables'][2:4]
        execution_results = ""
        for item in items:
            pos_item = observation.find(item.upper() + " [SEP] ") + len(item.upper() + " [SEP] ")
            pos_item_ed = observation[pos_item:].find(" [SEP] ") + pos_item
            execution_results += f"{item} - {observation[pos_item: pos_item_ed]} ** "

        return execution_results[:-2]

    def Click(env, curr_action):
        observation, reward, done, _ = env.step(curr_action)
        available_actions = env.get_available_actions()

        if not done:
            return available_actions['clickables']
        else:
            return reward

    def FeatureRetrieve(query, features):
        sim_scores = list()
        model_input = [[query, feature] for feature in features]
        if len(model_input) == 0:
            return ""
        sim_scores = semantic_match_model.predict(
            model_input,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=64,
        )
        if len(sim_scores) >= 3:
            relevant_para_idx = get_max_idx(sim_scores, 3)
        elif len(sim_scores) >= 1:
            relevant_para_idx = get_max_idx(sim_scores, len(sim_scores))
        
        relevant_para = ""
        for i, idx in enumerate(relevant_para_idx):
            relevant_para += features[idx] + ", "

        return relevant_para[:-2]

    def Pick(query, items):
        sim_scores = list()
        model_input = [[query, item] for item in items]
        sim_scores = semantic_match_model.predict(
            model_input,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=64,
        )
        relevant_para_idx = get_max_idx(sim_scores, 1)

        return items[relevant_para_idx[0]].split(" - ")[0]

if __name__ == "__main__":
    main()