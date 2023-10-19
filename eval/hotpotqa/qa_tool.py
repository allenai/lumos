import re
import sys
import math
import requests
import numpy as np
import torch
import wolframalpha
import asyncio
from bs4 import BeautifulSoup
from zeno_build.models import lm_config
from zeno_build.prompts import chat_prompt
from transformers import DPRReader, DPRReaderTokenizer
from data.fast_prompt import generate_from_openai_chat_completion
from eval.hotpotqa.utils import *


device = "cuda:0" if torch.cuda.is_available() else "cpu"
tokenizer = DPRReaderTokenizer.from_pretrained("facebook/dpr-reader-multiset-base")
model = DPRReader.from_pretrained("facebook/dpr-reader-multiset-base").to(device)


class QAActions:
    def KnowledgeQuery(x, typ=None):
        return search_step(x, typ)

    def SearchQuery(x):
        search_url = f"https://www.google.com/search?q={x.replace(' ', '+')}"
        response_text = requests.get(search_url).text

        soup = BeautifulSoup(response_text, features="html.parser")
        page_entries = [p.get_text().strip() for p in soup.find_all("div")]
        context = ""
        available_resource_num = 0
        used_page_entries = list()
        for i, entry in enumerate(page_entries):
            if len(entry) > 200 or "›" in entry or len(entry.split(" ")) < 10 or "..." not in entry or entry in used_page_entries:
                continue
            available_resource_num += 1
            context += f"Evidence {str(available_resource_num)}: " + entry + '\n'
            used_page_entries.append(entry)
            if available_resource_num >= 10:
                break

        return context.strip()
    
    def ParagraphRetrieve(x):
        sim_scores = []
        for i in range(0, len(x["titles"]), 10):
            encoded_inputs = tokenizer(
                questions=x["question"],
                titles=x["titles"][i: i+10],
                texts=x["texts"][i: i+10],
                padding='max_length',
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            outputs = model(**encoded_inputs)
            sim_scores += outputs.relevance_logits.cpu().detach().numpy().tolist()
        
        sim_scores = np.array(sim_scores)
        relevant_para_idx = get_max_idx(sim_scores, 5)
        relevant_para = ""
        for i, idx in enumerate(relevant_para_idx):
            relevant_para += f"Evidence {str(i+1)}: " + x["texts"][idx] + "\n"

        return relevant_para

    def QA(x, model_config):
        predictions = asyncio.run(
            generate_from_openai_chat_completion(
                full_contexts=[chat_prompt.ChatMessages(
                    [{"role": "user", "content": f"Please answer the following question according to the context. Please answer with one brief entity, beginning with 'Answer: '. If the answer can't be found in the context, answer 'No answer'. No more details and explanations.\n\nQuestion: {x['question']}\nContext: {x['context']}\n"}]
                )],
                model_config=model_config,
                temperature=0,
                max_tokens=100,
                tqdm=False
            )
        )

        if "No answer" in predictions[0]:
            if "Must answer yes/no" not in x['question']:
                # try the input without context
                predictions = asyncio.run(
                    generate_from_openai_chat_completion(
                        full_contexts=[chat_prompt.ChatMessages(
                            [{"role": "user", "content": f"Please answer the following question. Please answer with one brief entity, beginning with 'Answer: '. If the answer can't be answered, output 'No answer'. No more details and explanations.\n\nQuestion: {x['question']}\n"}]
                        )],
                        model_config=model_config,
                        temperature=0,
                        max_tokens=100,
                        tqdm=False
                    )
                )
            else:
                # force to predict yes/no
                predictions = asyncio.run(
                    generate_from_openai_chat_completion(
                        full_contexts=[chat_prompt.ChatMessages(
                            [{"role": "user", "content": f"Please answer the following question. Please answer with one brief entity, beginning with 'Answer: '.\n\nQuestion: {x['question']}\n"}]
                        )],
                        model_config=model_config,
                        temperature=0,
                        max_tokens=100,
                        tqdm=False
                    )
                )

        return f"Question: {x['question']}; {predictions[0]}"


if __name__ == "__main__":
    main()