import os
import re
import sys
import math
import openai
import wolframalpha
import requests
from bs4 import BeautifulSoup
from eval.strategyqa.qa_tool import *

wolfram_app_id = os.environ["WOLFRAM_API_KEY"]
client = wolframalpha.Client(wolfram_app_id)

entry_knowledge_sep = "*****"
table_text_sep = "+++++"

def search_step(x, typ=None):
    if "https://" in x:
        search_url = x
    elif "/wiki/" in x:
        search_url = f"https://en.wikipedia.org{x}"
    else:
        entity_ = x.replace(" ", "+")
        search_url = f"https://en.wikipedia.org/w/index.php?search={entity_}"
    response_text = requests.get(search_url).text

    soup = BeautifulSoup(response_text, features="html.parser")
    result_divs = soup.find_all("div", {"class": "mw-search-result-heading"})
    if result_divs:
        result_titles = [clean_str(div.get_text().strip()) for div in result_divs]
        return result_titles[:5]
    else:
        page_table = [p.get_text().strip() for p in soup.find_all("tr")]
        page = [p.get_text().strip() for p in soup.find_all("p") + soup.find_all("ul")]
        page_text, page_table_text = "", ""
        if any("may refer to:" in p for p in page):
            page = re.findall(r"<li><a href=\"(.*?)\" title=", response_text)
            result_titles = []
            for p in page:
                if "Category:" in p:
                    continue
                if '"' in p:
                    p = p[: p.find('"')]
                result_titles.append(p)
            return result_titles[:10]
        else:
            for p in page_table[:20]:
                page_table_text += clean_str(p).replace("\n", " ") + ' '
            
            for p in page:
                if len(p.split(" ")) > 10:
                    page_text += clean_str(p)
                    if not p.endswith("\n"):
                        page_text += "\n"
            if typ == "ambiguity":
                page_text = get_page_obs(page_text, 15)
            elif typ == "failure":
                page_text = get_page_obs(page_text, 15)
            else:
                page_text = get_page_obs(page_text, 40)
            
            if page_table_text:
                page_text = page_table_text + f" {table_text_sep} " + page_text

            return page_text

def get_max_idx(x, k):
    max_num_idx = np.argsort(x)[-k:]
    max_num_idx = max_num_idx[::-1]
    return max_num_idx

def WA_result(question):
    tmp = next(client.query(question).results)
    if "plaintext" not in tmp.subpod:
        return [i["plaintext"] for i in tmp.subpod]
    else:
        return tmp.subpod["plaintext"]

def clean_str(p):
    try:
        return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")
    except:
        return p.encode("unicode-escape").decode("unicode-escape").encode("latin1").decode("utf-8")

def get_page_obs(page, num=None):
    paragraphs = page.split("\n")
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    sentences = []
    for p in paragraphs:
        sentences += p.split('. ')
    sentences = [s.strip() + '.' for s in sentences if s.strip()]

    if num:
        return f" {entry_knowledge_sep} ".join(sentences[:num])
    else:
        return f" {entry_knowledge_sep} ".join(sentences)

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