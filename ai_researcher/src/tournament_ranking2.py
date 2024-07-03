from openai import OpenAI
import anthropic
from utils import call_api
import argparse
import json
import os
from utils import cache_output, format_plan_json, avg_score
import random 
from tqdm import tqdm
import retry
from collections import defaultdict
random.seed(2024)

@retry.retry(tries=3, delay=2)
def overall_score(experiment_plan, criteria, openai_client, model, seed):
    prompt = "You are a professor specialized in Natural Language Processing and Large Language Models. You are given a project proposal and you need to score it.\n"
    prompt += "The project proposal is:\n\n" 
    prompt += format_plan_json(experiment_plan)
    prompt += "\n\nYour should follow the scoring rubrics:\n" + criteria + "\n"
    prompt += "Now directly provide me with a final score between 1 and 10, no other explanation needed.\n"

    prompt_messages = [{"role": "user", "content": prompt}]
    response, cost = call_api(openai_client, model, prompt_messages, temperature=0., max_tokens=2, seed=seed, json_output=False)
    return prompt, response, cost

@retry.retry(tries=3, delay=2)
def better_idea(idea_1, idea_2, method, openai_client, model, seed, few_shot_demos=None, temperature=0.):
    prompt = "You are a reviewer specialized in Natural Language Processing and Large Language Models. You are given two project summaries. One of them is accepted by a top AI conference (like ICLR or ACL) and the other one is rejected. Your task is to identify the one that has been accepted.\n"
    
    ## zero-shot methods
    if "zero_shot" in method:
        prompt += "The two project proposals are:\n\n" 
        prompt += "paper 1:\n" + format_plan_json(idea_1) + "\n\n"
        prompt += "paper 2:\n" + format_plan_json(idea_2) + "\n\n"
        # prompt += "\nYou can consider factors like novelty, soundness, excitement, and potential impact.\n"
    
        if method == "zero_shot":
            prompt += "Now decide which one is the accepted idea. Directly return a number 1 or 2 and nothing else.\n"
        elif method == "zero_shot_cot":
            prompt += "Now decide which one is the accepted idea. Think step by step by writing a meta-review to compare the strengths and weaknesses of both ideas and explain why one idea is better than the other. After the meta-review, start a new line and directly return a number 1 or 2 to indicate the accepted idea and end the response.\n"
    
    ## few-shot methods
    elif "few_shot" in method:
        prompt += "Here are some examples:\n" + few_shot_demos
        prompt += "\n\nThe two project summaries given to you are:\n\n" 
        prompt += "paper 1:\n" + format_plan_json(idea_1) + "\n\n"
        prompt += "paper 2:\n" + format_plan_json(idea_2) + "\n\n"
        # prompt += "\nYou should consider factors like novelty, soundness, excitement,and potential impact.\n"
        
        if method == "few_shot":
            prompt += "Now decide which one is the accepted idea. Follow the above examples: return a number 1 or 2 and nothing else.\n"
        elif method == "few_shot_cot":
            prompt += "Now decide which one is the accepted idea. Follow the above examples: give a meta-review and score to each paper, and then start a new line and directly return a number 1 or 2 to indicate the accepted idea and end the response.\n"

    # print (prompt)
    prompt_messages = [{"role": "user", "content": prompt}]
    response, cost = call_api(openai_client, model, prompt_messages, temperature=temperature, max_tokens=3000, seed=seed, json_output=False)
    return prompt, response, cost


def tournament_ranking(idea_lst, filename_lst, openai_client, model, seed, cache_name, max_round=5):
    # Initialize scores for each idea using the first 200 characters as keys
    scores = defaultdict(int)
    # decision_correct = 0
    # decision_all = 0
    
    # Helper function to conduct a single round of the tournament
    def single_round(ideas, current_round=0, decision_correct=0, decision_all=0):
        ## shuffle ideas in the first round
        if current_round == 0:
            random.shuffle(ideas)
        
        match_pairs = []
        # Sort ideas based on current scores
        sorted_ideas = sorted(ideas, key=lambda idea: scores[format_plan_json(idea)[:200]], reverse=True)

        for i in range(0, len(sorted_ideas), 2):
            if i + 1 < len(sorted_ideas):
                match_pairs.append((sorted_ideas[i], sorted_ideas[i+1]))
            else:
                # If there is an odd number of ideas, the last one automatically wins this round
                scores[format_plan_json(sorted_ideas[i])[:200]] += 1

        for idea1, idea2 in tqdm(match_pairs):
            prompt, result, cost = better_idea(idea1, idea2, "zero_shot", openai_client, model, seed)
            if result.strip() == '1':
                scores[format_plan_json(idea1)[:200]] += 1
                # if idea1["score"] >= idea2["score"]:
                #     decision_correct += 1
            else:
                scores[format_plan_json(idea2)[:200]] += 1
                # if idea1["score"] <= idea2["score"]:
                #     decision_correct += 1

            # decision_all += 1
        
        return 
    
    # Conduct the tournament rounds until only one idea remains
    current_round = 0
    score_predictions = {}
    while current_round < max_round:
        print ("Current round: ", current_round + 1)
        single_round(idea_lst[:], current_round)
        # print ("Currect decision accuracy: {} / {} = {}".format(decision_correct, decision_all, decision_correct / decision_all))
        current_round += 1

        # Convert scores to a list matching the order of the original idea list
        final_scores = [scores[format_plan_json(idea)[:200]] for idea in idea_lst]

        for i in range(len(filename_lst)):
            score_predictions[filename_lst[i]] = final_scores[i]
        
        cache_id = cache_name.split('/')[-1]
        with open("logs/score_predictions_{}_round_{}.json".format(cache_id, current_round), "w") as f:
            json.dump(score_predictions, f, indent=4)
    
    return final_scores




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, default='gpt-4-1106-preview', help='api engine; https://openai.com/api/')
    parser.add_argument('--cache_name', type=str, default="openreview_benchmark", help='cache file name for the retrieved papers')
    parser.add_argument('--max_round', type=int, default=5, help="seed for GPT-4 generation")
    parser.add_argument('--seed', type=int, default=2024, help="seed for GPT-4 generation")
    args = parser.parse_args()

    with open("../keys.json", "r") as f:
        keys = json.load(f)
    random.seed(args.seed)

    ANTH_KEY = keys["anthropic_key"]
    OAI_KEY = keys["api_key"]
    ORG_ID = keys["organization_id"]
    
    if "claude" in args.engine:
        client = anthropic.Anthropic(
            api_key=ANTH_KEY,
        )
    else:
        client = OpenAI(
            organization=ORG_ID,
            api_key=OAI_KEY
        )

    filenames = os.listdir("../{}".format(args.cache_name))
    filenames = [f for f in filenames if f.endswith(".json")]

    score_predictions = {}
    filename_lst = []
    idea_lst = []

    for filename in filenames:
        with open("../{}/{}".format(args.cache_name, filename), "r") as f:
            paper = json.load(f)
        if "full_experiment_plan" in paper and isinstance(paper["full_experiment_plan"], dict):
            summary = paper["full_experiment_plan"]
            idea_lst.append(summary)
            filename_lst.append(filename)

    print ("total #ideas: ", len(idea_lst))
    final_scores = tournament_ranking(idea_lst, filename_lst, client, args.engine, args.seed, args.cache_name, args.max_round)
    

    
