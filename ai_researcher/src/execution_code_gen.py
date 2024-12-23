from openai import OpenAI
import anthropic
from utils import call_api
import argparse
import json
import os
from utils import cache_output, format_plan_json, clean_code_output
import retry
from tqdm import tqdm
import random 
random.seed(2024)

@retry.retry(tries=3, delay=2)
def execution_generation_method(experiment_plan, demo_experiment_plan, demo_execution_code, openai_client, model, seed):
    prompt = "You are an expert researcher in Natural Language Processing and your job is to write Python code to implement the given research idea.\n"
    prompt += "Below is an example of an experiment plan and its corresponding implementation, which should serve as a template for your code implementation.\n"
    prompt += "Example experiment plan:\n" + str(demo_experiment_plan) + "\n"
    prompt += "Example implementation code:\n" + demo_execution_code + "\n\n"
    prompt += "Now write the code to implement the following idea:\n" + str(experiment_plan) + "\n"
    prompt += "You should mostly follow the example implmentation code, but customize the necessary parts, mostly:\n"
    prompt += "1. Test data: generate 3 - 5 appropriate test examples for sanity check evaluation. Each test example should include an input and an output. If the idea involves adding external information, simulate the external information and add it to the input in the proposed method implementation. The output can optionally include explanation as well.\n"
    prompt += "2. Baseline method: implement one of the baseline methods mentioned in the experiment plan.\n"
    prompt += "3. Proposed method: implement the proposed method mentioned in the experiment plan. Remember to return both the final answer and the intermediate outputs.\n"
    prompt += "4. Evaluation metrics: you should implement two evaluator functions. The style evaluator checks if the proposed method is producing all the desired intermediate components. The output evaluator checks if the final answers match the reference answers. For most tasks, you can ask for a binary yes/no judgment for the answer correctness. For generation tasks, you can change to score on a scale of 1 - 10 and compute average scores of the baseline and proposed methods.\n\n"
    prompt += "Directly output the full Python code without any prefix or suffix (e.g., do not prepend \"```python\").\n"
    
    # print (prompt)
    prompt_messages = [{"role": "user", "content": prompt}]
    response, cost = call_api(openai_client, model, prompt_messages, temperature=0., max_tokens=4096, seed=seed, json_output=False)
    return prompt, response, cost

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, default='claude-3-opus-20240229', help='api engine; https://openai.com/api/')
    parser.add_argument('--cache_name', type=str, default=None, required=True, help='cache file name for the retrieved papers')
    parser.add_argument('--idea_name', type=str, default=None, required=True, help='the specific idea to be formulated into an experiment plan')
    parser.add_argument('--seed', type=int, default=2024, help="seed for GPT-4 generation")
    args = parser.parse_args()

    with open("../keys.json", "r") as f:
        keys = json.load(f)
    
    with open("prompts/execution_demo.py", "r") as f:
        demo_execution_code = f.read().strip()
    
    with open("prompts/experiment_plan_demo.json", "r") as f:
        demo_experiment_plan = json.load(f)
        demo_experiment_plan = demo_experiment_plan["full_experiment_plan"]

    ANTH_KEY = keys["anthropic_key"]
    OAI_KEY = keys["api_key"]
    ORG_ID = keys["organization_id"]
    S2_KEY = keys["s2_key"]
    
    if "claude" in args.engine:
        client = anthropic.Anthropic(
            api_key=ANTH_KEY,
        )
        cache_dir = "../cache_results_claude_may/"
    else:
        client = OpenAI(
            organization=ORG_ID,
            api_key=OAI_KEY,
            base_url=BASE_URL
        )
        cache_dir = "../cache_results_gpt4/"

    ## load the demo examples
    if args.idea_name == "all":
        filenames = os.listdir(os.path.join(cache_dir, "experiment_plans", args.cache_name))
    else:
        filenames = ['_'.join(args.idea_name.lower().split()) + ".json"]

    # print (filenames)
    for filename in tqdm(filenames):
        try:
            print ("working on idea: ", filename)
            cache_file = os.path.join(cache_dir, "experiment_plans", args.cache_name, filename)
            
            ## load the idea 
            with open(cache_file, "r") as f:
                ideas = json.load(f)
            experiment_plan = ideas["full_experiment_plan"]
            if ideas["novelty"] == "yes":
                prompt, response, cost = execution_generation_method(experiment_plan, demo_experiment_plan, demo_execution_code, client, args.engine, args.seed)
                execution_code = clean_code_output(response.strip())
                print (execution_code)
                print ("Total cost: ", cost)
                
                ## save the cache
                if not os.path.exists(os.path.join(cache_dir, "execution", args.cache_name)):
                    os.makedirs(os.path.join(cache_dir, "execution", args.cache_name))
                with open(cache_file.replace("experiment_plans", "execution").replace(".json", ".py"), "w") as f:
                    f.write(execution_code + "\n")
                
            else:
                print ("idea not novel, skipped...")
        except: 
            print ("error in generating code for idea")

