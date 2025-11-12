from pathlib import Path
import torch
from peft import PeftModel, LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import (
    read_json,
    save_json
)
from argparse import ArgumentParser
from openai import OpenAI
from tqdm import tqdm

QWEN_KEY = ""
GPT_KEY = ""
DEEP_KEY = ""

def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--output_file', type=Path)
    parser.add_argument('--data_file', type=Path)
    parser.add_argument('--base_file', type=Path)
    parser.add_argument('--model', type=str, default='GPT')
    return parser.parse_args()


def zero_shot_prompt_l1(data):
    function_name = data['namespace'].split('.')[-1]
    function_functionality = f"The functionality of the function: {data['requirement']['Functionality']} The parameters of the function: {data['requirement']['Arguments']}"
    prompt = f"""Please write a python function called '{function_name}'. {function_functionality}."""
    messages=[
        {'role': 'user', 'content': prompt}
    ]
    return messages

def zero_shot_prompt_l2(data):
    function_name = data['namespace'].split('.')[-1]
    function_functionality = f"The functionality of the function: {data['requirement']['Functionality']} The parameters of the function: {data['requirement']['Arguments']}"
    prompt = f"""Please write a python function called '{function_name}'. {function_functionality}.
    The reference context you might need is as follows:
    {data['context']}"""
    messages=[
        {'role': 'user', 'content': prompt}
    ]
    return messages

def zero_shot_prompt_l3(data):
    function_name = data['namespace'].split('.')[-1]
    function_functionality = f"The functionality of the function: {data['requirement']['Functionality']} The parameters of the function: {data['requirement']['Arguments']}"
    prompt = f"""Please write a python function called '{function_name}'. {function_functionality}.
    The reference context you might need is as follows:
    ##intra file context:
    {data['intra_context']}
    ##cross file context:
    {data['cross_context']}"""
    messages=[
        {'role': 'user', 'content': prompt}
    ]
    return messages



def two_round_prompt_l1(base, data, requirement):
    function_name = data['namespace'].split('.')[-1]
    function_functionality = f"The functionality of the function: {data['requirement']['Functionality']} The parameters of the function: {data['requirement']['Arguments']}"
    function_requirement = f"The function should meet the following requirements: {requirement}"
    prompt = f"""Please write a python function called '{function_name}'. {function_functionality}"""
    function_code = base['completion']
    messages=[
        {'role': 'user', 'content': prompt},
        {'role': 'assistant', 'content': function_code},
        {'role': 'user', 'content': function_requirement}
    ]
    return messages

def two_round_prompt_l2(base, data, requirement):
    function_name = data['namespace'].split('.')[-1]
    function_functionality = f"The functionality of the function: {data['requirement']['Functionality']} The parameters of the function: {data['requirement']['Arguments']}"
    function_requirement = f"The function should meet the following requirements: {requirement}"
    prompt = f"""Please write a python function called '{function_name}'. {function_functionality}
    The reference context you might need is as follows:
    {data['context']}"""
    function_code = base['completion']
    messages=[
        {'role': 'user', 'content': prompt},
        {'role': 'assistant', 'content': function_code},
        {'role': 'user', 'content': function_requirement}
    ]
    return messages

def n_round_prompt(base, prompts, requirement):
    function_requirement = f"The function should meet the following requirements: {requirement}"
    messages=[
        {'role': 'user', 'content': prompts},
        {'role': 'assistant', 'content': base},
        {'role': 'user', 'content': function_requirement}
    ]
    return messages

def inference(args, messages):
    if args.model == "QWEN":
        client = OpenAI(
            api_key=QWEN_KEY, 
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    elif args.model == "GPT":
        client = OpenAI(
            api_key=GPT_KEY, 
            base_url="https://api.chatanywhere.tech/v1",
        )
    elif args.model == "DEEP":
        client = OpenAI(
            api_key=DEEP_KEY, 
            base_url="https://api.deepseek.com/v1",
        )
    response = client.chat.completions.create(
        # model="claude-3-5-sonnet-20241022",
        model = "deepseek-chat",
        # model = "qwen-coder-turbo",
        # model = "qwen2.5-32b-instruct",
        # model="gpt-4o",
        # model="gpt-4-turbo",
        messages=messages,
        temperature=0,
        top_p=1,
    )
    return response.choices[0].message.content


if __name__ == '__main__':
    args = get_parser()
    data = read_json(args.data_file,True)
    result = []
    level = args.level

    if args.way == 1:
        for item in tqdm(data):
            funcname = "zero_shot_prompt_l" + str(level)
            temp_data = {}
            temp_data['namespace'] = item['namespace']
            temp_data['completion'] = inference(args,funcname(item))
            result.append(temp_data)
        save_json(args.output_file,result,True)

    elif args.way == 2:
        round1 = read_json(args.base_file,True)
        funcname = "multi_round_prompt_l" + str(level)
        for item, base in tqdm(zip(data, round1), total=min(len(data), len(round1)), desc="Processing items"):
            for key, value in item['requirements'].items():
                temp_data = {}
                temp_data['namespace'] = item['namespace']
                temp_data['type'] = key
                temp_data['test'] = value['test']
                temp_data['completion'] = inference(args,funcname(base, item,value['requirement']))
                result.append(temp_data)
            save_json(args.output_file,result,True)

    else:
        round1 = read_json(args.base_file,True)
        for item, base in tqdm(zip(data, round1), total=min(len(data), len(round1)), desc="Processing items"):
            flag = True
            based = base['completion']
            function_name = item['namespace'].split('.')[-1]
            function_functionality = f"The functionality of the function: {item['requirement']['Functionality']} The parameters of the function: {item['requirement']['Arguments']}"
            prompts = [
                    {'role': 'user', 'content': f"""Please write a python function called '{function_name}'. {function_functionality}"""},
                    {'role': 'assistant', 'content': based}
            ]
            for key in item['multi-turn']:
                temp_data = {}
                temp_data['namespace'] = item['namespace']
                temp_data['type'] = key
                temp_data['test'] = item['requirements'][key]['test']
                prompts.append({'role': 'user', 'content': f"The function should meet the following requirements:{item['requirements'][key]['requirement']}"})
                temp_data['completion'] = inference(args,prompts)
                prompts.append({'role': 'assistant', 'content': temp_data['completion']})
                result.append(temp_data)
            save_json(args.output_file,result,False)