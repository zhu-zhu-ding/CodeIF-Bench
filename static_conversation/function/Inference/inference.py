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
    parser.add_argument('--base_file', type=Path, default=None)
    parser.add_argument('--model', type=str, default='GPT')
    parser.add_argument('--way', type=int)
    return parser.parse_args()

def zero_shot_prompt(data):
    prompt = f"""You are an expert Python programmer, and here is your task: {data['prompt']}"""
    messages=[
        {'role': 'user', 'content': prompt}
    ]
    return messages

def two_round_prompt(base, data, requirement):
    prompt = f"""You are an expert Python programmer, and here is your task: {data['prompt']}"""
    function_code = base['completion']
    function_requirement = f"The function should meet the following requirements: {requirement}"
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
        # model = "deepseek-chat",
        # model = "qwen-coder-turbo",
        # model = "qwen2.5-32b-instruct",
        # model="gpt-4o",
        model="gpt-4-turbo",
        messages=messages,
        temperature=0,
        top_p=1,
    )
    return response.choices[0].message.content


if __name__ == '__main__':
    args = get_parser()
    data = read_json(args.data_file,True)
    round1 = read_json(args.base_file,False)
    result = []

    if args.way == 1:
        for item in tqdm(data):
            temp_data = {}
            temp_data['task_id'] = item['task_id']
            temp_data['completion'] = inference(args,zero_shot_prompt(item))
            result.append(temp_data)
        save_json(args.output_file,result,False)

    elif args.way == 2:
        round1 = read_json(args.base_file,False)
        for item, base in tqdm(zip(data, round1), total=min(len(data), len(round1)), desc="Processing items"):
            for key, value in item['requirements'].items():
                temp_data = {}
                temp_data['task_id'] = item['task_id']
                temp_data['requirement'] = key
                temp_data['completion'] = inference(args,multi_round_prompt(base, item,value['requirement']))
                result.append(temp_data)
            save_json(args.output_file,result,False)

    else:
        round1 = read_json(args.base_file,False)
        for item, base in tqdm(zip(data, round1), total=min(len(data), len(round1)), desc="Processing items"):
            based = base['completion']
            prompts = [
                    {'role': 'user', 'content': f"""You are an expert Python programmer, and here is your task: {item['prompt']}"""},
                    {'role': 'assistant', 'content': based}
            ]
            for key in item['multi-turn']:
                temp_data = {}
                temp_data['task_id'] = item['task_id']
                temp_data['requirement'] = key
                prompts.append({'role': 'user', 'content': f"The function should meet the following requirements:{item['requirements'][key]['requirement']}"})
                temp_data['completion'] = inference(model,prompts)
                prompts.append({'role': 'assistant', 'content': f"temp_data['completion']"})
                result.append(temp_data)
            save_json(args.output_file,result,False)



