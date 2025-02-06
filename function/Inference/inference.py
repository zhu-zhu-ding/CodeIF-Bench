import os
from pathlib import Path
import torch
from peft import PeftModel, LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import (
    read_json,
    save_json
)
from argparse import ArgumentParser
from tqdm import tqdm
import json

def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--output_file', type=Path)
    parser.add_argument('--data_file', type=Path)
    parser.add_argument('--base_file', type=Path, default=None)
    parser.add_argument('--lora_path', type=Path, default=None)
    parser.add_argument('--model_path', type=Path)
    parser.add_argument('--way', type=int)
    return parser.parse_args()

def zero_shot_prompt(data):
    prompt = f"""You are an expert Python programmer, and here is your task: {data['prompt']}"""
    messages=[
        {'role': 'user', 'content': prompt}
    ]
    return messages

def single_round_prompt(data, requirement):
    prompt = f"""You are an expert Python programmer, and here is your task: {data['prompt']}. The function should meet the following requirements: {requirement}"""
    messages=[
        {'role': 'user', 'content': prompt}
    ]
    return messages

def multi_round_prompt(base, data, requirement):
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

def inference(model, messages, max_context_length=16384, max_input_length=8192, max_new_tokens=1024):

    tokenizer.model_max_length = max_context_length

    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

    if inputs.shape[1] > max_input_length:
        inputs = inputs[:, -max_input_length:]
    print(inputs.shape[1])
    outputs = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        max_length=min(max_context_length, max_input_length + max_new_tokens),
        temperature=0.2,
        do_sample=False,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
    )

    return tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)

if __name__ == '__main__':
    args = get_parser()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto",trust_remote_code=True, torch_dtype=torch.bfloat16)

    if args.lora_path:
        model = PeftModel.from_pretrained(model, args.lora_path)
        model = model.merge_and_unload()

    data = read_json(args.data_file,True)
    result = []

    if args.way == 1:
        for item in tqdm(data):
            temp_data = {}
            temp_data['task_id'] = item['task_id']
            temp_data['completion'] = inference(model,zero_shot_prompt(item))
            result.append(temp_data)
        save_json(args.output_file,result)

    elif args.way == 2:
        for item in tqdm(data):
            for key, value in item['requirements'].items():
                temp_data = {}
                temp_data['task_id'] = item['task_id']
                temp_data['requirement'] = key
                temp_data['completion'] = inference(model,single_round_prompt(item,value['requirement']))
                result.append(temp_data)
            save_json(args.output_file,result,True)

    elif args.way == 3:
        round1 = read_json(args.base_file,False)
        for item, base in tqdm(zip(data, round1), total=min(len(data), len(round1)), desc="Processing items"):
            for key, value in item['requirements'].items():
                temp_data = {}
                temp_data['task_id'] = item['task_id']
                temp_data['requirement'] = key
                temp_data['completion'] = inference(model,multi_round_prompt(base, item,value['requirement']))
                result.append(temp_data)
            save_json(args.output_file,result,False)

    else:
        round1 = read_json(args.base_file,False)
        for item, base in tqdm(zip(data, round1), total=min(len(data), len(round1)), desc="Processing items"):
            flag = True
            based = base['completion']
            prompts = f"""You are an expert Python programmer, and here is your task: {item['prompt']}"""
            for key in item['multi-turn']:
                temp_data = {}
                temp_data['task_id'] = item['task_id']
                temp_data['requirement'] = key
                temp_data['completion'] = inference(model,n_round_prompt(based, prompts, item['requirements'][key]['requirement']))
                result.append(temp_data)
                print(n_round_prompt(based, prompts, item['requirements'][key]['requirement']))
                if flag:
                    prompts += "The function should meet the following requirements: "
                    flag = False
                temp_data['completion']
                prompts += item['requirements'][key]['requirement']
            save_json(args.output_file,result,False)