from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import (
    read_json,
    save_json
)
from argparse import ArgumentParser
from openai import OpenAI
from tqdm import tqdm
import subprocess
import tempfile
import os
import json
import inspect
from typing import List, Dict
import re
QWEN_KEY = ""
GPT_KEY = ""
DEEP_KEY = ""



def test_completions(completion_code: str, test_lines: List[str]) -> Dict:
    """Run tests against generated code. Return result dictionary."""
    temp_file_path = None
    result = {"pass": True, "failed": [], "error": None}
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as temp_file:
            temp_file.write(completion_code + "\n\n")
            for test in test_lines:
                temp_file.write(test + "\n")
            temp_file_path = temp_file.name

        proc = subprocess.run(
            ["python", temp_file_path],
            capture_output=True,
            text=True,
            check=False,
            timeout=10
        )
        if proc.returncode != 0:
            result["pass"] = False
            result["error"] = proc.stderr
            result["failed"] = test_lines  # Simplified fallback
    except Exception as e:
        result["pass"] = False
        result["error"] = str(e)
        result["failed"] = test_lines
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
    return result


def build_initial_prompt(requirements: List[Dict]) -> str:
    instruction_block = "\n".join([f"{idx+1}. {item['instruction']}" for idx, item in enumerate(requirements)])
    return (
        "You are an expert Python developer. Please complete the following task with all requirements:\n"
        + instruction_block +
        "\nWrite the function below."
    )

def build_feedback_prompt(instruction_text: str, error_msg: str = None) -> str:
    msg = f"Please note that you have not followed the instructions below:\n\"{instruction_text}\"\n\n"
    if error_msg:
        msg += f"Error message:\n```\n{error_msg.strip()}\n```\n\n"
    # msg += "Please revise your code accordingly, while keeping previous requirements satisfied."
    return msg

def extract_python(python_code):
        # pattern = r"```python\r?\n(.*?)```"
        pattern = r"```python\n(.*?)```"
        matches = re.findall(pattern, python_code, re.DOTALL)
        if matches:
            extracted_code = matches[0]
            return extracted_code
        else:
            return python_code

def remove_blank_lines_with_spaces(text):
    """
    Remove all blank lines (lines that are empty or contain only spaces) from a given string.
    
    Args:
        text (str): The input string.
        
    Returns:
        str: The string with blank lines removed.
    """
    return '\n'.join(line for line in text.splitlines() if line.strip())

def dynamic_test_dialogue(args, task_data: Dict, max_turns: int = None) -> Dict:
    logs = {
        "task_id": task_data["task_id"],
        "turns": [],
        "final_code": "",
        "all_tests_passed": False,
        "dialogue": [],
        "passed_requirements": [],
        "ca": 0.0,
        "ife": 0.0,
    }

    raw_requirements = task_data["requirements"]
    ordered_keys = task_data["multi-turn"]
    requirements = []
    pre_requirements = []

    for k in ordered_keys:
        if k!="Functionality Extension":
            v = raw_requirements[k]
            requirements.append({"type": k, "instruction": v["requirement"], "unit_test": v["unit_test"], "passed" : False})
    # print(requirements)
    # assert 1==2
    if max_turns is None:
        max_turns = len(requirements) * 2 + 2

    messages = []
    passed = set()
    total_turn = 1
    finished_flags = [False] * len(requirements)
    current_code = ""
    total_instruction = 0

    # First turn: only prompt
    messages.append({"role": "user", "content": task_data["prompt"]})
    code_gen = inference(args, messages)
    messages.append({"role": "assistant", "content": code_gen})
    logs["dialogue"] = messages.copy()
    total_instruction +=1
    pre_requirements.append({"type":"Base", "instruction": task_data["prompt"], "unit_test": task_data["test"], "passed" : False})

    current_code = extract_python(code_gen)
    current_code = remove_blank_lines_with_spaces(current_code)

    # Base test
    base_test_result = test_completions(current_code, task_data["test"])
    logs["turns"].append({
        "turn": total_turn,
        "code": current_code,
        "results": [{
            "instruction_type": "Base",
            "instruction": task_data["prompt"],
            "pass": base_test_result["pass"],
            "failed_tests": base_test_result["failed"],
            "error": base_test_result["error"]
        }],
        "IA": 1 if base_test_result["pass"] else 0,
        "IFR": 0,
        "CA": 1 if base_test_result["pass"] else 0,
        "IFE": 1 if base_test_result["pass"] else 0
    })
    total_turn += 1

    if base_test_result["pass"]:
        pre_requirements[0]['passed'] = True
        requirements = [{"type":"Base", "instruction": task_data["prompt"], "unit_test": task_data["test"], "passed" : True}] + requirements
    else:
        # One retry using feedback
        feedback_prompt = build_feedback_prompt(task_data["prompt"], base_test_result["error"])
        messages.append({"role": "user", "content": feedback_prompt})
        code_gen = inference(args, messages)
        messages.append({"role": "assistant", "content": code_gen})
        logs["dialogue"] = messages.copy()

        current_code = extract_python(code_gen)
        current_code = remove_blank_lines_with_spaces(current_code)

        retry_result = test_completions(current_code, task_data["test"])
        logs["turns"].append({
            "turn": total_turn,
            "code": current_code,
            "results": [{
                "instruction_type": "Base",
                "instruction": task_data["prompt"],
                "pass": retry_result["pass"],
                "failed_tests": retry_result["failed"],
                "error": retry_result["error"]
            }],
            "IA": 1 if retry_result["pass"] else 0,
            "IFR":0,
            "CA": 1 if retry_result["pass"] else 0,
            "IFE": 0.5 if retry_result["pass"] else 0
        })
        total_turn += 1
        if retry_result["pass"]:
            pre_requirements[0]['passed'] = True
            requirements = [{"type":"Base", "instruction": task_data["prompt"], "unit_test": task_data["test"], "passed" : True}] + requirements
    
    while total_turn <= max_turns:

        # select instruction
        current_req = None

        for req in requirements:
            select_test_result = test_completions(current_code, req["unit_test"])
            if select_test_result["pass"] == False:
                current_req = req
                break
        if current_req == None:
            break
        # inference instruction
        messages.append({"role": "user", "content": current_req["instruction"]})
        code_gen = inference(args, messages)
        messages.append({"role": "assistant", "content": code_gen})
        logs["dialogue"] = messages.copy()
        pre_requirements.append(current_req)
        total_instruction +=1

        current_code = extract_python(code_gen)
        current_code = remove_blank_lines_with_spaces(current_code)


        ## check CA
        CA = 0
        for req in pre_requirements:
            ca_test_result = test_completions(current_code, req["unit_test"])
            if ca_test_result["pass"] == True:
                CA+=1

        ## check IFR
        IFR = 0
        passed_num = 0
        for req in requirements:
            if req['passed']:
                passed_num+=1
                IFR_test_result = test_completions(current_code, req["unit_test"])
                if IFR_test_result["pass"] == False:
                    IFR+=1
                
        test_result = test_completions(current_code, current_req["unit_test"])

        logs["turns"].append({
            "turn": total_turn,
            "code":current_code,
            "results": [{
                "instruction_type": current_req["type"],
                "instruction": current_req["instruction"],
                "pass": test_result["pass"],
                "failed_tests": test_result["failed"],
                "error": test_result["error"]
            }],
            "IA": 1 if test_result["pass"] else 0,
            "IFR": IFR/passed_num if passed_num >0 else 0, ##TODO
            "CA": CA/len(pre_requirements),
            "IFE": CA/total_turn,
        })
        total_turn += 1

        if total_turn >= max_turns:
            break

        if test_result["pass"]:
            current_req['passed'] = True
        else:
            # One retry using feedback
            feedback_prompt = build_feedback_prompt(current_req["instruction"], test_result["error"])
            messages.append({"role": "user", "content": feedback_prompt})
            code_gen = inference(args, messages)
            messages.append({"role": "assistant", "content": code_gen})
            logs["dialogue"] = messages.copy()

            current_code = extract_python(code_gen)
            current_code = remove_blank_lines_with_spaces(current_code)

            retry_result = test_completions(current_code, current_req["unit_test"])
            
            ## check CA
            CA = 0
            for req in pre_requirements:
                ca_test_result = test_completions(current_code, req["unit_test"])
                if ca_test_result["pass"] == True:
                    CA+=1
            ## check IFR
            IFR = 0
            passed_num = 0
            for req in requirements:
                if req['passed']:
                    passed_num+=1
                    IFR_test_result = test_completions(current_code, req["unit_test"])
                    if IFR_test_result["pass"] == False:
                        IFR+=1
            logs["turns"].append({
                "turn": total_turn,
                "code":current_code,
                "results": [{
                    "instruction_type": current_req["type"],
                    "instruction": current_req["instruction"],
                    "pass": retry_result["pass"],
                    "failed_tests": retry_result["failed"],
                    "error": retry_result["error"]
                }],
                "IA": 1 if retry_result["pass"] else 0,
                "IFR": IFR/passed_num if passed_num >0 else 0,
                "CA": CA/len(pre_requirements),
                "IFE": CA/total_turn,
            })
            total_turn += 1
            if retry_result["pass"]:
                current_req['passed'] = True
                continue
            else:
                requirements = [item for item in requirements if item['instruction'] != current_req['instruction']]

    logs['followed_instruction'] = len(requirements)
    return logs




def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--output_file', type=Path)
    parser.add_argument('--data_file', type=Path)
    # parser.add_argument('--base_file', type=Path, default=None)
    parser.add_argument('--model', type=str, default='GPT')
    return parser.parse_args()

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
        model = "qwen-coder-turbo",
        # model = "qwen2.5-32b-instruct",
        # model="gpt-4o",
        # model="gpt-4-turbo",
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content


if __name__ == '__main__':
    args = get_parser()
    data = read_json(args.data_file,False)
    # data = data[:3]
    result = []

    for item in tqdm(data):
        # temp_data = {}
        # temp_data['task_id'] = item['task_id']
        # temp_data['completion'] = inference(args,zero_shot_prompt(item))
        result.append(dynamic_test_dialogue(args,item))
        save_json(args.output_file,result,False)




