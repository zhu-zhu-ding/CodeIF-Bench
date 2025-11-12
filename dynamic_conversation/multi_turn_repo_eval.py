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
import textwrap
import psutil
from func_timeout import func_set_timeout
import func_timeout
QWEN_KEY = ""
GPT_KEY = ""
DEEP_KEY = ""
import re

def extract_key_error_message(full_output: str, max_length: int = 1024) -> str:
    lines = full_output.strip().splitlines()
    key_lines = []

    try:
        fail_start = lines.index("=================================== FAILURES ===================================")
    except ValueError:
        fail_start = -1

    if fail_start != -1:
        for i in range(fail_start + 1, len(lines)):
            if lines[i].startswith("=") and "short test summary" in lines[i].lower():
                break
            key_lines.append(lines[i])
    summary_lines = [line for line in lines if line.strip().startswith("FAILED ")]

    result_text = "\n".join(key_lines + [""] + summary_lines) if key_lines else "\n".join(summary_lines)

    if len(result_text) > max_length:
        return result_text[:max_length - 3] + "..."
    return result_text

# ------------------------- Repo test tool ------------------------- 
def adjust_indent(code, new_indent):
    dedented = textwrap.dedent(code)
    return textwrap.indent(dedented, ' ' * new_indent)

@func_set_timeout(60)
def execution_tests(args, data, test_list):
    result = {"pass": True, "failed": [], "error": None}
    project_path = os.path.join(args.source_code_root, data['project_path'])
    command = ['python', 'setup.py', 'pytest', '--addopts']
    for test in test_list:
        process = subprocess.Popen(command + [test], cwd=project_path, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            while True:
                if psutil.Process(process.pid).memory_info().rss > 5 * 1024**3:
                    process.terminate(); process.wait()
                    return 'OOM'
                return_code = process.poll()
                if return_code is not None:
                    if return_code != 0:
                        _, stderr = process.communicate()
                        result["pass"] = False
                        result["error"] = extract_key_error_message(_)
                        result["failed"] = test_list
                        return result
                    else:
                        break
        except Exception as e:
            result["pass"] = False
            result["error"] =  str(e)
            result["failed"] = test_list  # Simplified fallback
            return result
        finally:
            process.terminate(); process.wait()
    return result

def SetUp_evaluation(args, data, completion):
    full_path = os.path.join(args.source_code_root, data['completion_path'])
    tmp_path = os.path.join(os.path.dirname(full_path), 'tmp_' + os.path.basename(full_path))
    subprocess.run(['cp', full_path, tmp_path])
    sos, eos = data['signature_position'][0]-1, data['body_position'][1]
    with open(full_path, 'r') as f:
        lines = f.readlines()
    new_lines = lines[:sos] + ['\n', completion, '\n'] + lines[eos:]
    with open(full_path, 'w') as f:
        f.write(''.join(new_lines))

def TearDown_evaluation(args, data):
    full_path = os.path.join(args.source_code_root, data['completion_path'])
    tmp_path = os.path.join(os.path.dirname(full_path), 'tmp_' + os.path.basename(full_path))
    subprocess.run(['mv', tmp_path, full_path]) 

def test_completions(args, completion_code, test_list, data) -> Dict:

    completion_code = adjust_indent(completion_code, data['indent'])
    SetUp_evaluation(args, data, completion_code)
    try:
        flag = execution_tests(args, data, test_list)
    except func_timeout.exceptions.FunctionTimedOut:
        flag = {"pass": False, "failed": test_list, "error": "time out"}
    TearDown_evaluation(args, data)
    return flag


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
    return msg

def extract_python(python_code):
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
        "task_id": task_data["namespace"],
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
            requirements.append({"type": k, "instruction": v["requirement"], "unit_test": v["test"], "passed" : False})
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
    pre_requirements.append({"type":"Base", "instruction": task_data["base_prompt"], "unit_test": task_data["tests"], "passed" : False})

    current_code = extract_python(code_gen)
    current_code = remove_blank_lines_with_spaces(current_code)

    # Base test
    base_test_result = test_completions(args, current_code, task_data["tests"],task_data )
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
        requirements = [{"type":"Base", "instruction": task_data["base_prompt"], "unit_test": task_data["tests"], "passed" : True}] + requirements
    else:
        # One retry using feedback
        feedback_prompt = build_feedback_prompt(task_data["base_prompt"], base_test_result["error"])
        messages.append({"role": "user", "content": feedback_prompt})
        code_gen = inference(args, messages)
        messages.append({"role": "assistant", "content": code_gen})
        logs["dialogue"] = messages.copy()

        current_code = extract_python(code_gen)
        current_code = remove_blank_lines_with_spaces(current_code)

        retry_result = test_completions(args, current_code, task_data["tests"], task_data)
        logs["turns"].append({
            "turn": total_turn,
            "code": current_code,
            "results": [{
                "instruction_type": "Base",
                "instruction": task_data["base_prompt"],
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
            requirements = [{"type":"Base", "instruction": task_data["base_prompt"], "unit_test": task_data["tests"], "passed" : True}] + requirements
    
    while total_turn < max_turns:

        # select instruction
        current_req = None

        for req in requirements:
            select_test_result = test_completions(args,current_code, req["unit_test"],task_data)
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

        current_code = extract_python(code_gen)
        current_code = remove_blank_lines_with_spaces(current_code)


        ## check CA
        CA = 0
        for req in pre_requirements:
            ca_test_result = test_completions(args,current_code, req["unit_test"],task_data)
            if ca_test_result["pass"] == True:
                CA+=1
        ## check IFR
        IFR = 0
        passed_num = 0
        for req in requirements:
            if req['passed']:
                passed_num+=1
                IFR_test_result = test_completions(args,current_code, req["unit_test"],task_data)
                if IFR_test_result["pass"] == False:
                    IFR+=1
                
        test_result = test_completions(args,current_code, current_req["unit_test"],task_data)

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
            "IFR": IFR/passed_num if passed_num >0 else 0,
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

            retry_result = test_completions(args,current_code, current_req["unit_test"],task_data)
            
            ## check CA
            CA = 0
            for req in pre_requirements:
                ca_test_result = test_completions(args,current_code, req["unit_test"],task_data)
                if ca_test_result["pass"] == True:
                    CA+=1
            ## check IFR
            IFR = 0
            passed_num = 0
            for req in requirements:
                if req['passed']:
                    passed_num+=1
                    IFR_test_result = test_completions(args,current_code, req["unit_test"],task_data)
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

    followed_instruction = 0
    for req in requirements:
        select_test_result = test_completions(args,current_code, req["unit_test"],task_data)
        if select_test_result["pass"]:
            followed_instruction+=1
    logs['followed_instruction'] = followed_instruction
    return logs




def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--output_file', type=Path)
    parser.add_argument('--data_file', type=Path)
    parser.add_argument('--source_code_root', type=Path, default=Path('/home/liminxiao//DEV/DevEval/Source_Code'))

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
        model="claude-3-5-sonnet-20241022",
        # model = "deepseek-chat",
        # model = "qwen-coder-turbo",
        # model = "qwen2.5-7b-instruct",
        # model="gpt-4o",
        # model="gpt-4-turbo",
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content


if __name__ == '__main__':
    args = get_parser()
    data = read_json(args.data_file,False)
    # data = data[0:3]
    result = []

    for item in tqdm(data):
        # temp_data = {}
        # temp_data['task_id'] = item['task_id']
        # temp_data['completion'] = inference(args,zero_shot_prompt(item))
        result.append(dynamic_test_dialogue(args,item))
        save_json(args.output_file,result,False)




