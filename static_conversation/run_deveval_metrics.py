from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Dict
import argparse
import func_timeout
import json
import os
import psutil
import re
import subprocess
import textwrap
from func_timeout import func_set_timeout
from tqdm import tqdm

from utils import read_json


def remove_blank_lines(code: str) -> str:
    lines = code.splitlines()
    return "\n".join(line for line in lines if line.strip())

def safe_indent(text: str, prefix: str, predicate=None) -> str:
    if predicate is None:
        predicate = lambda _line: True
    result = []
    for line in text.splitlines(keepends=True):
        if predicate(line):
            result.append(prefix + line)
        else:
            result.append(line)
    return "".join(result)


def safe_dedent(text: str) -> str:
    lines = text.splitlines(keepends=True)
    leading_spaces = []
    for line in lines:
        if line.strip() == "":
            continue
        stripped = line.lstrip(" ")
        leading_spaces.append(len(line) - len(stripped))
    trim = min(leading_spaces) if leading_spaces else 0
    if trim <= 0:
        return text
    trimmed_lines = []
    prefix = " " * trim
    for line in lines:
        if line.startswith(prefix):
            trimmed_lines.append(line[trim:])
        else:
            trimmed_lines.append(line)
    return "".join(trimmed_lines)


def adjust_indent(code, new_indent):
    code = remove_blank_lines(code)
    dedented = textwrap.dedent(code)
    return textwrap.indent(dedented, " " * new_indent)


@func_set_timeout(60)
def execution_tests(args, data, test_list):
    result = {"pass": True, "failed": [], "error": None}
    project_path = os.path.join(args.source_code_root, data["project_path"])
    command = ["python", "setup.py", "pytest", "--addopts"]
    for test in test_list:
        process = subprocess.Popen(
            command + [test],
            cwd=project_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            while True:
                if psutil.Process(process.pid).memory_info().rss > 5 * 1024**3:
                    process.terminate()
                    process.wait()
                    return "OOM"
                return_code = process.poll()
                if return_code is not None:
                    if return_code != 0:
                        _, stderr = process.communicate()
                        result["pass"] = False
                        result["error"] = _
                        result["failed"] = test_list
                        return result
                    else:
                        break
        except Exception as e:
            result["pass"] = False
            result["error"] = str(e)
            result["failed"] = test_list
            return result
        finally:
            process.terminate()
            process.wait()
    return result


def SetUp_evaluation(args, data, completion):
    full_path = os.path.join(args.source_code_root, data["completion_path"])
    tmp_path = os.path.join(
        os.path.dirname(full_path), "tmp_" + os.path.basename(full_path)
    )
    subprocess.run(["cp", full_path, tmp_path])
    sos, eos = data["signature_position"][0] - 1, data["body_position"][1]
    with open(full_path, "r") as f:
        lines = f.readlines()
    new_lines = lines[:sos] + ["\n", completion, "\n"] + lines[eos:]
    with open(full_path, "w") as f:
        f.write("".join(new_lines))


def TearDown_evaluation(args, data):
    full_path = os.path.join(args.source_code_root, data["completion_path"])
    tmp_path = os.path.join(
        os.path.dirname(full_path), "tmp_" + os.path.basename(full_path)
    )
    subprocess.run(["mv", tmp_path, full_path])


def test_completions(args, completion_code, test_list, data) -> Dict:
    completion_code = adjust_indent(completion_code, data["indent"])
    SetUp_evaluation(args, data, completion_code)
    try:
        flag = execution_tests(args, data, test_list)
    except func_timeout.exceptions.FunctionTimedOut:
        flag = {"pass": False, "failed": test_list, "error": "time out"}
    TearDown_evaluation(args, data)
    return flag


def extract_python_code_from_content(content, function_name=None):
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, str) and (
                    "def " in value or "import " in value or "class " in value
                ):
                    if function_name and re.search(
                        rf"\bdef\s+{re.escape(function_name)}\s*\(", value
                    ):
                        return value
                    return value
    except json.JSONDecodeError:
        pass

    markdown_pattern = r"```(?:python)?\s*(.*?)```"
    matches = re.findall(markdown_pattern, content, re.DOTALL)
    if matches:
        if function_name:
            for match in matches:
                if re.search(rf"\bdef\s+{re.escape(function_name)}\s*\(", match):
                    return match.strip()
        return matches[0].strip()

    if "def " in content or "import " in content or "class " in content:
        if function_name:
            segments = content.split("```")
            for segment in segments:
                if re.search(rf"\bdef\s+{re.escape(function_name)}\s*\(", segment):
                    return segment.strip()
        return content.strip()

    return content


def get_assistant_contents(session_messages):
    assistant_contents = []
    for message in session_messages:
        if isinstance(message, list):
            message = message[0]
        if message.get("role") == "assistant" and "content" in message:
            assistant_contents.append(message["content"])
    return assistant_contents


def process_session_data(session_messages, function_name=None):
    assistant_contents = get_assistant_contents(session_messages)
    results = []
    for i, content in enumerate(assistant_contents):
        code = extract_python_code_from_content(content, function_name)
        results.append({"index": i, "original_content": content, "extracted_code": code})
    return results


def get_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--source_code_root",
        type=Path,
        default=Path("/home/wpd/workspace/codeif-icse/Source_Code_v3_3"),
    )
    return parser.parse_args()


def build_test_plan(data):
    IA_test = [data["tests"]]
    CA_test = [data["tests"]]
    temp_test = data["tests"]
    for req in data["multi-turn"]:
        IA_test.append([data["requirements"][req]["test"]])
        temp_test = temp_test.copy()
        temp_test.append(data["requirements"][req]["test"])
        CA_test.append(temp_test.copy())
    return IA_test, CA_test


def compute_metrics(args, data_list):
    IA_pass_counts = defaultdict(int)
    IA_total_counts = defaultdict(int)
    CA_ratio_sum = defaultdict(float)
    CA_total_counts = defaultdict(int)
    forgetting_sum = defaultdict(float)
    forgetting_total_counts = defaultdict(int)
    max_round_index = -1

    # function_name_patterns = [
    #     r"function called ['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]",
    #     r"function named ['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]",
    #     r"define (?:a|the) function ['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]",
    # ]

    # def get_function_name(entry):
    #     prompt = entry.get("prompt", "") or ""
    #     for pattern in function_name_patterns:
    #         match = re.search(pattern, prompt)
    #         if match:
    #             return match.group(1)
    #     context = entry.get("context", "") or ""
    #     context_match = re.search(
    #         r"###The function:\s*([A-Za-z_][A-Za-z0-9_]*)###", context
    #     )
    #     if context_match:
    #         return context_match.group(1)
    #     return None

    for data in tqdm(data_list):
        IA_test, CA_test = build_test_plan(data)

        def ensure_list(value):
            if isinstance(value, list):
                return value
            return [value]

        instructions = [{"name": "base", "tests": ensure_list(data["tests"])}]
        for req in data["multi-turn"]:
            instructions.append(
                {"name": req, "tests": ensure_list(data["requirements"][req]["test"])}
            )

        function_name = data['namespace'].split('.')[-1]
        print(function_name)
        results = process_session_data(data["session_messages"], function_name)
        prev_passed = set()

        round_limit = min(len(results), len(CA_test), len(IA_test))
        for round_idx in range(round_limit):
            result = results[round_idx]
            max_round_index = max(max_round_index, round_idx)
            cumulative_tests = CA_test[round_idx]
            total_tests = len(cumulative_tests)
            if total_tests == 0:
                continue

            current_passed = set()
            test_outcome = {}
            passed_count = 0
            forgetting_failures = 0
            prev_passed_count = len(prev_passed)

            for test_name in cumulative_tests:
                temp_result = test_completions(
                    args, result["extracted_code"], [test_name], data
                )
                if isinstance(temp_result, str):
                    temp_flag = False
                else:
                    temp_flag = temp_result["pass"]
                test_outcome[test_name] = temp_flag
                if temp_flag:
                    passed_count += 1
                    current_passed.add(test_name)
                elif test_name in prev_passed:
                    forgetting_failures += 1

            if prev_passed_count:
                forgetting_rate = forgetting_failures / prev_passed_count
            else:
                forgetting_rate = 0.0

            instruction_slice = instructions[: round_idx + 1]
            instruction_total = len(instruction_slice)
            instruction_passed = 0
            for instruction in instruction_slice:
                tests_for_instruction = instruction["tests"]
                if all(
                    test_outcome.get(test_name, False)
                    for test_name in tests_for_instruction
                ):
                    instruction_passed += 1
            ca_ratio = (
                instruction_passed / instruction_total if instruction_total else 0
            )

            IA_total_counts[round_idx] += 1
            IA_result = test_completions(
                args, result["extracted_code"], IA_test[round_idx], data
            )
            if IA_result["pass"]:
                IA_pass_counts[round_idx] += 1

            CA_ratio_sum[round_idx] += ca_ratio
            CA_total_counts[round_idx] += 1
            forgetting_sum[round_idx] += forgetting_rate
            if prev_passed_count:
                forgetting_total_counts[round_idx] += 1
            prev_passed.update(current_passed)

    rounds = list(range(max_round_index + 1)) if max_round_index >= 0 else []

    IA_rates = []
    CA_rates = []
    forgetting_rates = []
    IA_counts = []
    CA_counts = []
    forgetting_counts = []

    for idx in rounds:
        if IA_total_counts[idx]:
            IA_rates.append(IA_pass_counts[idx] / IA_total_counts[idx] * 100)
        else:
            IA_rates.append(0.0)
        IA_counts.append(IA_total_counts[idx])

        if CA_total_counts[idx]:
            CA_rates.append(CA_ratio_sum[idx] / CA_total_counts[idx] * 100)
        else:
            CA_rates.append(0.0)
        CA_counts.append(CA_total_counts[idx])

        if forgetting_total_counts[idx]:
            forgetting_rates.append(forgetting_sum[idx] / forgetting_total_counts[idx] * 100)
        else:
            forgetting_rates.append(0.0)
        forgetting_counts.append(forgetting_total_counts[idx])

    return {
        "rounds": rounds,
        "IA_rates": IA_rates,
        "IA_counts": IA_counts,
        "CA_rates": CA_rates,
        "CA_counts": CA_counts,
        "forgetting_rates": forgetting_rates,
        "forgetting_counts": forgetting_counts,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute IA, CA, and forgetting metrics for JSONL records.")
    parser.add_argument("--input_path", type=Path, help="Path to the JSONL file.")
    parser.add_argument("--source_code_root", type=Path, help="Path to the JSONL file.")
    args = parser.parse_args()

    data_list = read_json(args.input_path, False)
    metrics = compute_metrics(args, data_list)
    rounds = metrics["rounds"]
    IA_rates = metrics["IA_rates"]
    IA_counts = metrics["IA_counts"]
    CA_rates = metrics["CA_rates"]
    CA_counts = metrics["CA_counts"]
    forgetting_rates = metrics["forgetting_rates"]
    forgetting_counts = metrics["forgetting_counts"]

    print("Rounds evaluated:", rounds)
    print("IA per round (%):", IA_rates)
    print("CA per round (%):", CA_rates)
    print("Forgetting rates per round (%):", forgetting_rates)

    ia_valid = [rate for rate, count in zip(IA_rates, IA_counts) if count > 0]
    ca_valid = [rate for rate, count in zip(CA_rates, CA_counts) if count > 0]
    forgetting_valid = [
        rate for rate, count in zip(forgetting_rates, forgetting_counts) if count > 0
    ]

    if ia_valid:
        print("Average IA (%):", sum(ia_valid) / len(ia_valid))
    if ca_valid:
        print("Average CA (%):", sum(ca_valid) / len(ca_valid))
    if forgetting_valid:
        print("Average forgetting rate (%):", sum(forgetting_valid) / len(forgetting_valid))


if __name__ == "__main__":
    main()
