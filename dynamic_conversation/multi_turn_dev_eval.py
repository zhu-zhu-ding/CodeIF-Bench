from __future__ import annotations

import os
import re
import subprocess
import textwrap
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional

import psutil
import func_timeout
from func_timeout import func_set_timeout
from openai import OpenAI
from tqdm import tqdm

from utils import read_json, save_json

QWEN_KEY = ""
QWEN_URL= ""
GPT_KEY = ""
GPT_URL= ""
DEEP_KEY = ""
DEEP_URL = ""

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def extract_key_error_message(full_output: str, max_length: int = 1024) -> str:
    lines = full_output.strip().splitlines()
    key_lines: List[str] = []
    try:
        fail_start = lines.index(
            "=================================== FAILURES ==================================="
        )
    except ValueError:
        fail_start = -1

    if fail_start != -1:
        for idx in range(fail_start + 1, len(lines)):
            if lines[idx].startswith("=") and "short test summary" in lines[idx].lower():
                break
            key_lines.append(lines[idx])

    summary_lines = [line for line in lines if line.strip().startswith("FAILED ")]
    result_text = (
        "\n".join(key_lines + [""] + summary_lines) if key_lines else "\n".join(summary_lines)
    )
    if len(result_text) > max_length:
        return result_text[: max_length - 3] + "..."
    return result_text


def adjust_indent(code: str, new_indent: int) -> str:
    return textwrap.indent(textwrap.dedent(code), " " * new_indent)


@func_set_timeout(60)
def execution_tests(args, data, test_list: List[str]):
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
                    return {"pass": False, "failed": test_list, "error": "OOM"}
                return_code = process.poll()
                if return_code is not None:
                    output, _ = process.communicate()
                    if return_code != 0:
                        result["pass"] = False
                        result["error"] = extract_key_error_message(output)
                        result["failed"] = test_list
                        return result
                    break
        except Exception as exc:
            result["pass"] = False
            result["error"] = str(exc)
            result["failed"] = test_list
            return result
        finally:
            process.terminate()
            process.wait()
    return result


def setup_evaluation(args, data, completion: str) -> None:
    full_path = os.path.join(args.source_code_root, data["completion_path"])
    tmp_path = os.path.join(os.path.dirname(full_path), "tmp_" + os.path.basename(full_path))
    subprocess.run(["cp", full_path, tmp_path], check=False)
    sos, eos = data["signature_position"][0] - 1, data["body_position"][1]
    with open(full_path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()
    new_lines = lines[:sos] + ["\n", completion, "\n"] + lines[eos:]
    with open(full_path, "w", encoding="utf-8") as fh:
        fh.write("".join(new_lines))


def teardown_evaluation(args, data) -> None:
    full_path = os.path.join(args.source_code_root, data["completion_path"])
    tmp_path = os.path.join(os.path.dirname(full_path), "tmp_" + os.path.basename(full_path))
    subprocess.run(["mv", tmp_path, full_path], check=False)


def test_completions(args, completion_code: str, test_list: List[str], data) -> Dict:
    completion_code = adjust_indent(completion_code, data["indent"])
    setup_evaluation(args, data, completion_code)
    try:
        result = execution_tests(args, data, test_list)
    except func_timeout.exceptions.FunctionTimedOut:
        result = {"pass": False, "failed": test_list, "error": "time out"}
    finally:
        teardown_evaluation(args, data)
    return result


def extract_python(python_code: str) -> str:
    pattern = r"```python\n(.*?)```"
    matches = re.findall(pattern, python_code, re.DOTALL)
    if matches:
        return matches[0]
    return python_code


def remove_blank_lines_with_spaces(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if line.strip())


# ---------------------------------------------------------------------------
# Instruction pool management
# ---------------------------------------------------------------------------

def build_instruction_pool(task_data: Dict) -> List[Dict]:
    pool: List[Dict] = []
    base_instruction = task_data.get("base_prompt") or task_data.get("prompt", "")
    base_tests = task_data.get("tests", [])
    if base_instruction or base_tests:
        pool.append(
            {
                "id": "base",
                "label": "Base Task",
                "instruction": base_instruction,
                "tests": base_tests,
                "active": True,
                "passed": False,
                "ever_passed": False,
                "pending": False,
            }
        )

    requirements = task_data.get("requirements", {})
    for idx, key in enumerate(task_data.get("multi-turn", [])):
        req_info = requirements.get(key, {})
        instruction_text = req_info.get("requirement", "")
        if type(req_info.get("test"))==list:
            tests = req_info.get("test")
        else:
            tests = [req_info.get("test")] or []
        if not instruction_text and not tests:
            continue
        pool.append(
            {
                "id": f"req_{idx}",
                "label": key,
                "instruction": instruction_text,
                "tests": tests,
                "active": True,
                "passed": False,
                "ever_passed": False,
                "pending": False,
            }
        )
    return pool


def evaluate_active_instructions(
    args, code: str, pool: List[Dict], task_data: Dict
) -> Dict[str, Dict]:
    results = {}
    for info in pool:
        if not info["active"]:
            continue
        tests = info.get("tests") or []
        if not tests:
            results[info["id"]] = {"pass": True, "failed": [], "error": None}
            info["passed"] = True
            info["ever_passed"] = True
            continue
        evaluation = test_completions(args, code, tests, task_data)
        results[info["id"]] = evaluation
    return results


def update_instruction_states(pool: List[Dict], evaluations: Dict[str, Dict]) -> None:
    for info in pool:
        if not info["active"]:
            continue
        result = evaluations.get(info["id"])
        if result is None:
            continue
        if result.get("pass"):
            info["passed"] = True
            if not info["ever_passed"]:
                info["ever_passed"] = True
        else:
            info["passed"] = False
        if info["pending"]:
            if result.get("pass"):
                info["pending"] = False
            else:
                info["pending"] = False
                info["active"] = False


# ---------------------------------------------------------------------------
# Model inference and dialogue loop
# ---------------------------------------------------------------------------

def inference(args, messages: List[Dict[str, str]]) -> str:
    if args.model == "QWEN":
        client = OpenAI(
            api_key=QWEN_KEY, base_url=QWEN_URL
        )
    elif args.model == "GPT":
        client = OpenAI(api_key=GPT_KEY, base_url=GPT_URL)
    elif args.model == "DEEP":
        client = OpenAI(api_key=DEEP_KEY, base_url=DEEP_URL)
    else:
        raise ValueError(f"Unsupported model type: {args.model}")

    response = client.chat.completions.create(
        model=args.model_name,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content


def dynamic_test_dialogue(
    args, task_data: Dict, max_turns: Optional[int] = None
) -> Dict:
    instruction_pool = build_instruction_pool(task_data)
    if not instruction_pool:
        return {}

    if max_turns is None:
        max_turns = len(instruction_pool) + 1

    logs = {
        "task_id": task_data.get("namespace") or task_data.get("task_id"),
        "rounds": [],
        "final_code": "",
        "all_tests_passed": False,
        "dialogue": [],
        "instructions": instruction_pool,
    }

    messages: List[Dict[str, str]] = []
    instruction_lines = [
        f"{idx + 1}. {info['instruction']}" for idx, info in enumerate(instruction_pool) if info["instruction"]
    ]
    requirements_block = "\n".join(instruction_lines)
    base_prompt = task_data.get("prompt", "") + "\nPlease output the function directly!"
    messages.append({"role": "user", "content": base_prompt})
    completion = inference(args, messages)
    messages.append({"role": "assistant", "content": completion})
    logs["dialogue"] = messages.copy()

    current_code = remove_blank_lines_with_spaces(extract_python(completion))

    round_idx = 0
    current_focus_id = instruction_pool[0]["id"] if instruction_pool else None

    while round_idx < max_turns:
        evaluations = evaluate_active_instructions(args, current_code, instruction_pool, task_data)

        previously_followed = [
            item for item in instruction_pool if item["active"] and item["ever_passed"]
        ]
        forgetting_candidates = len(previously_followed)
        forgetting_failures = sum(
            1
            for item in previously_followed
            if not evaluations.get(item["id"], {}).get("pass")
        )
        ifr_value = (
            forgetting_failures / forgetting_candidates if forgetting_candidates else 0.0
        )

        update_instruction_states(instruction_pool, evaluations)

        active_remaining = [info for info in instruction_pool if info["active"]]
        passed_instruction_count = sum(1 for info in instruction_pool if info["passed"])
        total_instructions = len(instruction_pool)

        if current_focus_id:
            focus_result = evaluations.get(current_focus_id)
            if focus_result is None:
                focus_pass = 1 if not active_remaining else 0
            else:
                focus_pass = 1 if focus_result.get("pass") else 0
        else:
            focus_pass = 1 if not active_remaining else 0

        ca_value = (
            passed_instruction_count / total_instructions if total_instructions else 0.0
        )
        rounds_completed = round_idx + 1
        ife_value = passed_instruction_count / rounds_completed

        logs["rounds"].append(
            {
                "round": round_idx + 1,
                "code": current_code,
                "metrics": {
                    "IA": focus_pass,
                    "CA": ca_value,
                    "IFR": ifr_value,
                    "IFE": ife_value,
                    "passed_instruction_count": passed_instruction_count,
                },
                "results": evaluations,
            }
        )
        logs["final_code"] = current_code

        active_failures = [info for info in active_remaining if not info["passed"]]
        if not active_remaining:
            break
        if not active_failures:
            logs["all_tests_passed"] = True
            break

        targeted = active_failures[0]
        targeted["pending"] = True
        failure_detail = evaluations[targeted["id"]]
        # feedback_prompt = build_feedback_prompt(targeted["instruction"], failure_detail)
        feedback_prompt = build_feedback_prompt(targeted["instruction"], failure_detail) + "\nPlease output the function directly!"
        messages.append({"role": "user", "content": feedback_prompt})

        completion = inference(args, messages)
        messages.append({"role": "assistant", "content": completion})
        logs["dialogue"] = messages.copy()

        current_code = remove_blank_lines_with_spaces(extract_python(completion))
        current_focus_id = targeted["id"]

        round_idx += 1

    if logs["rounds"]:
        per_round_metrics = [
            {
                "round": entry["round"],
                "IA": entry["metrics"]["IA"],
                "CA": entry["metrics"]["CA"],
                "IFR": entry["metrics"]["IFR"],
                "IFE": entry["metrics"]["IFE"],
                "passed_instruction_count": entry["metrics"]["passed_instruction_count"],
            }
            for entry in logs["rounds"]
        ]
        rounds_count = len(per_round_metrics)
        avg_metrics = {
            "IA": sum(item["IA"] for item in per_round_metrics) / rounds_count,
            "CA": sum(item["CA"] for item in per_round_metrics) / rounds_count,
            "IFR": sum(item["IFR"] for item in per_round_metrics) / rounds_count,
            "IFE": sum(item["IFE"] for item in per_round_metrics) / rounds_count,
            "passed_instruction_count": per_round_metrics[-1]["passed_instruction_count"],
        }
        logs["metrics_summary"] = {"per_round": per_round_metrics, "average": avg_metrics}
    else:
        logs["metrics_summary"] = {
            "per_round": [],
            "average": {"IA": 0.0, "CA": 0.0, "IFR": 0.0, "IFE": 0.0},
        }

    return logs


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def build_feedback_prompt(instruction_text: str, failure_detail: Dict) -> str:
    lines = [
        "The current implementation did not satisfy the following requirement:",
        f"\"{instruction_text}\"",
        "",
    ]
    if failure_detail.get("failed"):
        failing_test = failure_detail["failed"][0]
        if isinstance(failing_test, dict):
            test_text = failing_test.get("test", "")
            output_text = failing_test.get("output")
        else:
            test_text = str(failing_test)
            output_text = failure_detail.get("error")
        lines.extend(
            [
                "Here is one failing test:",
                "```python",
                test_text.strip(),
                "```",
            ]
        )
        if output_text:
            lines.extend(
                [
                    "",
                    "Observed output:",
                    "```",
                    output_text,
                    "```",
                ]
            )
    elif failure_detail.get("error"):
        lines.extend(
            [
                "The execution error was:",
                "```",
                failure_detail["error"],
                "```",
            ]
        )
    lines.append("")
    lines.append(
        "Please update the code to address this issue while keeping previous requirements satisfied."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--output_file", type=Path)
    parser.add_argument("--data_file", type=Path)
    parser.add_argument(
        "--source_code_root",
        type=Path
    )
    parser.add_argument("--model", type=str, default="GPT")
    parser.add_argument("--model_name", type=str, default="GPT")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_parser()
    data = read_json(args.data_file, False)
    data = data[:1]
    results = []
    for item in tqdm(data):
        dialogue_log = dynamic_test_dialogue(args, item)
        results.append(dialogue_log)
        save_json(args.output_file, results, False)
