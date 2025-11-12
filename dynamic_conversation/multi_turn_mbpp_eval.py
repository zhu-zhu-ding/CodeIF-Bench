from pathlib import Path
import subprocess
import tempfile
import os
import sys
import re
from typing import Dict, List, Optional
from argparse import ArgumentParser

from tqdm import tqdm
from openai import OpenAI

from utils import read_json, save_json

QWEN_KEY = ""
QWEN_URL= ""
GPT_KEY = ""
GPT_URL= ""
DEEP_KEY = ""
DEEP_URL = ""


def extract_python(python_code: str) -> str:
    pattern = r"```python\n(.*?)```"
    matches = re.findall(pattern, python_code, re.DOTALL)
    if matches:
        return matches[0]
    return python_code


def remove_blank_lines_with_spaces(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if line.strip())


def run_python_snippet(code: str, test_snippet: str, timeout: int = 5) -> Dict[str, Optional[str]]:
    """Execute code + test snippet in a temporary file."""
    script_body = f"{code.rstrip()}\n\n{test_snippet.rstrip()}\n"
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tmp:
            tmp.write(script_body)
            temp_path = tmp.name
        proc = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        passed = proc.returncode == 0
        output = (proc.stdout or "") + (proc.stderr or "")
        return {"pass": passed, "output": output.strip()}
    except subprocess.TimeoutExpired:
        return {"pass": False, "output": "timeout"}
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def test_completions(code: str, tests: List[str]) -> Dict:
    """Run each unit test independently to capture individual failures."""
    failures = []
    first_error_output = None
    for snippet in tests:
        result = run_python_snippet(code, snippet)
        if not result["pass"]:
            failure_record = {"test": snippet, "output": result["output"]}
            failures.append(failure_record)
            if first_error_output is None:
                first_error_output = result["output"]
    return {
        "pass": len(failures) == 0,
        "failed": failures,
        "error": first_error_output,
    }


def build_instruction_pool(task_data: Dict) -> List[Dict]:
    """Create a list of instruction/test bundles."""
    pool = []
    pool.append(
        {
            "id": "base",
            "label": "Base Task",
            "instruction": task_data["prompt"],
            "tests": task_data.get("test", []),
            "active": True,
            "passed": False,
            "ever_passed": False,
            "pending": False,
        }
    )
    requirements = task_data.get("requirements", {})
    for idx, key in enumerate(task_data.get("multi-turn", [])):
        req = requirements.get(key, {})
        tests = req.get("unit_test", [])
        if not tests:
            continue
        pool.append(
            {
                "id": f"req_{idx}",
                "label": key,
                "instruction": req.get("requirement", ""),
                "tests": tests,
                "active": True,
                "passed": False,
                "ever_passed": False,
                "pending": False,
            }
        )
    return pool


def build_feedback_prompt(instruction_text: str, failure_detail: Dict) -> str:
    lines = [
        "The current implementation did not satisfy the following requirement:",
        f"\"{instruction_text}\"",
        "",
    ]
    if failure_detail["failed"]:
        failing_test = failure_detail["failed"][0]
        lines.extend(
            [
                "Here is one failing test:",
                "```python",
                failing_test["test"].strip(),
                "```",
            ]
        )
        if failing_test["output"]:
            lines.extend(
                [
                    "",
                    "Observed output:",
                    "```",
                    failing_test["output"],
                    "```",
                ]
            )
    elif failure_detail["error"]:
        lines.extend(
            [
                "The execution error was:",
                "```",
                failure_detail["error"],
                "```",
            ]
        )
    lines.append("")
    lines.append("Please update the code to address this issue while keeping previous requirements satisfied.")
    return "\n".join(lines)


def evaluate_active_instructions(code: str, pool: List[Dict]) -> Dict[str, Dict]:
    results = {}
    for info in pool:
        if not info["active"]:
            continue
        evaluation = test_completions(code, info["tests"])
        results[info["id"]] = evaluation
    return results


def update_instruction_states(pool: List[Dict], evaluations: Dict[str, Dict]) -> None:
    for info in pool:
        if not info["active"]:
            continue
        eval_result = evaluations.get(info["id"])
        if eval_result is None:
            continue
        if eval_result["pass"]:
            info["passed"] = True
            if not info["ever_passed"]:
                info["ever_passed"] = True
        else:
            info["passed"] = False
        if info["pending"]:
            if eval_result["pass"]:
                info["pending"] = False
            else:
                info["pending"] = False
                info["active"] = False


def inference(args, messages):
    if args.model == "QWEN":
        client = OpenAI(
            api_key=QWEN_KEY,
            base_url=QWEN_URL,
        )
    elif args.model == "GPT":
        client = OpenAI(
            api_key=GPT_KEY,
            base_url=GPT_URL,
        )
    elif args.model == "DEEP":
        client = OpenAI(
            api_key=DEEP_KEY,
            base_url=DEEP_URL,
        )
    else:
        raise ValueError(f"Unsupported model type: {args.model}")

    response = client.chat.completions.create(
        model = args.model_name,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content


def dynamic_test_dialogue(args, task_data: Dict, max_turns: Optional[int] = None) -> Dict:
    instruction_pool = build_instruction_pool(task_data)
    if not instruction_pool:
        return {}

    if max_turns is None:
        max_turns = len(instruction_pool) + 1

    logs = {
        "task_id": task_data["task_id"],
        "rounds": [],
        "final_code": "",
        "all_tests_passed": False,
        "dialogue": [],
        "instructions": instruction_pool,
    }

    messages = []
    instruction_lines = [
        f"{idx + 1}. {info['instruction']}" for idx, info in enumerate(instruction_pool)
    ]
    requirements_block = "\n".join(instruction_lines)
    base_prompt = task_data["prompt"] + "\nDo not output test cases. Please output the function directly."
    messages.append({"role": "user", "content": base_prompt})
    completion = inference(args, messages)
    messages.append({"role": "assistant", "content": completion})
    logs["dialogue"] = messages.copy()

    current_code = remove_blank_lines_with_spaces(extract_python(completion))

    round_idx = 0
    current_focus_id = instruction_pool[0]["id"] if instruction_pool else None
    while round_idx < max_turns:
        evaluations = evaluate_active_instructions(current_code, instruction_pool)

        previously_followed = [
            item for item in instruction_pool if item["active"] and item["ever_passed"]
        ]
        forgetting_candidates = len(previously_followed)
        forgetting_failures = sum(
            1
            for item in previously_followed
            if not evaluations.get(item["id"], {}).get("pass")
        )
        if forgetting_candidates:
            ifr_value = forgetting_failures / forgetting_candidates
        else:
            ifr_value = 0.0

        update_instruction_states(instruction_pool, evaluations)

        active_remaining = [info for info in instruction_pool if info["active"]]
        passed_instruction_count = sum(1 for info in instruction_pool if info["passed"])
        total_instructions = len(instruction_pool)
        ever_passed_count = sum(1 for info in instruction_pool if info["ever_passed"])

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
        if rounds_completed > 0:
            ife_value = passed_instruction_count / rounds_completed
        else:
            ife_value = 0.0

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
        feedback_prompt = build_feedback_prompt(targeted["instruction"], failure_detail)
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
        logs["metrics_summary"] = {
            "per_round": per_round_metrics,
            "average": avg_metrics,
        }
    else:
        logs["metrics_summary"] = {
            "per_round": [],
            "average": {"IA": 0.0, "CA": 0.0, "IFR": 0.0, "IFE": 0.0},
        }

    return logs


def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--output_file", type=Path)
    parser.add_argument("--data_file", type=Path)
    parser.add_argument("--model_name", type=str, default="GPT")
    parser.add_argument("--model", type=str, default="GPT")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_parser()
    data = read_json(args.data_file, False)
    data = data[:1]
    result = []
    for item in tqdm(data):
        dialogue_log = dynamic_test_dialogue(args, item)
        result.append(dialogue_log)
        save_json(args.output_file, result, False)
