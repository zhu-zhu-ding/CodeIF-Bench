import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from utils import read_json



def extract_python_code_from_content(content: str) -> str:
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            for value in data.values():
                if isinstance(value, str) and (
                    "def " in value or "import " in value or "class " in value
                ):
                    return value
    except json.JSONDecodeError:
        pass

    markdown_pattern = r"```(?:python)?\s*(.*?)```"
    matches = re.findall(markdown_pattern, content, re.DOTALL)
    if matches:
        return matches[0].strip()

    if "def " in content or "import " in content or "class " in content:
        return content.strip()

    return content


def get_assistant_contents(session_messages: List[Dict]) -> List[str]:
    assistant_contents = []
    for message in session_messages:
        if isinstance(message, list):
            message = message[0]
        if message.get("role") == "assistant" and "content" in message:
            assistant_contents.append(message["content"])
    return assistant_contents


def process_session_data(session_messages: List[Dict]) -> List[Dict]:
    contents = get_assistant_contents(session_messages)
    results = []
    for index, content in enumerate(contents):
        code = extract_python_code_from_content(content)
        results.append({"index": index, "original_content": content, "extracted_code": code})
    return results


def run_python_tests(code: str, test_snippet: str, timeout: int = 30) -> Dict[str, Optional[str]]:
    script_content = f"{code.rstrip()}\n\n{test_snippet.rstrip()}\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tmp:
        tmp.write(script_content)
        tmp_path = tmp.name
    try:
        proc = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        passed = proc.returncode == 0
        output = proc.stdout + proc.stderr
        if not passed:
            print(output)
        return {"pass": passed, "output": output}
    except subprocess.TimeoutExpired:
        return {"pass": False, "output": "timeout"}
    finally:
        os.unlink(tmp_path)


def build_test_plan(entry: Dict) -> Dict[str, List[List[str]]]:
    ia_tests: List[List[str]] = []
    ca_tests: List[List[str]] = []

    initial_tests = entry.get("test", [])
    if initial_tests:
        ia_tests.append(initial_tests)
        ca_tests.append(initial_tests.copy())

    cumulative = initial_tests.copy()
    for requirement_name in entry.get("multi-turn", []):
        requirement = entry["requirements"].get(requirement_name, {})
        unit_tests = requirement.get("unit_test", [])
        if not unit_tests:
            continue
        ia_tests.append(unit_tests)
        cumulative += unit_tests
        ca_tests.append(cumulative.copy())

    return {"IA": ia_tests, "CA": ca_tests}


def evaluate_round(code: str, tests: List[str]) -> Dict[str, List[str]]:
    passed = []
    failed = []
    for snippet in tests:
        result = run_python_tests(code, snippet)
        if result["pass"]:
            passed.append(snippet)
        else:
            failed.append(snippet)
    return {"passed": passed, "failed": failed}


def compute_metrics(entries: List[Dict]) -> Dict[str, List[float]]:
    ia_pass_counts = defaultdict(int)
    ia_total_counts = defaultdict(int)
    ca_pass_ratio_sum = defaultdict(float)
    ca_total_counts = defaultdict(int)
    forgetting_rate_sum = defaultdict(float)
    forgetting_total_counts = defaultdict(int)
    max_round_index = -1

    from tqdm import tqdm
    for entry in tqdm(entries):
        tests_plan = build_test_plan(entry)
        ia_tests = tests_plan["IA"]
        ca_tests = tests_plan["CA"]
        session_results = process_session_data(entry.get("session_messages", []))

        prev_passed_tests = set()
        round_count = min(len(session_results), len(ia_tests), len(ca_tests))
        for round_index in range(round_count):
            code = session_results[round_index]["extracted_code"]
            ia_result = evaluate_round(code, ia_tests[round_index])
            ca_result = evaluate_round(code, ca_tests[round_index])

            max_round_index = max(max_round_index, round_index)

            ia_total_counts[round_index] += 1
            if not ia_result["failed"]:
                ia_pass_counts[round_index] += 1

            total_ca_tests = len(ca_tests[round_index])
            ca_pass_ratio = len(ca_result["passed"]) / total_ca_tests if total_ca_tests else 0.0
            ca_pass_ratio_sum[round_index] += ca_pass_ratio
            ca_total_counts[round_index] += 1

            current_passed = set(ca_result["passed"])
            prev_passed_count = len(prev_passed_tests)
            if prev_passed_count > 0:
                forgotten = len(prev_passed_tests - current_passed)
                forgetting_rate_sum[round_index] += forgotten / prev_passed_count
                forgetting_total_counts[round_index] += 1
            prev_passed_tests.update(current_passed)

    rounds = list(range(max_round_index + 1)) if max_round_index >= 0 else []
    ia_rates = []
    ca_rates = []
    forgetting_rates = []

    for idx in rounds:
        ia_rate = (
            ia_pass_counts[idx] / ia_total_counts[idx] * 100 if ia_total_counts[idx] else 0.0
        )
        ca_rate = (
            ca_pass_ratio_sum[idx] / ca_total_counts[idx] * 100 if ca_total_counts[idx] else 0.0
        )
        forgetting_rate = (
            forgetting_rate_sum[idx] / forgetting_total_counts[idx] * 100
            if forgetting_total_counts[idx]
            else 0.0
        )
        ia_rates.append(ia_rate)
        ca_rates.append(ca_rate)
        forgetting_rates.append(forgetting_rate)

    return {"rounds": rounds, "IA": ia_rates, "CA": ca_rates, "forgetting": forgetting_rates}


def main():
    parser = argparse.ArgumentParser(description="Compute IA, CA, and forgetting metrics for JSONL records.")
    parser.add_argument("--input_path", type=Path)
    args = parser.parse_args()

    entries = read_json(args.jsonl_path)
    metrics = compute_metrics(entries)

    rounds = metrics["rounds"]
    ia_rates = metrics["IA"]
    ca_rates = metrics["CA"]
    forgetting_rates = metrics["forgetting"]

    print(f"Rounds evaluated: {rounds}")
    print(f"IA per round (%): {ia_rates}")
    print(f"CA per round (%): {ca_rates}")
    print(f"Forgetting per round (%): {forgetting_rates}")

    if ia_rates:
        print(f"Average IA (%): {sum(ia_rates) / len(ia_rates):.2f}")
    if ca_rates:
        print(f"Average CA (%): {sum(ca_rates) / len(ca_rates):.2f}")
    valid_forgetting = [rate for rate in forgetting_rates if rate > 0 or rate == 0 and rounds]
    if valid_forgetting:
        print(f"Average forgetting rate (%): {sum(valid_forgetting) / len(valid_forgetting):.2f}")


if __name__ == "__main__":
    main()
