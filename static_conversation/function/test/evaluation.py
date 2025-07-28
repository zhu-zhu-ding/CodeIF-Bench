from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union, Iterable, Dict
import itertools
import re
import numpy as np
import tqdm
from utils import (
    read_json,
    save_json
)
from data import MBPP_CODE, MBPP_SAMPLE, read_problems, stream_jsonl, write_jsonl
from execution import check_correctness


def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])


def extract_fortran(fortran_code):
        pattern = r"```python\r?\n(.*?)```"
        matches = re.findall(pattern, fortran_code, re.DOTALL)
        if matches:
            extracted_code = matches[0]
            return extracted_code
        else:
            return fortran_code

def remove_blank_lines_with_spaces(text):
    """
    Remove all blank lines (lines that are empty or contain only spaces) from a given string.
    
    Args:
        text (str): The input string.
        
    Returns:
        str: The string with blank lines removed.
    """
    return '\n'.join(line for line in text.splitlines() if line.strip())

def remove_triple_quotes_comments(function_str):
    pattern = r"    \"\"\"\r?\n(.*?)\"\"\""
    
    function_str_no_comments = re.sub(pattern, "", function_str, flags=re.DOTALL)
    
    return function_str_no_comments


def evaluate_functional_correctness(
    sample_file: str = MBPP_SAMPLE,
    k: List[int] = [1, 10, 100],
    n_workers: int = 4,
    timeout: float = 3.0,
    problem_file: str = MBPP_CODE,
    way: int = 0,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """

    problems = read_problems(problem_file)
    
    if way == 1:
        # Check the generated samples against test suites.
        with ThreadPoolExecutor(max_workers=n_workers) as executor:

            futures = []
            n_samples = 0
            results = defaultdict(list)

            print("Reading samples...")
            for sample in tqdm.tqdm(stream_jsonl(sample_file)):
                task_id = sample["task_id"]
                completion = extract_fortran(sample["completion"])
                # Only the problem's tests
                problem = problems[task_id]
                tests = problem["test"]
                args = (problem, completion, timeout, tests)
                future = executor.submit(check_correctness, *args)
                futures.append(future)
                n_samples += 1

            print("Running test suites...")
            for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
                result = future.result()
                results[result["task_id"]].append(result)

        # Calculate pass@k.
        total, correct = [], []
        for result in results.values():
            # result.sort()
            passed = [r["passed"] for r in result]
            total.append(len(passed))
            correct.append(sum(passed))
        total = np.array(total)
        correct = np.array(correct)

        ks = k
        pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean()
                    for k in ks if (total >= k).all()}

        # Finally, save the results in one file:
        def combine_results():
            for sample in stream_jsonl(sample_file):
                task_id = sample["task_id"]
                result = results[task_id].pop(0)
                sample["result"] = result["result"]
                sample["passed"] = result["passed"]
                sample["completion"] = extract_fortran(sample["completion"])
                yield sample

        out_file = sample_file + "_results.jsonl"
        print(f"Writing results to {out_file}...")
        write_jsonl(out_file, tqdm.tqdm(combine_results(), total=n_samples))

        return pass_at_k

    elif way == 2:
        pass_total = {}
        requirement_results = {
            "Input-Output Conditions": [],
            "Exception Handling": [],
            "Edge Case Handling": [],
            "Functionality Extension": [],
            "Annotation Coverage": [],
            "Code Complexity": [],
            "Code Standard": [],
        }
        n_samples = 0
        print("Reading samples...")
        task_samples = defaultdict(list)
        for sample in tqdm.tqdm(stream_jsonl(sample_file)):
            task_id = sample["task_id"]
            task_samples[task_id].append(sample)
        for task_id in task_samples:
            problem = problems[task_id]
            samples = task_samples[task_id]
            requirements = problem.get("requirements", {})
            # For each requirement in the problem, check the corresponding unit_tests
            for sample, requirement in zip(samples, requirements):
                completion = extract_fortran(sample["completion"])
                completion = remove_triple_quotes_comments(completion)
                completion = remove_blank_lines_with_spaces(completion)
                details = requirements.get(requirement)
                if requirement == "Functionality Extension":
                    tests = ( [details["unit_test"]] if isinstance(details["unit_test"], str) else details["unit_test"])
                else:
                    tests = ( [problem["test"]] if isinstance(problem["test"], str) else problem["test"]) + ( [details["unit_test"]] if isinstance(details["unit_test"], str) else details["unit_test"])
                # Assuming the tests are the same for each sample and requirement
                result = check_correctness(problem, completion, timeout, tests, requirement)
                requirement_results[requirement].append(result)
                n_samples += 1
        
        print("Running test suites...")
        # Now calculate pass@k for each requirement
        total_num = 0
        total_correct = 0
        for requirement, task_results in requirement_results.items():
            total, correct = 0, 0
            if not task_results:
                continue
            # Calculate pass@k for this requirement
            for results in task_results:
                passed = results["passed"]
                total += 1
                if passed:
                    correct += 1
            
            total_num += total
            total_correct += correct

            print("total:", total)
            print("correct:", correct)
            pass_at_k = 1 - (total - correct)/total

            # Store the pass@k for this requirement into pass_total
            pass_total[requirement] = pass_at_k

            # Save the results in one file
            def combine_results(requirement):
                for sample in stream_jsonl(sample_file):
                    task_id = sample["task_id"]
                    currequirement = sample["requirement"]
                    if currequirement == requirement:
                        requirement_results_for_task = requirement_results.get(requirement, [])
                        if requirement_results_for_task:
                            result = requirement_results_for_task.pop(0)
                            sample["result"] = result["result"]
                            sample["passed"] = result["passed"]
                            sample["completion"] = extract_fortran(sample["completion"])
                            yield sample


        for requirement in requirement_results:
            out_file = f"{sample_file}_{requirement}_results.jsonl"
            print(f"Writing results to {out_file}...")
            write_jsonl(out_file, tqdm.tqdm(combine_results(requirement), total=len(requirement_results[requirement])))

        print("total_num:", total_num)
        print("total_correct:", total_correct)
        total_pass_at_k = 1 - (total_num - total_correct)/total_num
        pass_total["total"] = total_pass_at_k

        # Return the pass@k for all requirements
        return pass_total

    elif way == 3:
        pass_total = {}
        requirement_results = {
            "Input-Output Conditions": [],
            "Exception Handling": [],
            "Edge Case Handling": [],
            "Functionality Extension": [],
            "Annotation Coverage": [],
            "Code Complexity": [],
            "Code Standard": [],
        }
        n_samples = 0
        print("Reading samples...")
        task_samples = defaultdict(list)
        for sample in tqdm.tqdm(stream_jsonl(sample_file)):
            task_id = sample["task_id"]
            task_samples[task_id].append(sample)
        for task_id in task_samples:
            problem = problems[task_id]
            samples = task_samples[task_id]
            requirements = problem.get("requirements", {})
            lists = problem.get("multi-turn", {})
            accumulated_tests = [problem["test"]] if isinstance(problem["test"], str) else problem["test"]
            # For each requirement in the problem, check the corresponding unit_tests
            for sample, l in zip(samples, lists):
                completion = extract_fortran(sample["completion"])
                completion = remove_triple_quotes_comments(completion)
                completion = remove_blank_lines_with_spaces(completion)
                details = requirements[l]
                if l == "Functionality Extension":
                    continue
                # Assuming the tests are the same for each sample and requirement
                accumulated_tests += [details["unit_test"]] if isinstance(details["unit_test"], str) else details["unit_test"]
                result = check_correctness(problem, completion, timeout, accumulated_tests, l)
                requirement_results[l].append(result)
                n_samples += 1
        
        print("Running test suites...")
        # Now calculate pass@k for each requirement
        total_num = 0
        total_correct = 0
        for requirement, task_results in requirement_results.items():
            total, correct = 0, 0
            if not task_results:
                continue
            # Calculate pass@k for this requirement
            for results in task_results:
                passed = results["passed"]
                total += 1
                if passed:
                    correct += 1
            
            total_num += total
            total_correct += correct

            print("total:", total)
            print("correct:", correct)
            pass_at_k = 1 - (total - correct)/total

            # Store the pass@k for this requirement into pass_total
            pass_total[requirement] = pass_at_k

            # Save the results in one file
            def combine_results(requirement):
                for sample in stream_jsonl(sample_file):
                    task_id = sample["task_id"]
                    currequirement = sample["requirement"]
                    if currequirement == requirement:
                        requirement_results_for_task = requirement_results.get(requirement, [])
                        if requirement_results_for_task:
                            result = requirement_results_for_task.pop(0)
                            sample["result"] = result["result"]
                            sample["passed"] = result["passed"]
                            sample["completion"] = extract_fortran(sample["completion"])
                            yield sample


        for requirement in requirement_results:
            out_file = f"{sample_file}_{requirement}_results.jsonl"
            print(f"Writing results to {out_file}...")
            write_jsonl(out_file, tqdm.tqdm(combine_results(requirement), total=len(requirement_results[requirement])))

        print("total_num:", total_num)
        print("total_correct:", total_correct)
        total_pass_at_k = 1 - (total_num - total_correct)/total_num
        pass_total["total"] = total_pass_at_k

        # Return the pass@k for all requirements
        return pass_total

    elif way == 4:
        pass_total = {}
        n_samples = 0
        print("Reading samples...")
        task_samples = defaultdict(list)
        results = []
        for sample in tqdm.tqdm(stream_jsonl(sample_file)):
            task_id = sample["task_id"]
            task_samples[task_id].append(sample)
        for task_id in task_samples:
            problem = problems[task_id]
            samples = task_samples[task_id]
            requirements = problem.get("requirements", {})
            lists = problem.get("multi-turn", {})
            accumulated_tests = [problem["test"]] if isinstance(problem["test"], str) else problem["test"]
            # For each requirement in the problem, check the corresponding unit_tests
            for sample, l in zip(samples, lists):
                completion = extract_fortran(sample["completion"])
                completion = remove_triple_quotes_comments(completion)
                completion = remove_blank_lines_with_spaces(completion)
                details = requirements[l]
                if l == "Functionality Extension":
                    continue
                # Assuming the tests are the same for each sample and requirement
                accumulated_tests += [details["unit_test"]] if isinstance(details["unit_test"], str) else details["unit_test"]
                result = check_correctness(problem, completion, timeout, accumulated_tests, l)
                results.append(result)
                n_samples += 1
        
        save_json("result.json",results,False)

        return 0

    elif way == 5:
        pass_total = {}
        n_samples = 0
        requirement_results = []
        print("Reading samples...")
        task_samples = defaultdict(list)
        for sample in tqdm.tqdm(stream_jsonl(sample_file)):
            task_id = sample["task_id"]
            task_samples[task_id].append(sample)
        for task_id in task_samples:
            problem = problems[task_id]
            samples = task_samples[task_id]
            requirements = problem.get("requirements", {})

            accumulated_tests = [problem["test"]] if isinstance(problem["test"], str) else problem["test"]
            
            for sample in samples:
                completion = extract_fortran(sample["completion"])
                completion = remove_triple_quotes_comments(completion)
                completion = remove_blank_lines_with_spaces(completion)
                requirement = sample["requirement"]
                details = requirements.get(requirement)
                accumulated_tests = ( [details["unit_test"]] if isinstance(details["unit_test"], str) else details["unit_test"])
                # print(accumulated_tests)
                result = check_correctness(problem, completion, timeout, accumulated_tests, requirement)
                requirement_results.append(result)
                n_samples += 1
        
        print("Running test suites...")
        
        out_file = f"{sample_file}_IA.jsonl"
        print(f"Writing results to {out_file}...")
        write_jsonl(out_file, tqdm.tqdm(requirement_results, total=len(requirement_results)))

        return []

    elif way == 6:
        pass_total = {}
        n_samples = 0
        requirement_results = []
        print("Reading samples...")
        task_samples = defaultdict(list)
        for sample in tqdm.tqdm(stream_jsonl(sample_file)):
            task_id = sample["task_id"]
            task_samples[task_id].append(sample)
        for task_id in task_samples:
            problem = problems[task_id]
            samples = task_samples[task_id]
            requirements = problem.get("requirements", {})

            accumulated_tests = [problem["test"]] if isinstance(problem["test"], str) else problem["test"]
            
            for sample in samples:
                completion = extract_fortran(sample["completion"])
                completion = remove_triple_quotes_comments(completion)
                completion = remove_blank_lines_with_spaces(completion)
                requirement = sample["requirement"]
                details = requirements.get(requirement)
                if requirement == "Functionality Extension":
                    continue
                else:                
                    accumulated_tests += ( [details["unit_test"]] if isinstance(details["unit_test"], str) else details["unit_test"])

                result = check_correctness(problem, completion, timeout, accumulated_tests, requirement)
                requirement_results.append(result)
                n_samples += 1
        
        print("Running test suites...")
        
        out_file = f"{sample_file}_CA.jsonl"
        print(f"Writing results to {out_file}...")
        write_jsonl(out_file, tqdm.tqdm(requirement_results, total=len(requirement_results)))


        # Return the pass@k for all requirements
        return []

    elif way == 7:
        base = read_json("base.jsonl", False)
        base1 = {item["task_id"]: item for item in base}

        pass_total = {}
        n_samples = 0
        requirement_results = []
        print("Reading samples...")
        task_samples = defaultdict(list)
        for sample in tqdm.tqdm(stream_jsonl(sample_file)):
            task_id = sample["task_id"]
            task_samples[task_id].append(sample)
        for task_id in task_samples:


            problem = problems[task_id]
            samples = task_samples[task_id]
            requirements = problem.get("requirements", {})

            prepassed = []

            base_result = base1[task_id]
            if base_result["result"] == "passed":
                prepassed.append("Base")
            accumulated_tests = [problem["test"]] if isinstance(problem["test"], str) else problem["test"]
            
            for sample in samples:
                completion = extract_fortran(sample["completion"])
                completion = remove_triple_quotes_comments(completion)
                completion = remove_blank_lines_with_spaces(completion)
                requirement = sample["requirement"]
                details = requirements.get(requirement)
                if requirement == "Functionality Extension":
                   continue
                else:                
                    accumulated_tests += ( [details["unit_test"]] if isinstance(details["unit_test"], str) else details["unit_test"])
                result = check_correctness(problem, completion, timeout, accumulated_tests, requirement)
                IFR = 0
                if result["result"] == "passed":
                    prepassed.append(requirement)
                    IFR = 0
                else:
                    num_unpassed = 0
                    for item in prepassed:
                        if item == "Base":
                            temp_tests = [problem["test"]] if isinstance(problem["test"], str) else problem["test"]
                        else:
                            details = requirements.get(item)
                            temp_tests = [details["unit_test"]] if isinstance(details["unit_test"], str) else details["unit_test"]
                        result1 = check_correctness(problem, completion, timeout, temp_tests, item)
                        if result1["result"] != "passed":
                            num_unpassed += 1
                    IFR = num_unpassed / len(prepassed) if len(prepassed) !=0 else 0

                result["IFR"] = IFR
                requirement_results.append(result)
                
                n_samples += 1
        
        print("Running test suites...")
        
        out_file = f"{sample_file}_IFA_result.jsonl"
        print(f"Writing results to {out_file}...")
        write_jsonl(out_file, tqdm.tqdm(requirement_results, total=len(requirement_results)))

        return []
