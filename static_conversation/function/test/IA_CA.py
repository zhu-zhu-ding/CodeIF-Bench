from collections import defaultdict
import json
from utils import (
    read_json,
    save_json
)
base = read_json("base_results.jsonl", False)
base1 = {item["task_id"]: item for item in base}

data = read_json("CA.jsonl", False)

data1 = defaultdict(list)
for item in data:
    task_id = item["task_id"]
    data1[task_id].append(item)

round_correct = defaultdict(int)
round_total = defaultdict(int)

total_correct = 0
total_num = 0

for task_id in data1:
    # print(item)
    round_total[0] += 1
    total_num += 1
    if base1[task_id]["result"] == "passed":
        round_correct[0] += 1
        total_correct += 1
    for i, item in enumerate(data1[task_id]):
        round_total[i+1] += 1
        total_num += 1
        if item["result"] == "passed":
            round_correct[i+1] += 1
            total_correct += 1

for i in sorted(round_total):
    correct = round_correct[i]
    total = round_total[i]
    acc = correct / total if total > 0 else 0
    print(f"Round {i+1}: Correct = {correct}, Total = {total}, Accuracy = {acc:.2%}")

print(f"Total: Correct = {total_correct}, Total = {total_num}, Accuracy = {total_correct/total_num:.2%}")