from collections import defaultdict
import json
from utils import (
    read_json,
    save_json
)

data = read_json("IFA_result.jsonl", False)
data1 = defaultdict(list)
for item in data:
    task_id = item["task_id"]
    data1[task_id].append(item)

flag = True

round_IFR = defaultdict(int)
round_total = defaultdict(int)

for i in range(8):
    round_IFR[i] = 0
    round_total[i] = 0

for task_id in data1:
    for i, item in enumerate(data1[task_id]):
        print(item["IFR"])
        round_IFR[i] += item["IFR"]
        round_total[i] += 1

for i in sorted(round_total):
    IFR = round_IFR[i]
    total = round_total[i]
    acc = IFR / total if total > 0 else 0
    print(f"Round {i+1}: Correct = {IFR}, Total = {total}, Accuracy = {acc:.2%}")