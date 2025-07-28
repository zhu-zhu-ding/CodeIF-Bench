from collections import defaultdict
import json
from utils import (
    read_json,
    save_json
)

data = read_json("IFR.jsonl", False)
data1 = defaultdict(list)
for item in data:
    namespace = item["namespace"]
    data1[namespace].append(item)

round_IFR = defaultdict(int)
round_total = defaultdict(int)

for i in range(8):
    round_IFR[i] = 0
    round_total[i] = 0

for namespace in data1:
    for i, item in enumerate(data1[namespace]):
        print(item["IFR"])
        round_IFR[i] += item["IFR"]
        round_total[i] += 1

for i in sorted(round_total):
    IFR = round_IFR[i]
    total = round_total[i]
    acc = IFR / total if total > 0 else 0
    print(f"Round {i+1}: Correct = {IFR}, Total = {total}, Accuracy = {acc:.2%}")