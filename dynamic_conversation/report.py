
import json
from collections import defaultdict

def load_all_turns_from_jsonls(filepaths):
    all_tasks = []
    for path in filepaths:
        with open(path, 'r') as f:
            for line in f:
                task = json.loads(line)
                all_tasks.append(task)
    return all_tasks

def compute_pairwise_metrics(all_tasks):
    pair_metrics = defaultdict(lambda: {"IA": [], "IFR": [], "CA": [], "IFE": []})

    for task in all_tasks:
        turns = task['turns']
        n = len(turns)
        i = 0
        while i < n:
            if i + 1 < n:
                pair_name = f"{i+1}-{i+2}"
                ia = (turns[i]['IA'] + turns[i+1]['IA']) / 2
                ifr = (turns[i]['IFR'] + turns[i+1]['IFR']) / 2
                ca = (turns[i]['CA'] + turns[i+1]['CA']) / 2
                ife = (turns[i]['IFE'] + turns[i+1]['IFE']) / 2
                i += 2
            else:
                pair_name = f"{i+1}-{i+2}"
                ia = turns[i]['IA']
                ifr = turns[i]['IFR']
                ca = turns[i]['CA']
                ife = turns[i]['IFE']
                i += 1

            pair_metrics[pair_name]['IA'].append(ia)
            pair_metrics[pair_name]['IFR'].append(ifr)
            pair_metrics[pair_name]['CA'].append(ca)
            pair_metrics[pair_name]['IFE'].append(ife)

    return pair_metrics

def print_metrics_summary(pair_metrics):
    all_ia, all_ifr, all_ca, all_ife = [], [], [], []
    for pair in sorted(pair_metrics, key=lambda x: int(x.split('-')[0])):
        metrics = pair_metrics[pair]
        avg_ia = sum(metrics['IA']) / len(metrics['IA'])*100
        avg_ifr = sum(metrics['IFR']) / len(metrics['IFR'])*100
        avg_ca = sum(metrics['CA']) / len(metrics['CA'])*100
        avg_ife = sum(metrics['IFE']) / len(metrics['IFE'])*100
        all_ia.append(avg_ia)
        all_ifr.append(avg_ifr)
        all_ca.append(avg_ca)
        all_ife.append(avg_ife)

        print(f"{pair}: IA={avg_ia:.1f}, IFR={avg_ifr:.1f}, CA={avg_ca:.1f}, IFE={avg_ife:.1f}")
    total_avg_ia = sum(all_ia) / len(all_ia)
    total_avg_ifr = sum(all_ifr) / len(all_ifr)
    total_avg_ca = sum(all_ca) / len(all_ca)
    total_avg_ife = sum(all_ife) / len(all_ife)
    print()
    print(f"avg: IA={total_avg_ia:.1f}, IFR={total_avg_ifr:.1f}, CA={total_avg_ca:.1f}, IFE={total_avg_ife:.1f}")
###input file path###
file1 = ""

all_tasks = load_all_turns_from_jsonls([file1])
pair_metrics = compute_pairwise_metrics(all_tasks)
print_metrics_summary(pair_metrics)
