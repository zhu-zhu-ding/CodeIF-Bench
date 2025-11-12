#!/usr/bin/env python3
"""Compute the average metrics across all entries in a JSONL results file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict
from typing import DefaultDict, Dict, Iterable



def extract_metrics(entry: Dict) -> Dict[str, Numeric]:
    """Return a dictionary of metrics for a single experiment entry."""
    summary = entry.get("metrics_summary", {})
    average = summary.get("average", {})
    if average:
        return average

    rounds = entry.get("rounds") or []
    if rounds:
        return rounds[-1].get("metrics", {})

    return {}


def iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            yield json.loads(stripped)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute average metrics across all records in a JSONL file."
    )
    parser.add_argument("--input_path", type=Path, help="Path to the input JSONL file.")
    args = parser.parse_args()

    totals: Dict[str, float] = {}
    count = 0
    round_totals: DefaultDict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    round_counts: DefaultDict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for record in iter_jsonl(args.input_path):
        metrics = extract_metrics(record)
        if not metrics:
            continue
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                totals[key] = totals.get(key, 0.0) + float(value)
        count += 1

        for idx, round_entry in enumerate(record.get("rounds") or [], start=1):
            round_metrics = round_entry.get("metrics", {})
            for key, value in round_metrics.items():
                if isinstance(value, (int, float)):
                    round_totals[idx][key] += float(value)
                    round_counts[idx][key] += 1

    if count == 0 or not totals:
        print("No metrics found in the provided file.")
        return

    averages = {key: total / count for key, total in totals.items()}

    per_round_average: Dict[str, Dict[str, float]] = {}
    for round_idx in sorted(round_totals.keys()):
        metrics_avg: Dict[str, float] = {}
        for metric_name, total in round_totals[round_idx].items():
            divisor = round_counts[round_idx].get(metric_name, 0)
            if divisor:
                metrics_avg[metric_name] = total / divisor
        if metrics_avg:
            per_round_average[str(round_idx)] = metrics_avg

    output = {
        "count": count,
        "average_metrics": averages,
        "per_round_average": per_round_average,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
