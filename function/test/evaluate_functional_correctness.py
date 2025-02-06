import sys
from data import MBPP_SAMPLE, MBPP_CODE
from evaluation import evaluate_functional_correctness
from argparse import ArgumentParser

def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--sample_file', type=str, default=MBPP_SAMPLE)
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--timeout', type=float, default=3.0)
    parser.add_argument('--problem_file', type=str, default=MBPP_CODE)
    parser.add_argument('--way', type=int, default=1)
    return parser.parse_args()

def entry_point(
    sample_file: str = MBPP_SAMPLE,
    k: int = "1",
    n_workers: int = 4,
    timeout: float = 3.0,
    problem_file: str = MBPP_CODE,
    way: int = 1,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    k = list(map(int, k.split(",")))
    results = evaluate_functional_correctness(sample_file, k, n_workers, timeout, problem_file, way)
    print(results)

if __name__ == "__main__":
    args = get_parser()
    entry_point(
        sample_file=args.sample_file,
        k="1",
        n_workers=args.n_workers,
        timeout=args.timeout,
        problem_file=args.problem_file,
        way =args.way,
    )