ROOT=/home/user/benchmark/function

python evaluate_functional_correctness.py \
    --sample_file $ROOT/result.jsonl \
    --n_workers 4 \
    --timeout 3.0 \
    --problem_file $ROOT/L_1_part_1.jsonl \
    --way 1