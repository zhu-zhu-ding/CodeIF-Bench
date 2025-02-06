ROOT=/home/user/benchmark/repo

python inference_gpt.py \
    --data_file $ROOT/L_1_part_2.jsonl \
    --output_file $ROOT/result.jsonl \
    --base_file $ROOT/pre_result.jsonl \
    --level 1 \
    --way 1 \
    --model QWEN
    # --base_file $ROOT/pre_result.jsonl
