ROOT=/home/user/benchmark/function

python inference_gpt.py \
    --data_file $ROOT/L_1_part_1.jsonl \
    --output_file $ROOT/result.jsonl \
    --model GPT \
    --way 1
    # --base_file $ROOT/pre_result.jsonl 
