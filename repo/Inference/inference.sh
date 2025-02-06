ROOT=/home/user/benchmark/repo

python inference.py \
    --data_file $ROOT/L_1_part_2.jsonl \
    --output_file $ROOT/result.jsonl \
    --model_path /home/user/model/Qwen2.5-Coder-7B-Instruct \
    --level 1 \
    --way 1
    # --lora_path 
    # --base_file $ROOT/pre_result.jsonl \
