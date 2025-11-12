ROOT=/home/user/benchmark/repo

python $ROOT/check_source_code.py $ROOT/Source_Code

# Compute Pass@1
python run_pass@k.py \
    --output_file $ROOT/result.jsonl \
    --log_file $ROOT/result_result.jsonl \
    --source_code_root $ROOT/Source_Code \
    --data_file $ROOT/dev_data_level_1.jsonl \
    --n 1 \
    --k 1
