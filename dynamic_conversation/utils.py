import json
import os
import subprocess
import threading


def read_json(data_path,is_list=True):
    if is_list:
        try:
            return json.load(open(data_path, 'r',encoding="utf-8"))
        except Exception as e:
            print(f"read json_path {data_path} exception:{e}")
            return None
    else:
        try:
            return [json.loads(line) for line in open(data_path, 'r',encoding="utf-8")]
        except Exception as e:
            print(f"read json_path {data_path} exception:{e}")
            return None

def save_json(data_path,data_list,is_list=True):
    if is_list:
        try:
            open(data_path, 'w', encoding="utf-8",).write(json.dumps(data_list,indent=4))
        except Exception as e:
            print(f"save json_path {data_path} exception:{e}")
            return None
    else:
        try:
            with open(data_path, 'w', encoding="utf-8") as jsonl_file:
                for save_item in data_list:
                    jsonl_file.write(json.dumps(save_item) + '\n')
        except Exception as e:
            print(f"save json_path {data_path} exception:{e}")
            return None