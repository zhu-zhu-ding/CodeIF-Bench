
from llm_factory import LLMFactory
from pathlib import Path
import json
from tqdm import tqdm
import argparse
from utils import read_json, save_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=Path, help="Path to the input file.")
    parser.add_argument("--output_path", type=Path, help="Path to the output file.")
    parser.add_argument("--model_name", type=str, help="LLM")

    args = parser.parse_args()

    ######load data##########
    with open(args.input_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    data = data[:1]
    result_list = []
    for sample in tqdm(data):
        target_llm = LLMFactory(llm_type="closed", model=args.model_name)
        sample['result'] = []
        sample['tokens'] = []
        #####prompt######
        prompt = sample['prompt']+"Please output the python function directly!"

        session_messages = [
            {'role': 'user', 'content': prompt}
        ]
        result = target_llm.generate(session_messages)
        session_messages.append({'role': 'assistant', 'content': result})
        for key in tqdm(sample['multi-turn']):
            prompt = f"{sample['requirements'][key]['requirement']}"
            session_messages.append({'role': 'user', 'content': prompt})
            result = target_llm.generate(session_messages)
            session_messages.append({'role': 'assistant', 'content': result})
            sample['session_messages'] = session_messages
            sample['tokens'].append(target_llm.last_usage)
        sample['all_tokens'] = target_llm.total_usage
        result_list.append(sample)
        save_json(args.output_path,result_list,False)
