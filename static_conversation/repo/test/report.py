import json
log_file = "/home/user/benchmark/Experiments/level_1/claude-3-5-sonnet-20241022/multi_round/result_result.jsonl"
result = {}
with open(log_file, 'r') as f:
    for line in f:
        js = json.loads(line)
        if js['type'] not in result:
            result[js['type']] = {'data_len':0,'data_pass':0}
        else:
            result[js['type']]['data_len'] +=1
        if js['Result'] == "Pass":
            result[js['type']]['data_pass'] +=1
total = 0
total_pass = 0 
for key, value in result.items():
    total+=value['data_len']
    total_pass+= value['data_pass']
    print(f"{key}: pass@1 = {value['data_pass']/value['data_len']*100}%")
print(f"total: pass@1 = {total_pass/total*100}%")

