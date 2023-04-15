import os
import json

def read_json_list(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f.readlines()]


def save_json_list(path, lines):
    with open(path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')


def read_json_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json_data(path, json_data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False)


result_500 = read_json_list('data-renmin/output/checkpoint-500/generate_results.json')
result_2000 = read_json_list('data-renmin/output/checkpoint-2000/generate_results.json')

assert len(result_2000) == len(result_500)

not_equal_n = 0
for line_2000, line_500 in zip(result_2000, result_500):
    if line_2000['predict'] != line_500['predict']:
        print(f'input: {line_500["input"]}')
        print(f'label: {line_2000["label"]}')
        print(f'predict2000: {line_2000["predict"]}')
        print(f'predict500: {line_500["predict"]}')
        not_equal_n += 1
print(f'data size: {len(result_2000)}')
print(f'not equal size: {not_equal_n}')
