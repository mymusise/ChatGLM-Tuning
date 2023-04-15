import json
import torch
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from transformers import AutoModel
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoTokenizer

torch.set_default_tensor_type(torch.cuda.HalfTensor)

from cover_alpaca2jsonl import format_example

model = AutoModel.from_pretrained("/work/models/chatglm-6b", trust_remote_code=True, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained("/work/models/chatglm-6b", trust_remote_code=True)
peft_path = "data-renmin/output/checkpoint-500/adapter_model.bin"

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=True,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)

model = get_peft_model(model, peft_config)
model.load_state_dict(torch.load(peft_path), strict=False)
torch.set_default_tensor_type(torch.cuda.FloatTensor)

test_file = 'data-renmin/data/dev.json'
with open(test_file, 'r', encoding='utf-8') as f:
    lines = [json.loads(line) for line in f.readlines()]

lines = lines[:100]

results = []
with torch.no_grad():
    for idx, item in enumerate(lines):
        input_text = item['instruction']
        label = item['output']
        ids = tokenizer.encode(input_text)
        input_ids = torch.LongTensor([ids])
        out = model.generate(
            input_ids=input_ids,
            max_length=350,
            do_sample=False,
            temperature=0
        )
        out_text = tokenizer.decode(out[0][len(ids):])

        if idx < 5:
            print(f'Input: {input_text}')
            print(f'label: {label}')
            print(f'predict: {out_text}')

        results.append({
            'input': input_text,
            'label': label,
            'predict': out_text
        })

trained_model_path = peft_path
path = os.path.join(os.path.dirname(trained_model_path), 'generate_results.json')
with open(path, 'w', encoding='utf-8') as f:
    for line in results:
        f.write(json.dumps(line, ensure_ascii=False) + '\n')
