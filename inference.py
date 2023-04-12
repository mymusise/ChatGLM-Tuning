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
peft_path = "output_taobao/checkpoint-1500/adapter_model.bin"

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=True,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)

model = get_peft_model(model, peft_config)
model.load_state_dict(torch.load(peft_path), strict=False)
torch.set_default_tensor_type(torch.cuda.FloatTensor)

with open('data-taobao/AdvertiseGen/dev.json', 'r', encoding='utf-8') as f:
    lines = [json.loads(line) for line in f.readlines()]

with torch.no_grad():
    for idx, item in enumerate(lines):
        input_text = item['content']
        label = item['summary']
        ids = tokenizer.encode(input_text)
        input_ids = torch.LongTensor([ids])
        out = model.generate(
            input_ids=input_ids,
            max_length=150,
            do_sample=False,
            temperature=0
        )
        out_text = tokenizer.decode(out[0][len(ids):])

        print(f'Input: {input_text}')
        print(f'label: {label}')
        print(f'predict: {out_text}')
