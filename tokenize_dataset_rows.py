import argparse
import json
from tqdm import tqdm

import datasets
import transformers


def preprocess(tokenizer, example, max_seq_length=512):
    prompt = example["context"]
    target = example["target"]
    prompt_ids = tokenizer.encode(prompt, max_length=max_seq_length, truncation=True)
    target_ids = tokenizer.encode(
        target, max_length=max_seq_length, truncation=True, add_special_tokens=False
    )
    input_ids = prompt_ids + target_ids + [tokenizer.eos_token_id]
    return {"input_ids": input_ids, "seq_len": len(prompt_ids)}


def read_jsonl(path):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "THUDM/chatglm-6b", trust_remote_code=True
    )
    with open(path, "r") as f:
        for line in tqdm(f):
            example = json.loads(line)
            yield preprocess(tokenizer, example)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_path", type=str, default="data/alpaca_data.jsonl")
    parser.add_argument("--save_path", type=str, default="data/alpaca")
    parser.add_argument("--max_seq_length", type=int, default=384)
    args = parser.parse_args()

    dataset = datasets.Dataset.from_generator(
        lambda: read_jsonl("data/alpaca_data.jsonl")
    )
    dataset.save_to_disk(args.save_path)


if __name__ == "__main__":
    main()
