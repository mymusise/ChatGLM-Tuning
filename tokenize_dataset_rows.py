import argparse
import json
import random
import tqdm.auto as tqdm

import datasets
import transformers


def read_jsonl(path):
    # Manually open because .splitlines is different from iterating over lines
    with open(path, "r") as f:
        for line in f:
            yield json.loads(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_path", type=str, default="data/alpaca_data.jsonl")
    parser.add_argument("--save_path", type=str, default="data/alpaca")
    parser.add_argument("--max_seq_length", type=int, default=384)
    args = parser.parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "THUDM/chatglm-6b", trust_remote_code=True
    )

    all_tokenized = []
    for elem in tqdm.tqdm(read_jsonl(args.jsonl_path)):
        all_tokenized.append(
            tokenizer.encode(
                elem["text"], max_length=args.max_seq_length, truncation=True,
            )
        )
    random.shuffle(all_tokenized)

    ds = datasets.Dataset.from_dict({"input_ids": all_tokenized})
    ds.save_to_disk(args.save_path)
    print(f"Generated {len(all_tokenized)} samples.")


if __name__ == "__main__":
    main()
