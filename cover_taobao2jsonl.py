import argparse
import json
from tqdm import tqdm


def format_example(example: dict) -> dict:
    # context = f"Instruction: {example['instruction']}\n"
    # if example.get("input"):
    #     context += f"Input: {example['input']}\n"
    # context += "Answer: "
    # target = example["output"]
    return {"content": example['content'], "summary": example['summary']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data-taobao/AdvertiseGen/train.json")
    parser.add_argument("--save_path", type=str, default="./data-taobao/AdvertiseGen/train_json_list.json")

    args = parser.parse_args()
    with open(args.data_path) as f:
        examples = [json.loads(line) for line in f.readlines()]

    with open(args.save_path, 'w') as f:
        for example in tqdm(examples, desc="formatting.."):
            f.write(json.dumps(format_example(example), ensure_ascii=False) + '\n')


if __name__ == "__main__":
    main()
