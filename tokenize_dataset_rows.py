import argparse
import json
from tqdm import tqdm

import datasets
import transformers


# def preprocess(tokenizer, config, example, max_seq_length):
#     prompt = example["context"]
#     target = example["target"]
#     prompt_ids = tokenizer.encode(prompt, max_length=max_seq_length, truncation=True)
#     target_ids = tokenizer.encode(
#         target,
#         max_length=max_seq_length,
#         truncation=True,
#         add_special_tokens=False)
#     input_ids = prompt_ids + target_ids + [config.eos_token_id]
#     return {"input_ids": input_ids, "seq_len": len(prompt_ids)}


def preprocess(tokenizer, config, example, max_seq_length):
    prompt = example["context"]
    target = example["target"]
    prompt_ids = tokenizer.encode(prompt, max_length=max_seq_length, truncation=True)
    target_ids = tokenizer.encode(
        target,
        max_length=max_seq_length,
        truncation=True,
        add_special_tokens=False)
    input_ids = prompt_ids + target_ids + [config.eos_token_id]
    return {"input_ids": input_ids, "seq_len": len(prompt_ids)}


def preprocess_chatglm2(tokenizer, config, example, max_seq_length):
    prompt = example["context"]
    target = example["target"]

    return {"prompt": prompt, "target": target}

# def preprocess_function_train(examples):
#     max_seq_length = data_args.max_source_length + data_args.max_target_length + 1
#
#     model_inputs = {
#         "input_ids": [],
#         "labels": [],
#     }
#     for i in range(len(examples[prompt_column])):
#         if examples[prompt_column][i] and examples[response_column][i]:
#             query, answer = examples[prompt_column][i], examples[response_column][i]
#
#             history = examples[history_column][i] if history_column is not None else None
#             prompt = tokenizer.build_prompt(query, history)
#
#             prompt = prefix + prompt
#             a_ids = tokenizer.encode(text=prompt, add_special_tokens=True, truncation=True,
#                                      max_length=data_args.max_source_length)
#             b_ids = tokenizer.encode(text=answer, add_special_tokens=False, truncation=True,
#                                      max_length=data_args.max_target_length)
#
#             context_length = len(a_ids)
#             input_ids = a_ids + b_ids + [tokenizer.eos_token_id]
#             labels = [tokenizer.pad_token_id] * context_length + b_ids + [tokenizer.eos_token_id]
#
#             pad_len = max_seq_length - len(input_ids)
#             input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
#             labels = labels + [tokenizer.pad_token_id] * pad_len
#             if data_args.ignore_pad_token_for_loss:
#                 labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]
#
#             model_inputs["input_ids"].append(input_ids)
#             model_inputs["labels"].append(labels)
#
#     return model_inputs



def read_jsonl(path, max_seq_length, skip_overlength=False):
    model_name = "THUDM/chatglm-6b"
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True)
    config = transformers.AutoConfig.from_pretrained(
        model_name, trust_remote_code=True, device_map='auto')
    with open(path, "r") as f:
        for line in tqdm(f.readlines()):
            example = json.loads(line)
            feature = preprocess(tokenizer, config, example, max_seq_length)
            if skip_overlength and len(feature["input_ids"]) > max_seq_length:
                continue
            feature["input_ids"] = feature["input_ids"][:max_seq_length]
            yield feature


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_path", type=str, default="data/alpaca_data.jsonl")
    parser.add_argument("--save_path", type=str, default="data/alpaca")
    parser.add_argument("--max_seq_length", type=int, default=384)
    parser.add_argument("--skip_overlength", type=bool, default=False)
    args = parser.parse_args()

    dataset = datasets.Dataset.from_generator(
        lambda: read_jsonl(args.jsonl_path, args.max_seq_length, args.skip_overlength)
    )
    dataset.save_to_disk(args.save_path)


if __name__ == "__main__":
    main()
