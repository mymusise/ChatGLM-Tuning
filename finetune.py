from transformers import TrainingArguments
from transformers import Trainer, HfArgumentParser
from modeling_chatglm import ChatGLMForConditionalGeneration
import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType
from dataclasses import dataclass, field
import datasets
import os


@dataclass
class FinetuneArguments:
    dataset_path: str = field(default="data/alpaca")
    model_path: str = field(default="output")
    lora_rank: int = field(default=8)


class CastOutputToFloat(nn.Sequential):
    def forward(self, x): return super().forward(x).to(torch.float32)


class ModifiedTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        input_shape = inputs["input_ids"].shape
        return model(
            input_ids=inputs["input_ids"],
            attention_mask=torch.ones(1, 1, input_shape[-1], input_shape[-1]).bool(),
            labels=inputs["input_ids"],
        ).loss


def data_collator(features: list) -> dict:
    len_ids = [len(feature['input_ids']) for feature in features]
    longest = max(len_ids)
    input_ids = []
    for ids_l, feature in sorted(zip(len_ids, features), key=lambda x:-x[0]):
        ids = feature['input_ids']
        _ids = torch.LongTensor(ids + [150004] * (longest - ids_l))
        input_ids.append(_ids)
    return {"input_ids": torch.stack(input_ids)}


def save_tunable_parameters(model, path):
    saved_params = {
        k: v.to("cpu")
        for k, v in model.named_parameters()
        if v.requires_grad
    }
    torch.save(saved_params, path)


def main():
    finetune_args, training_args = HfArgumentParser(
        (FinetuneArguments, TrainingArguments)).parse_args_into_dataclasses()

    # init model
    model = ChatGLMForConditionalGeneration.from_pretrained(
        "THUDM/chatglm-6b", load_in_8bit=True, trust_remote_code=True, device_map='auto')
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.is_parallelizable = True
    model.model_parallel = True
    model.lm_head = CastOutputToFloat(model.lm_head)
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

    # setup peft
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=finetune_args.lora_rank,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    model = get_peft_model(model, peft_config)

    # load dataset
    dataset = datasets.load_from_disk(finetune_args.dataset_path)

    # start train
    trainer = ModifiedTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        data_collator=data_collator,
    )
    trainer.train()

    # save model
    save_tunable_parameters(model, os.path.join(training_args.output_dir, "chatglm-lora.pt"))



if __name__ == "__main__":
    main()
