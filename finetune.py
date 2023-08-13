from transformers.integrations import TensorBoardCallback
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainingArguments
from transformers import Trainer, HfArgumentParser
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType
from dataclasses import dataclass, field
import datasets
import os


# 从预训练模型加载tokenizer
# tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("/home/che/Models/chatglm-6b", trust_remote_code=True)


# 定义FinetuneArguments数据类，用于存储微调的参数
@dataclass
class FinetuneArguments:
    dataset_path: str = field(default="data/alpaca")   # 数据集路径
    model_path: str = field(default="output")          # 模型保存路径
    lora_rank: int = field(default=8)                  # Lora排名，用于peft模型的设置


# 自定义CastOutputToFloat类，继承自nn.Sequential，用于将输出转换为float32类型
class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)


# 数据处理函数data_collator，用于将输入数据按照最长序列长度进行padding
def data_collator(features: list) -> dict:
    len_ids = [len(feature["input_ids"]) for feature in features]
    longest = max(len_ids)
    input_ids = []
    labels_list = []
    for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
        ids = feature["input_ids"]
        seq_len = feature["seq_len"]
        labels = (
            [-100] * (seq_len - 1) + ids[(seq_len - 1) :] + [-100] * (longest - ids_l)
        )
        ids = ids + [tokenizer.pad_token_id] * (longest - ids_l)
        _ids = torch.LongTensor(ids)
        labels_list.append(torch.LongTensor(labels))
        input_ids.append(_ids)
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)
    return {
        "input_ids": input_ids,
        "labels": labels,
    }


# 自定义ModifiedTrainer类，继承自Trainer，用于微调训练，并对模型保存进行了自定义
class ModifiedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return model(
            input_ids=inputs["input_ids"],
            labels=inputs["labels"],
        ).loss

    def save_model(self, output_dir=None, _internal_call=False):
        from transformers.trainer import TRAINING_ARGS_NAME

        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        saved_params = {
            k: v.to("cpu") for k, v in self.model.named_parameters() if v.requires_grad
        }
        torch.save(saved_params, os.path.join(output_dir, "adapter_model.bin"))


def main():
    # 创建TensorBoard的SummaryWriter，用于记录训练过程的日志
    writer = SummaryWriter()
    # 使用HfArgumentParser解析命令行参数并存储为FinetuneArguments和TrainingArguments两个数据类的实例
    finetune_args, training_args = HfArgumentParser(
        (FinetuneArguments, TrainingArguments)
    ).parse_args_into_dataclasses()

    # init model
    # 初始化模型，从预训练模型加载微调模型
    model = AutoModel.from_pretrained(
        "/home/che/Models/chatglm-6b", load_in_8bit=True, trust_remote_code=True, device_map="auto"
    )
    model.gradient_checkpointing_enable()             # 开启梯度检查点
    model.enable_input_require_grads()                # 开启输入的梯度计算
    model.is_parallelizable = True                    # 模型可并行计算
    model.model_parallel = True                       # 使用模型并行计算
    model.lm_head = CastOutputToFloat(model.lm_head)  # 将输出转换为float32类型
    model.config.use_cache = (
        False  # silence the warnings. Please re-enable for inference! # 关闭缓存以减少内存占用，但在推断时需要重新开启
    )

    # setup peft
    # 设置peft模型，设置LoraConfig，用于构造peft模型
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=finetune_args.lora_rank,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    # 加载peft模型
    model = get_peft_model(model, peft_config)

    # load dataset
    # 从磁盘加载数据集
    dataset = datasets.load_from_disk(finetune_args.dataset_path)
    print(f"\n{len(dataset)=}\n")  # 打印数据集的样本数量

    # start train
    # 开始训练
    trainer = ModifiedTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        callbacks=[TensorBoardCallback(writer)],  # 添加TensorBoard的回调函数，用于记录训练过程的日志
        data_collator=data_collator,
    )
    trainer.train()  # 执行训练
    writer.close()   # 关闭TensorBoard的SummaryWriter
    # save model
    # 保存模型
    model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
