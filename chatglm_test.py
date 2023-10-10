from transformers import AutoModel, AutoTokenizer

model_chatglm_path = '/home/adminz/ChatGLM-Tuning/model_path/chatglm'

tokenizer = AutoTokenizer.from_pretrained(model_chatglm_path, trust_remote_code=True)


model = AutoModel.from_pretrained(model_chatglm_path, trust_remote_code=True, load_in_8bit=True, device_map='auto')


# input_text = "假设你现在是一个游戏策划，现在你要设计一款生存类的RPG游戏，请帮忙设计这款游戏的世界机制："

# ids = tokenizer.encode(input_text, return_tensors='pt')
# print('ids:',ids)


# outs = model.generate(ids, max_length=128)

# response = tokenizer.batch_decode(outs)
# print(response[0])


from peft import LoraConfig, TaskType, prepare_model_for_int8_training, get_peft_model

model.gradient_checkpointing_enable()
model.is_parallelizable = True
model.model_parallel = True
model.config.use_cache = (
    False  # silence the warnings. Please re-enable for inference!
)

prepare_model_for_int8_training(model)
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)
model = get_peft_model(model, peft_config)



datas = [
{
    "context": "大师，我最近事业不顺利，感觉很迷茫，该怎么办？",
    "target": "孩子，生活中的起伏和挫折是常态。放下烦恼，用平静的心态面对困境，努力奋斗，同时保持内心的善良与慈悲，你将找到属于你的光明之路。",
}]

#
# for item in datas:
#     text = f"问:{item['context']}\n答:"
#     ids = tokenizer.encode(text, return_tensors='pt')
#     outs = model.generate(input_ids=ids, max_length=128)
#     print(tokenizer.batch_decode(outs)[0])
#
#
from tokenize_dataset_rows import preprocess
from datasets import Dataset
#
dataset = [preprocess(tokenizer, model.config, item, max_seq_length=256) for item in datas]
#
dataset = Dataset.from_list(dataset)


from finetune import ModifiedTrainer, data_collator
from transformers import TrainingArguments

training_args = TrainingArguments(
    "output",
    fp16 =True,
    gradient_accumulation_steps=1,
    per_device_train_batch_size = 5,
    learning_rate = 1e-4,
    num_train_epochs=80,
    logging_steps=10,
    remove_unused_columns=False,
    seed=0,
    data_seed=0,
    group_by_length=False,
)
#
#
#
trainer = ModifiedTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    data_collator=data_collator,
)
print('trainer:',trainer)

trainer.train()


# import torch

# model.config.use_cache = (
#     True
# )
#
# for item in datas:
#     text = f"问:{item['context']}\n答:"
#     ids = tokenizer.encode(text, return_tensors='pt')
#     outs = model.generate(input_ids=ids, max_length=128)
#     print(tokenizer.batch_decode(outs)[0])
