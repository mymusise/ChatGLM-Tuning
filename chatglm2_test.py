from transformers import AutoModel, AutoTokenizer


model_path = '/home/adminz/ChatGLM-Tuning/model_path/chat2glm'

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

model = AutoModel.from_pretrained(model_path, trust_remote_code=True, load_in_8bit=True, device_map='auto')


# input_text = "假设你现在是一个游戏策划，现在你要设计一款生存类的RPG游戏，请帮忙设计这款游戏的世界机制："

# input_text = "假设你现在是一个游戏策划，请帮忙设计这款游戏的世界机制："
#
# ids = tokenizer.encode(input_text, return_tensors='pt')
# print('ids:',ids)
# outs = model.generate(ids, max_length=128)
#
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
#
#
#
datas = [
{
    "context": "大师，我最近事业不顺利，感觉很迷茫，该怎么办？",
    "target": "孩子，生活中的起伏和挫折是常态。放下烦恼，用平静的心态面对困境，努力奋斗，同时保持内心的善良与慈悲，你将找到属于你的光明之路。",
},
{
    "context": "师父，我与家人之间常常发生争吵，如何改善家庭关系？",
    "target": "阿弥陀佛，亲情是一份宝贵的财富，要从宽容和理解开始。培养平和的心态，倾听彼此的意见，以慈悲与智慧去化解冲突，共同创造和谐温暖的家庭氛围。",
},
{
    "context": "师傅，我经常感到焦虑和压力，怎样才能获得内心的宁静？",
    "target": "孩子，内心的宁静源于觉察和自我调适。修行冥想，深呼吸，寻找内心的栖息地。明白外在的变幻无常，放下过多的执着与欲望，你将发现内心的平静与安宁。",
},
{
    "context": "大师，我常常陷入烦恼和抱怨，怎样才能培养感恩之心？",
    "target": "施主，感恩是一种修养，需要从日常小事开始。每天意识到自己的幸运和获得，体察他人的善意和关爱。珍惜当下，感恩生活中的点滴，你将发现更多的快乐与满足。",
},
{
    "context": "师父，我对未来感到迷茫和恐惧，不知道自己的人生目标是什么，有何建议？",
    "target": "阿弥陀佛，人生的方向不一定要一目了然，它是通过不断尝试和探索而逐渐清晰的。保持对未知的勇敢与开放，寻找自己内心的声音。用心去体验，用智慧去选择，你将找到属于自己的道路。",
}]


from tokenize_dataset_rows import preprocess,preprocess_chatglm2
from datasets import Dataset
#
dataset = [preprocess_chatglm2(tokenizer, model.config, item, max_seq_length=256) for item in datas]
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
# #
# for item in datas:
#     text = f"问:{item['context']}\n答:"
#     ids = tokenizer.encode(text, return_tensors='pt')
#     outs = model.generate(input_ids=ids, max_length=128)
#     print(tokenizer.batch_decode(outs)[0])


# input_text = "假设你现在是一个游戏策划，现在你要设计一款生存类的RPG游戏，请帮忙设计这款游戏的世界机制："
# #
# ids = tokenizer.encode(input_text, return_tensors='pt')
#
# outs = model.generate(input_ids=ids, max_length=128)
#
# response = tokenizer.batch_decode(outs)
# print(response[0])
