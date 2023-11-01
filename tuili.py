from transformers import AutoModel, AutoTokenizer

output_path = '/home/adminz/ChatGLM-Tuning/output'
model_path = '/home/adminz/ChatGLM-Tuning/model_path/chat2glm'



tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

model = AutoModel.from_pretrained(model_path, trust_remote_code=True, load_in_8bit=True, device_map='auto')



from peft import PeftModel

model = PeftModel.from_pretrained(model, output_path)

# datas = [
# {
#     "context": "大师，我最近事业不顺利，感觉很迷茫，该怎么办？",
#     "target": "孩子，生活中的起伏和挫折是常态。放下烦恼，用平静的心态面对困境，努力奋斗，同时保持内心的善良与慈悲，你将找到属于你的光明之路。",
# },
# {
#     "context": "师父，我与家人之间常常发生争吵，如何改善家庭关系？",
#     "target": "阿弥陀佛，亲情是一份宝贵的财富，要从宽容和理解开始。培养平和的心态，倾听彼此的意见，以慈悲与智慧去化解冲突，共同创造和谐温暖的家庭氛围。",
# },
# {
#     "context": "师傅，我经常感到焦虑和压力，怎样才能获得内心的宁静？",
#     "target": "孩子，内心的宁静源于觉察和自我调适。修行冥想，深呼吸，寻找内心的栖息地。明白外在的变幻无常，放下过多的执着与欲望，你将发现内心的平静与安宁。",
# },
# {
#     "context": "大师，我常常陷入烦恼和抱怨，怎样才能培养感恩之心？",
#     "target": "施主，感恩是一种修养，需要从日常小事开始。每天意识到自己的幸运和获得，体察他人的善意和关爱。珍惜当下，感恩生活中的点滴，你将发现更多的快乐与满足。",
# },
# {
#     "context": "师父，我对未来感到迷茫和恐惧，不知道自己的人生目标是什么，有何建议？",
#     "target": "阿弥陀佛，人生的方向不一定要一目了然，它是通过不断尝试和探索而逐渐清晰的。保持对未知的勇敢与开放，寻找自己内心的声音。用心去体验，用智慧去选择，你将找到属于自己的道路。",
# }]


import torch
import json

answers = []
from cover_alpaca2jsonl import format_example

instructions = json.load(open("/home/adminz/ChatGLM-Tuning/data/alpaca_data.json"))

with torch.no_grad():
    for idx, item in enumerate(instructions[:3]):
        feature = format_example(item)
        input_text = feature['context']
        ids = tokenizer.encode(input_text)
        input_ids = torch.LongTensor([ids])
        out = model.generate(
            input_ids=input_ids,
            max_length=150,
            do_sample=False,
            temperature=0
        )
        out_text = tokenizer.decode(out[0])
        answer = out_text.replace(input_text, "").replace("\nEND", "").strip()
        item['infer_answer'] = answer
        print(out_text)
        print(f"### {idx+1}.Answer:\n", item.get('output'), '\n\n')
        answers.append({'index': idx, **item})