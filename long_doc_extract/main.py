import os
import sys, json
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from modeling_chatglm import ChatGLMForConditionalGeneration
import torch


torch.set_default_tensor_type(torch.cuda.HalfTensor)
model = ChatGLMForConditionalGeneration.from_pretrained("/work/models/chatglm-6b", trust_remote_code=True, device_map='auto')



from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/work/models/chatglm-6b", trust_remote_code=True)

src = '在即将举办的第十一届广州国际汽车展览会上，豪华运动型轿车全新英菲尼迪Q50 2.0T车型将全球首发，助力天才车手维特尔叱咤2013赛季F1赛场成功卫冕“四冠王”称号的冠军战车RB9也将震撼登临，共同奉上一场精彩绝伦的视觉盛宴。此外，创造2013收视率奇迹的《爸爸去哪儿》节目重量级嘉宾更是会神秘亮相，讲述节目、嘉宾与明星家庭专属座驾英菲尼迪JX的温馨情缘。作为同级别最前沿的豪华运动型轿车，全新英菲尼迪Q50集智能、乐趣和感官体验于一身，完美迎合了“年轻心态高端消费者”的全方位需求。此次车展，搭载全新动力总成的英菲尼迪Q50 2.0T车型将世界首演，它专为中国市场量身打造，也是英菲尼迪首款搭载涡轮增压发动机的车型。英菲尼迪Q50 2.0T车型不仅拥有领先同级别其它车型的极佳加速表现，在燃油经济性上也表现优异，完美地实现了高性能和低油耗的最佳平衡，将助推英菲尼迪Q50成为同级别最具竞争力的车型。\n'


from peft import get_peft_model, LoraConfig, TaskType

peft_path = "../output_belle/checkpoint-15000/adapter_model.bin"

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=True,
    r=8,
    lora_alpha=32, lora_dropout=0.1
)

model = get_peft_model(model, peft_config)
model.load_state_dict(torch.load(peft_path), strict=False)
torch.set_default_tensor_type(torch.cuda.FloatTensor)



# import json
#
# instructions = json.load(open("data/alpaca_data.json"))
#
# answers = []
# from cover_alpaca2jsonl import format_example
#
# with torch.no_grad():
#     for idx, item in enumerate(instructions[:3]):
#         feature = format_example(item)
#         input_text = feature['context']
#         ids = tokenizer.encode(input_text)
#         input_ids = torch.LongTensor([ids])
#         out = model.generate(
#             input_ids=input_ids,
#             max_length=150,
#             do_sample=False,
#             temperature=0
#         )
#         out_text = tokenizer.decode(out[0])
#         answer = out_text.replace(input_text, "").replace("\nEND", "").strip()
#         item['infer_answer'] = answer
#         print(out_text)
#         print(f"### {idx+1}.Answer:\n", item.get('output'), '\n\n')
#         answers.append({'index': idx, **item})


def generate_response(input_text):
    ids = tokenizer.encode(input_text)
    input_ids = torch.LongTensor([ids])
    out = model.generate(
        input_ids=input_ids,
        max_length=1000,
        do_sample=False,
        temperature=0
    )
    out_text = tokenizer.decode(out[0])
    answer = out_text.replace(input_text, "").replace("\nEND", "").strip()
    # item['infer_answer'] = answer
    print(answer)
    # print(f"### {idx + 1}.Answer:\n", item.get('output'), '\n\n')
    # answers.append({'index': idx, **item})

def extract_pages(json_data):
    node = json_data['node']
    json_page_result = json_data['json_page_result']
    assert len(node) == len(json_page_result)

    pages = []
    for page_result, page_node in zip(json_page_result, node):
        text_lines = page_result['text_lines']
        content = page_node['content']
        page_text = ''
        for text_line in text_lines:
            line = ''
            for char_id in text_line['char_id_list']:
                index = int(char_id.split('#')[-1])
                char = content[index]['str']
                line += char
            page_text += line
        pages.append(page_text)
    return pages


data_dir = './data2'

for root, dirs, files in os.walk(data_dir):
    for file in files:
        file_path = os.path.join(root, file)
        print(f'\n\n\n\nfile_path: {file_path}')
        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        pages = extract_pages(json_data)

        prompt = 'test'
        response, history = model.chat(tokenizer, prompt, history=None, top_p=0.8, temperature=0.01)
        print(f'抽取结果：\n{response}')

        print(model.chat(tokenizer, prompt, history=None, top_p=0.8, temperature=0.01)[0])

generate_response(src)