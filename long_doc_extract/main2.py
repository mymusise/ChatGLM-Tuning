import os
import json

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("/work/models/chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("/work/models/chatglm-6b", trust_remote_code=True).half().cuda()

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


if __name__ == '__main__':
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