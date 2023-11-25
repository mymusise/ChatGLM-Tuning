# ChatGLM-Tuning

一种平价的chatgpt实现方案，基于清华的 [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B) + LoRA 进行finetune.

数据集: [alpaca](https://github.com/tatsu-lab/stanford_alpaca)

有colab的同学可以直接在colab上尝试： <a href="https://colab.research.google.com/github/mymusise/ChatGLM-Tuning/blob/master/examples/finetune.ipynb">
        <img alt="Build" src="https://colab.research.google.com/assets/colab-badge.svg">
    </a>


[官方ptuning代码](https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning)


## Demo

- [开源版的文心一言](https://github.com/visual-openllm/visual-openllm)


## S1 Finetune

### 准备

- 显卡: 显存 >= 16G (最好24G或者以上)
- 环境：
- - python>=3.8
- - cuda>=11.6, cupti, cuDNN, TensorRT等深度学习环境
- - pip3 install -r requirements.txt
其中requirements.txt中的安装包bitsandbytes 建议安装0.41.2.post2这个版本，以前的版本可能会提示报错：
        bitsandbytes/libbitsandbytes_cpu.so: undefined symbol: cget_col_row_stats

### 数据预处理


转化alpaca数据集为jsonl

```bash
python cover_alpaca2jsonl.py \
    --data_path data/alpaca_data.json \
    --save_path data/alpaca_data.jsonl \
    
```

tokenization

```bash
python tokenize_dataset_rows.py \
    --jsonl_path data/alpaca_data.jsonl \
    --save_path data/alpaca \
    --max_seq_length 200 \ 
    --skip_overlength  False
    --chatglm_path model_path/chatglm
    --version v1                 
    
```

- `--jsonl_path` 微调的数据路径, 格式jsonl, 对每行的['context']和['target']字段进行encode
- `--save_path` 输出路径
- `--max_seq_length` 样本的最大长度
- `--chatglm_path` 导入模型的路径（可以选择chatglm或chatglm2的不同路径）
- `--version` 模型的版本（v1指chatglm,v2指chatglm2）

### 训练

```bash
python finetune.py \
    --dataset_path data/alpaca \
    --lora_rank 8 \
    --per_device_train_batch_size 6 \
    --gradient_accumulation_steps 1 \
    --max_steps 52000 \
    --save_steps 1000 \
    --save_total_limit 2 \
    --learning_rate 1e-4 \
    --fp16 \
    --remove_unused_columns false \
    --logging_steps 50 \
    --output_dir output
    --chatglm_path model_path/chat_glm
```

### 推理

参考 [infer.ipynb](infer.ipynb)

<details><summary><b>Finetune前后对比</b></summary>

利用Alpaca数据集合对ChatGLM-6B Finetune后，在Alpaca数据集上表现得更好:
- `Answer:` 是模型的输出
- `#### Answer:` 是原答案
![](https://user-images.githubusercontent.com/6883957/226977555-c00c796f-4fdb-4613-810a-8b9a6068bb1b.jpeg)


</details>

### 部署

文件夹下`web_demo.py`，`web_demo2.py`与`api.py`的使用方法与官方的一致。

使用前需要修改`peft_path`为你自己训练的模型路径，修改`peft_config`中的`r`与你训练时的`--lora_rank`一致。


## S2. Reward Model

## S3. PPO


## LoRA

| LoRA                                  | Dataset      |
| ------------------------------------- | ------------ |
| mymusise/chatglm-6b-alpaca-lora       | Alpaca       |
| mymusise/chatglm-6b-alpaca-zh-en-lora | Alpaca-zh-en |
| *(on the way)*                        | Alpaca-zh    |

### 使用预训练好的LoRA

参考 [examples/infer_pretrain.ipynb](https://colab.research.google.com/github/mymusise/ChatGLM-Tuning/blob/master/examples/infer_pretrain.ipynb)


# TODO:

- ~~bs > 1 support~~
- 使用中文数据
- 加入RLHF