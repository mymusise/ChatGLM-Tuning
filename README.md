# ChatGLM-Tuning

一种平价的chatgpt实现方案，基于清华的 [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B) 进行finetune.

数据集: [alpaca](https://github.com/tatsu-lab/stanford_alpaca)


有colab的同学可以直接在colab上尝试： <a href="https://colab.research.google.com/github/mymusise/ChatGLM-Tuning/blob/master/examples/finetune.ipynb">
        <img alt="Build" src="https://colab.research.google.com/assets/colab-badge.svg">
    </a>


## 准备

- 显卡: 显存 >= 16G (最好24G或者以上)
- 环境：
- - python>=3.8
- - cuda>=11.6, cupti, cuDNN, TensorRT等深度学习环境
- - pip3 install -r requirements.txt


## 数据预处理

```bash
python tokenize_dataset_rows.py \
    --jsonl_path data/alpaca_data.jsonl \
    --save_path data/alpaca \
    --max_seq_length 320
```

- `--jsonl_path` 微调的数据路径, 格式jsonl, 对每行的['text']字段进行encode
- `--save_path` 输出路径
- `--max_seq_length` 样本的最大长度

## 训练

```bash
python finetune.py \
    --dataset_path data/alpaca \
    --lora_rank 8 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --max_steps 52000 \
    --save_steps 1000 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --fp16 \
    --logging_steps 50 \
    --output_dir output
```

# 推理

参考 [infer.ipynb](infer.ipynb)


# TODO:

- ~ bs > 1 support ~
- 使用中文数据
- 加入RLHF