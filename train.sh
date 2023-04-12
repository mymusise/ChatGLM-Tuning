#!/bin/bash
nohup python finetune.py --dataset_path ./data-taobao/AdvertiseGen/tokenized-data --lora_rank 8 --per_device_train_batch_size 1 --gradient_accumulation_steps 16 \
--max_steps 25000 --save_steps 500 --save_total_limit 2 --learning_rate 1e-4 --fp16 --remove_unused_columns false --logging_steps 50 --output_dir output_taobao \
> data-taobao/041101.log 2>&1 &