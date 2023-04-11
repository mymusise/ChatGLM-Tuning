#!/bin/bash
nohup python finetune.py --dataset_path data/belle --lora_rank 8 --per_device_train_batch_size 2 --gradient_accumulation_steps 4 \
--max_steps 250000 --save_steps 5000 --save_total_limit 2 --learning_rate 1e-4 --fp16 --remove_unused_columns false --logging_steps 500 --output_dir output_belle \
> 033101.log 2>&1 &