#!/bin/bash
CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --nproc_per_node=2 finetune_ddp.py --dataset_path ./data-renmin/tokenized-data \
--lora_rank 8 --per_device_train_batch_size 8 --gradient_accumulation_steps 8 \
--num_train_epochs 2 --save_steps 500 --learning_rate 1e-4 --fp16 --remove_unused_columns false --logging_steps 10 --output_dir data-renmin/output

#CUDA_VISIBLE_DEVICES=2 python finetune_ddp.py --dataset_path ./data-taobao/AdvertiseGen/tokenized-data \
#--lora_rank 8 --per_device_train_batch_size 1 --gradient_accumulation_steps 2 \
#--max_steps 25000 --save_steps 500 --save_total_limit 2 --learning_rate 1e-4 --fp16 --remove_unused_columns false --logging_steps 50 --output_dir output_taobao