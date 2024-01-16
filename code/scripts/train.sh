#!/bin/bash

deepspeed --include localhost:0,1,2,3 --master_addr 127.0.0.1 --master_port 28048 train_sft.py \
    --model openllama_peft \
    --stage 1\
    --imagebind_ckpt_path ../pretrained_ckpt/imagebind_ckpt/\
    --vicuna_ckpt_path ../pretrained_ckpt/vicuna_ckpt/7b_v0/\
    --max_tgt_len 1024\
    --save_path  ./ckpt/pandagpt_7b_v0_peft/\
    --log_path ./ckpt/pandagpt_7b_v0_peft/log_rest/
