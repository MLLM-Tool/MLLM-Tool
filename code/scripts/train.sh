#!/bin/bash

deepspeed --include localhost:0,1,2,3 --master_addr 127.0.0.1 --master_port 28070 train_sft.py \
    --model openllama_peft \
    --stage 1\
    --imagebind_ckpt_path ../pretrained_checkpoint/imagebind_ckpt/\
    --llm_ckpt_path ../pretrained_ckpt/LLM_ckpt/vicuna_7b/\
    --max_tgt_len 512\
    --epochs 5\
    --save_path  ./ckpt/mllmtool_vicuna_7b/\
    --log_path ./ckpt/mllmtool_vicuna_7b/log/\
    --version v1
