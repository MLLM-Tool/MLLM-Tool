#!/bin/bash

deepspeed --include localhost:2,3 --master_addr 127.0.0.1 --master_port 28048 train_sft.py \
    --model openllama_peft \
    --stage 1\
    --imagebind_ckpt_path ../pretrained_ckpt/imagebind_ckpt/\
    --vicuna_ckpt_path /inspurfs/group/gaoshh/wangchy/LLM/vicuna-7b-v1.5/snapshots/de56c35b1763eaae20f4d60efd64af0a9091ebe5\
    --max_tgt_len 1024\
    --save_path  ./ckpt/toollmm_vicuna_7b_v1_github_peft/\
    --log_path ./ckpt/toollmm_vicuna_7b_v1_github_peft/log_rest/
