#!/bin/bash

deepspeed --include localhost:0,1 --master_addr 127.0.0.1 --master_port 28070 train_sft.py \
    --model openllama_peft \
    --stage 1\
    --imagebind_ckpt_path ../pretrained_ckpt/imagebind_ckpt/\
    --vicuna_ckpt_path /inspurfs/group/gaoshh/wangchy/LLM/llama-7b/snapshots/8416d3fefb0cb3ff5775a7b13c1692d10ff1aa16/\
    --max_tgt_len 1024\
    --save_path  ./ckpt/toollmm_llama_7b_v0_peft/\
    --log_path ./ckpt/toollmm_llama_7b_v0_peft/log_rest/
