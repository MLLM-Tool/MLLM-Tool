#!/bin/bash

deepspeed --include localhost:0,1,2,3 --master_addr 127.0.0.1 --master_port 28040 train_sft.py \
    --model openllama_peft \
    --stage 1\
    --imagebind_ckpt_path ../pretrained_ckpt/imagebind_ckpt/\
    --vicuna_ckpt_path /inspurfs/group/gaoshh/wangchy/LLM/Llama-2-13b-hf/snapshots/99afe33d7eaa87c7fc6ea2594a0e4e7e588ee0a4\
    --max_tgt_len 1024\
    --save_path  ./ckpt/toollmm_llama2_13b_v3_peft/\
    --log_path ./ckpt/toollmm_llama2_13b_v3_peft/log_rest/
