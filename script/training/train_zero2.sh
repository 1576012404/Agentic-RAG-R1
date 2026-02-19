#!/bin/bash

set -ex

GPUs="0,1"

# export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=$GPUs
export NCCL_P2P_LEVEL=NVL
export CUDA_LAUNCH_BLOCKING=1
export EVAL_LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export EVAL_LLM_API_KEY="sk-d06d58564b4e42e99c0b8e2566289867"
export EVAL_LLM_MODEL_NAME="qwen-plus"

number_of_gpus=$(echo $GPUs | tr ',' '\n' | wc -l)

accelerate launch \
    --config_file ./src/config/accelerate_config/train_zero2.yaml \
    --main_process_port 12347 \
    --num_processes $number_of_gpus \
    --mixed_precision "bf16" \
    ./src/train.py