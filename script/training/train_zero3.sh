#!/bin/bash

export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=0
export NCCL_P2P_LEVEL=NVL
export CUDA_LAUNCH_BLOCKING=1

#check
export EVAL_LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export EVAL_LLM_API_KEY="sk-d06d58564b4e42e99c0b8e2566289867"
export EVAL_LLM_MODEL_NAME="qwen-plus"

accelerate launch \
    --config_file ./src/config/accelerate_config/train_zero3.yaml \
    --main_process_port 12347 \
    --num_processes 2 \
    --mixed_precision "fp16" \
    ./src/train.py