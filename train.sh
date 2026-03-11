#!/bin/bash

# Usage: bash train.sh <task> <gpu_ids>
# Example:
#   bash train.sh uhdblur 0,1
#   bash train.sh uhdrain 2,3
#   bash train.sh uhdhaze 4,5
#   bash train.sh uhdlol 6,7

TASK=$1
GPU_IDS=${2:-"0,1"}

NUM_GPUS=$(echo $GPU_IDS | tr ',' '\n' | wc -l)

if [ -z "$TASK" ]; then
    echo "Usage: bash train.sh <task> <gpu_ids>"
    echo "Available tasks: uhdblur, uhdrain, uhdhaze, uhdlol"
    exit 1
fi

CONFIG="options/train_C2SSM_${TASK}.yml"

if [ ! -f "$CONFIG" ]; then
    echo "Config file not found: $CONFIG"
    exit 1
fi

echo "Training C2SSM on task: $TASK"
echo "Using GPUs: $GPU_IDS (${NUM_GPUS} GPUs)"
echo "Config: $CONFIG"

CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS \
    --master_port=4337 \
    basicsr/train.py -opt $CONFIG --launcher pytorch
