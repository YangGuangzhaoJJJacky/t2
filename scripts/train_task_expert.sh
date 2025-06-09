#!/bin/bash
export HYDRA_FULL_ERROR=1
export HF_HOME=/mnt/data-raid/yangguangzhao/.cache
export PYTHONPATH=$PYTHONPATH:/home/yangguangzhao/t2/evaluation

# ------- 动态读取传入的参数 -------
NODE=${1:-0}  # 默认值为 0（如果未提供参数）
TASK="aqua_rat"
NUM_ITERS=4

# ------- 找 checkpoint -------
MATCHED_CKPT=$(ls results/$NODE/*/policy_params.pt 2>/dev/null | head -n 1)

if [ -z "$MATCHED_CKPT" ]; then
  load_ckpt=None
else
  load_ckpt=$MATCHED_CKPT
fi

# ------- 启动训练 -------
CUDA_VISIBLE_DEVICES=0,1  python svd_reinforce_hydra.py \
    base_model@_global_=qwen306b \
    task@_global_=$TASK \
    mode@_global_=training \
    task_loader.node=$NODE \
    +output_path="results/$NODE" \
    num_iters=$NUM_ITERS \
    load_ckpt="$load_ckpt"