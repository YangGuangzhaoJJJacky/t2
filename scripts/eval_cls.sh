#!/bin/bash
export HYDRA_FULL_ERROR=1
export HF_HOME=/mnt/data-raid/yangguangzhao/.cache
export PYTHONPATH=$PYTHONPATH:/home/yangguangzhao/t2/evaluation

# 读取传入的参数
NODE=${1:-0}
CLS=${2:-0}

TASK="aqua_rat"
NUM_ITERS=0  # 只评估，不训练

# 找对应的checkpoint
CHECKPOINT_PATH="/mnt/data-raid/yangguangzhao/t2/results_history/results_self_cls_math_9/$CLS/aqua_rat_1_mm1_qwen306b_RL-lr0.002-mGN0.001-klC0.01-rrN0CNone-st/policy_params.pt"

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "错误: 找不到checkpoint文件: $CHECKPOINT_PATH"
    exit 1
fi

# 启动评估
CUDA_VISIBLE_DEVICES=0,1 python svd_reinforce_hydra.py \
    base_model@_global_=qwen306b \
    task@_global_=$TASK \
    mode@_global_=training \
    optimization@_global_=reinforce \
    task_loader.node=$NODE \
    +output_path="results/$NODE" \
    num_iters=$NUM_ITERS \
    load_ckpt="$CHECKPOINT_PATH"
