#!/bin/bash
#
# LoRA 训练脚本 - 与 SVD 对比实验用
# 使用方法: bash scripts/train_task_expert_lora.sh [NODE_ID]
#

export HYDRA_FULL_ERROR=1
export HF_HOME=/mnt/data-raid/yangguangzhao/.cache
export PYTHONPATH=$PYTHONPATH:/home/yangguangzhao/t2/evaluation

# ------- 动态读取传入的参数 -------
NODE=${1:-0}  # 默认值为 0（如果未提供参数）

# ------- 任务配置 -------
# TASK="cls"
# NUM_ITERS=3
TASK="aqua_rat"
NUM_ITERS=3

# ------- LoRA 超参数 -------
LORA_RANK=8      # LoRA 秩，可选: 4, 8, 16, 32
LORA_ALPHA=1    # LoRA 缩放因子，通常是 rank 的 2 倍
LORA_DROPOUT=0.5 # Dropout 率
TRAIN_LAYERS="mlp"  # 🔥 训练层选择: "mlp", "self_attn", "both"

# ------- 找 checkpoint -------
MATCHED_CKPT=$(ls results/$NODE/*/policy_params_latest.pt 2>/dev/null | head -n 1)

if [ -z "$MATCHED_CKPT" ]; then
  load_ckpt=None
else
  load_ckpt=$MATCHED_CKPT
fi

echo "=========================================="
echo "🚀 LoRA 训练实验"
echo "=========================================="
echo "节点: $NODE"
echo "任务: $TASK"
echo "迭代次数: $NUM_ITERS"
echo "LoRA Rank: $LORA_RANK"
echo "LoRA Alpha: $LORA_ALPHA"
echo "训练层: $TRAIN_LAYERS"
echo "Checkpoint: $load_ckpt"
echo "=========================================="

# ------- 启动训练 -------
CUDA_VISIBLE_DEVICES=0,1 python svd_reinforce_hydra.py \
    policy@_global_=lora \
    base_model@_global_=qwen306b \
    task@_global_=$TASK \
    mode@_global_=training \
    optimization@_global_=reinforce \
    task_loader.node=$NODE \
    lora_rank=$LORA_RANK \
    lora_alpha=$LORA_ALPHA \
    lora_dropout=$LORA_DROPOUT \
    train_layers=$TRAIN_LAYERS \
    +output_path="results/$NODE" \
    num_iters=$NUM_ITERS \
    load_ckpt="$load_ckpt" \
    exp_suffix="lora_r${LORA_RANK}_${TRAIN_LAYERS}_a${LORA_ALPHA}"

