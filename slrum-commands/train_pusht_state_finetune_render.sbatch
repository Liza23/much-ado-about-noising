#!/bin/bash
#SBATCH -p general
#SBATCH --job-name=pusht-state-finetune-render
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# --- setup ---
mkdir -p logs
hostname
date
nvidia-smi -L

# --- env ---
source ~/.bashrc
export UV_PROJECT_ENVIRONMENT=.venv

# --- sanity ---
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
uv run python - << 'EOF'
import torch
print("CUDA available:", torch.cuda.is_available())
print("Num GPUs:", torch.cuda.device_count())
EOF

# --- launch ---
cd /home/ldahiya/max_vla/much-ado-about-noising

# Finetune state ChiUNet (100% success) with render augmentation
# Base: MLP(state) → ChiUNet [loaded from ckpt, encoder MLP matches]
# New:  draft → render → ResNet18(future_img) → fusion → refined action
# Note: future_encoder (ResNet18) and fusion start from random init
uv run python examples/train_pusht.py \
    task=pusht_state_render \
    network=chiunet_render \
    optimization=vla_mip_policy \
    optimization.pretrained_ckpt="ckpt/pusht_ph_state_mip_chiunet_256_seed1_success100.pt" \
    optimization.auto_resume=false \
    optimization.batch_size=32 \
    log.log_dir="/data/user_data/ldahiya/pusht_baselines/state_finetune_render" \
    log.project="pusht_baselines" \
    log.exp_name="pusht-state-finetune-render" \
    log.log_freq=1000 \
    log.eval_freq=5000 \
    log.save_freq=10000
