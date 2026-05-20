#!/bin/bash
#SBATCH -p general
#SBATCH --job-name=pusht-state
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

# PushT state-based policy (obs: [agent_x, agent_y, block_x, block_y, block_angle])
uv run python examples/train_pusht.py \
    task=pusht_state \
    log.log_dir="/data/user_data/ldahiya/pusht_baselines/state" \
    log.project="pusht_baselines" \
    log.exp_name="pusht-state" \
    log.log_freq=1000 \
    log.eval_freq=5000 \
    log.save_freq=10000
