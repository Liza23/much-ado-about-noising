#!/bin/bash
#SBATCH -p general
#SBATCH --job-name=vla-mip-norender
#SBATCH --gres=gpu:1
#SBATCH -c 32
#SBATCH --mem=128G
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

# VLA-MIP without renderer
# No draft head (no renderer), no state in decoder
# Decoder gets only current image tokens
# vla_mip degrades gracefully to single-step refinement without renderer
uv run python examples/train_pusht.py \
    task=pusht_vla_crossattn \
    network=vla_crossattn \
    optimization=vla_mip_policy \
    optimization.auto_resume=false \
    task.use_image_renderer=false \
    network.use_state_in_decoder=false \
    optimization.gradient_steps=500000 \
    log.log_dir="/data/user_data/ldahiya/0217_vla_mip/norender" \
    log.project="0217_vla_mip" \
    log.exp_name="vla-mip-norender" \
    log.log_freq=1000 \
    log.eval_freq=10000 \
    log.save_freq=10000
