#!/bin/bash
#SBATCH -p general
#SBATCH --job-name=tool-hang-image-render-ft
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 48:00:00
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
export MUJOCO_GL=egl

# --- sanity ---
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
uv run python - << 'EOF'
import torch
print("CUDA available:", torch.cuda.is_available())
print("Num GPUs:", torch.cuda.device_count())
EOF

# --- launch ---
cd /home/ldahiya/max_vla/much-ado-about-noising

# Finetune image ChiUNet (~0.8 success) with render augmentation.
# Base ChiUNet weights loaded via key remap: net.X -> net.base_network.X
# draft_head, future_encoder, fusion start from random init (not in base ckpt).
# Optimizer is reset (fresh finetune from step 0).
uv run python examples/train_robomimic.py \
    task=tool_hang_ph_image_render \
    network=chiunet_render \
    optimization=vla_mip_policy \
    optimization.pretrained_ckpt="/home/ldahiya/max_vla/much-ado-about-noising/ckpt/tool_hang_ph_image_mip_chiunet_256_seed1_success64.pt" \
    optimization.auto_resume=false \
    log.log_dir="/data/user_data/ldahiya/tool_hang_baselines/image_render_finetune" \
    log.project="tool_hang_baselines" \
    log.exp_name="tool-hang-image-render-finetune" \
    log.log_freq=1000 \
    log.eval_freq=5000 \
    log.save_freq=10000
