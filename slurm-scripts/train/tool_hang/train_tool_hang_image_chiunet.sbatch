#!/bin/bash
#SBATCH -p general
#SBATCH --job-name=tool-hang-image-chiunet
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

# tool_hang image-based policy with ChiUNet + MIP loss
# num_envs=4 (set in task config, lower due to MuJoCo rendering overhead)
uv run python examples/train_robomimic.py \
    task=tool_hang_ph_image \
    network=chiunet \
    optimization=vla_mip_policy \
    log.log_dir="/data/user_data/ldahiya/tool_hang_baselines/image_chiunet" \
    log.project="tool_hang_baselines" \
    log.exp_name="tool-hang-image-chiunet" \
    log.log_freq=1000 \
    log.eval_freq=5000 \
    log.save_freq=10000
