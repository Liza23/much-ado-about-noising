#!/bin/bash
#SBATCH -p general
#SBATCH --job-name=vla-norender
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

NGPUS=1
MASTER_PORT=$(shuf -i 29500-29999 -n 1)

# --- sanity ---
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
uv run python - << 'EOF'
import torch
print("CUDA available:", torch.cuda.is_available())
print("Num GPUs:", torch.cuda.device_count())
EOF

# --- launch ---
cd /home/ldahiya/max_vla/much-ado-about-noising

# VLA without rendering: current image cross-attention only
# Future image branch gets zeros (no env stepping overhead)
# batch_size is PER GPU, effective batch = batch_size * NGPUS
# uv run torchrun --nnodes=1 --nproc_per_node=$NGPUS --master_port=$MASTER_PORT examples/train_pusht.py \
#     task=pusht_vla_crossattn \
#     network=vla_crossattn \
#     optimization=geometric_policy \
#     optimization.auto_resume=false \
#     task.use_image_renderer=false \
#     optimization.batch_size=256 \
#     optimization.gradient_steps=200000 \
#     log.log_dir="/data/user_data/ldahiya/vla_norender" \
#     log.project="vla_pusht" \
#     log.exp_name="vla-norender" \
#     log.log_freq=500 \
#     log.eval_freq=5000 \
#     log.save_freq=10000

uv run python examples/train_pusht.py \
    task=pusht_vla_crossattn \
    network=vla_crossattn \
    optimization=geometric_policy \
    optimization.auto_resume=false \
    task.use_image_renderer=false \
    optimization.gradient_steps=500000 \
    log.log_dir="/data/user_data/ldahiya/0213_pusht_max_vla_corrected/norender_eval" \
    log.project="0213_max_vla_pusht" \
    log.exp_name="vla-norender" \
    log.log_freq=1000 \
    log.eval_freq=10000 \
    log.save_freq=10000
