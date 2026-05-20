#!/bin/bash
#SBATCH -p general
#SBATCH --job-name=vla-mip-agent
#SBATCH --gres=gpu:1
#SBATCH -c 16
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

# VLA-MIP with agent-only renderer
# Draft head = MIP step 1, Decoder = MIP step 2
# No state in decoder (avoids shortcut)
uv run python examples/train_pusht.py \
    task=pusht_vla_agent_only \
    network=vla_crossattn \
    optimization=vla_mip_policy \
    optimization.auto_resume=false \
    optimization.batch_size=32 \
    network.use_state_in_decoder=false \
    log.log_dir="/data/user_data/ldahiya/0217_vla_mip/mip_render" \
    log.project="0217_vla_mip" \
    log.exp_name="vla-mip-agent-render"
