#!/bin/bash
#SBATCH -p general
#SBATCH --job-name=pusht-mip-render
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

# Render-augmented MIP baseline: ResNet18 + ChiUNet + agent-only rendering
# Step 1: ResNet18(curr_img) → obs_emb → ChiUNet → draft action
#         render draft[:, 0, :] → future_img
#         ResNet18(future_img) → future_emb → fusion(obs_emb, future_emb)
# Step 2: ChiUNet(noisy_act, enhanced_obs_emb) → refined action
# Ablation: same arch as pusht image chiunet baseline, only adds rendering
uv run python examples/train_pusht.py \
    task=pusht_mip_render \
    network=chiunet_render \
    optimization=vla_mip_policy \
    optimization.auto_resume=false \
    optimization.batch_size=32 \
    log.log_dir="/data/user_data/ldahiya/pusht_baselines/agent_renderer" \
    log.project="pusht_baselines" \
    log.exp_name="pusht-agent-renderer"
