#!/bin/bash
#SBATCH --job-name=oracle-rollout-10
#SBATCH --output=logs/eval_oracle_intent_rollout_libero_10_%j.out
#SBATCH --error=logs/eval_oracle_intent_rollout_libero_10_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00

set -euo pipefail

mkdir -p logs
source /home/ldahiya/max_vla/much-ado-about-noising/.venv/bin/activate
cd /home/ldahiya/max_vla/much-ado-about-noising

export NUMBA_CACHE_DIR="$(mktemp -d /tmp/numba_${SLURM_JOB_ID:-manual}_XXXXXX)"
export MPLCONFIGDIR="$(mktemp -d /tmp/mpl_${SLURM_JOB_ID:-manual}_XXXXXX)"
export XDG_RUNTIME_DIR="$(mktemp -d /tmp/xdg_${SLURM_JOB_ID:-manual}_XXXXXX)"
export MUJOCO_GL=egl
trap 'rm -rf "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_RUNTIME_DIR"' EXIT

python examples/eval_oracle_intent_rollout.py \
    task=libero_10_suite_image_flow_intent \
    network=mlp_flow_intent \
    optimization.loss_type=flow \
    optimization.model_path="/data/user_data/ldahiya/libero-train/libero-10-suite-image/flow_intent_curriculum/models/model_best.pt" \
    ++eval_oracle.num_episodes=20 \
    ++eval_oracle.num_steps=9 \
    log.wandb_mode=disabled \
    log.log_dir="/tmp/oracle_rollout_libero10"
