#!/bin/bash
#SBATCH --job-name=debug-eval-spa
#SBATCH --output=logs/debug_residual_eval_spatial_%j.out
#SBATCH --error=logs/debug_residual_eval_spatial_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00

set -euo pipefail

mkdir -p logs
source /home/ldahiya/max_vla/much-ado-about-noising/.venv/bin/activate
cd /home/ldahiya/max_vla/much-ado-about-noising

export NUMBA_CACHE_DIR="$(mktemp -d /tmp/numba_${SLURM_JOB_ID:-manual}_XXXXXX)"
export MPLCONFIGDIR="$(mktemp -d /tmp/mpl_${SLURM_JOB_ID:-manual}_XXXXXX)"
export XDG_RUNTIME_DIR="$(mktemp -d /tmp/xdg_${SLURM_JOB_ID:-manual}_XXXXXX)"
export MUJOCO_GL=egl
trap 'rm -rf "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_RUNTIME_DIR"' EXIT

BASE_CKPT="/data/user_data/ldahiya/libero-train/libero-spatial-suite-image/baseline/models/model_latest.pt"

# Short episodes (max 64 sim steps) and 5 eval episodes (1 batch per task).
# Full eval: 10 tasks × 1 batch × 5 envs × 64 steps = 3200 sim steps total.
# At ~1 sim step/sec for image, should take ~1 hr. Hangs should surface early.
python examples/debug_residual_eval.py \
    task=libero_spatial_suite_image \
    residual_parl.flow_intent_ckpt="${BASE_CKPT}" \
    +residual_parl.flow_num_steps=9 \
    task.max_episode_steps=64 \
    task.num_envs=5 \
    log.eval_episodes=5 \
    log.project=debug \
    log.group=debug \
    log.exp_name=eval_hang \
    log.log_dir="/tmp/debug_eval_${SLURM_JOB_ID:-manual}" \
    log.save_video=false
