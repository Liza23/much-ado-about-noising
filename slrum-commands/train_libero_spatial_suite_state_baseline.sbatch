#!/bin/bash
#SBATCH --job-name=libero-spatial-suite-base
#SBATCH --output=logs/libero_spatial_suite_state_baseline_%j.out
#SBATCH --error=logs/libero_spatial_suite_state_baseline_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --requeue
#SBATCH --time=48:00:00

set -euo pipefail

mkdir -p logs
source /home/ldahiya/max_vla/much-ado-about-noising/.venv/bin/activate
cd /home/ldahiya/max_vla/much-ado-about-noising

export NUMBA_CACHE_DIR="$(mktemp -d /tmp/numba_${SLURM_JOB_ID:-manual}_XXXXXX)"
export MPLCONFIGDIR="$(mktemp -d /tmp/mpl_${SLURM_JOB_ID:-manual}_XXXXXX)"
export XDG_RUNTIME_DIR="$(mktemp -d /tmp/xdg_${SLURM_JOB_ID:-manual}_XXXXXX)"
export MUJOCO_GL=egl
trap 'rm -rf "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_RUNTIME_DIR"' EXIT

python examples/train_libero.py \
    task=libero_spatial_suite_state \
    optimization.gradient_steps=1000000 \
    optimization.batch_size=256 \
    optimization.loss_type=flow \
    optimization.weight_decay=1e-4 \
    log.eval_episodes=50 \
    log.eval_freq=20000 \
    log.save_freq=100000 \
    log.project=libero-experiments \
    log.group=libero-spatial-suite-state \
    log.exp_name=baseline \
    log.log_dir="/data/user_data/ldahiya/libero-train/libero-spatial-suite-state/baseline" \
    task.num_envs=5 \
    optimization.auto_resume=true \
    log.save_video=false
