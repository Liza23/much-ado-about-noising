#!/bin/bash
#SBATCH --job-name=libero-bowl-plate
#SBATCH --output=logs/libero_goal_bowl_on_plate_%j.out
#SBATCH --error=logs/libero_goal_bowl_on_plate_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
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
    task=libero_goal_bowl_on_plate_image \
    network.rgb_model_name=resnet50 \
    optimization.gradient_steps=300000 \
    optimization.batch_size=128 \
    optimization.loss_type=flow \
    optimization.weight_decay=1e-4 \
    log.eval_episodes=20 \
    log.eval_freq=25000 \
    log.eval_nsteps=9 \
    log.save_freq=25000 \
    log.project=libero-experiments \
    log.group=libero-goal-single-task \
    log.exp_name=bowl_on_plate_resnet50_bs128 \
    log.log_dir="/data/user_data/ldahiya/libero-train/libero-goal-single-task/bowl_on_plate_resnet50_bs128" \
    task.num_envs=5 \
    optimization.auto_resume=true \
    log.save_video=false
