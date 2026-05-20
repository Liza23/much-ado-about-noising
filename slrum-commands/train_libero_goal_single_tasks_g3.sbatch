#!/bin/bash
#SBATCH --job-name=libero-st-g3
#SBATCH --output=logs/libero_goal_single_tasks_g3_%j.out
#SBATCH --error=logs/libero_goal_single_tasks_g3_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --requeue
#SBATCH --time=48:00:00

# Group 3: wine_bottle_on_cabinet | turn_on_stove
# ~8h/task x 2 tasks = ~16h

set -euo pipefail

mkdir -p logs
source /home/ldahiya/max_vla/much-ado-about-noising/.venv/bin/activate
cd /home/ldahiya/max_vla/much-ado-about-noising

export NUMBA_CACHE_DIR="$(mktemp -d /tmp/numba_${SLURM_JOB_ID:-manual}_XXXXXX)"
export MPLCONFIGDIR="$(mktemp -d /tmp/mpl_${SLURM_JOB_ID:-manual}_XXXXXX)"
export XDG_RUNTIME_DIR="$(mktemp -d /tmp/xdg_${SLURM_JOB_ID:-manual}_XXXXXX)"
export MUJOCO_GL=egl
trap 'rm -rf "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_RUNTIME_DIR"' EXIT

COMMON_ARGS="
    network.rgb_model_name=resnet50
    optimization.gradient_steps=100000
    optimization.batch_size=128
    optimization.loss_type=flow
    optimization.weight_decay=1e-4
    optimization.auto_resume=false
    log.eval_episodes=20
    log.eval_freq=20000
    log.eval_nsteps=9
    log.save_freq=100000
    log.project=libero-experiments
    log.group=libero-goal-single-task
    log.save_video=false
    task.num_envs=5
"

echo "=== [$(date)] task8: wine_bottle_on_cabinet ==="
python examples/train_libero.py \
    task=libero_goal_wine_bottle_on_cabinet_image \
    log.exp_name=wine_bottle_on_cabinet_resnet50_bs128 \
    log.log_dir="/data/user_data/ldahiya/libero-train/libero-goal-single-task/wine_bottle_on_cabinet_resnet50_bs128" \
    $COMMON_ARGS

echo "=== [$(date)] task9: turn_on_stove ==="
python examples/train_libero.py \
    task=libero_goal_turn_on_stove_image \
    log.exp_name=turn_on_stove_resnet50_bs128 \
    log.log_dir="/data/user_data/ldahiya/libero-train/libero-goal-single-task/turn_on_stove_resnet50_bs128" \
    $COMMON_ARGS

echo "=== [$(date)] Group 3 complete ==="
