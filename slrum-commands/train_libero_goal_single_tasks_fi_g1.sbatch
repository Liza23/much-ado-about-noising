#!/bin/bash
#SBATCH --job-name=libero-st-fi-g1
#SBATCH --output=logs/libero_goal_single_tasks_fi_g1_%j.out
#SBATCH --error=logs/libero_goal_single_tasks_fi_g1_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --requeue
#SBATCH --time=48:00:00

# Flow-intent Group 1: open_middle_drawer | open_top_drawer_bowl | push_plate_stove

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
    network=mlp_flow_intent
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

echo "=== [$(date)] task1: open_middle_drawer ==="
python examples/train_libero.py \
    task=libero_goal_open_middle_drawer_image_flow_intent \
    log.exp_name=open_middle_drawer_flow_intent_resnet50_bs128 \
    log.log_dir="/data/user_data/ldahiya/libero-train/libero-goal-single-task/open_middle_drawer_flow_intent_resnet50_bs128" \
    $COMMON_ARGS

echo "=== [$(date)] task2: open_top_drawer_bowl ==="
python examples/train_libero.py \
    task=libero_goal_open_top_drawer_bowl_image_flow_intent \
    log.exp_name=open_top_drawer_bowl_flow_intent_resnet50_bs128 \
    log.log_dir="/data/user_data/ldahiya/libero-train/libero-goal-single-task/open_top_drawer_bowl_flow_intent_resnet50_bs128" \
    $COMMON_ARGS

echo "=== [$(date)] task3: push_plate_stove ==="
python examples/train_libero.py \
    task=libero_goal_push_plate_stove_image_flow_intent \
    log.exp_name=push_plate_stove_flow_intent_resnet50_bs128 \
    log.log_dir="/data/user_data/ldahiya/libero-train/libero-goal-single-task/push_plate_stove_flow_intent_resnet50_bs128" \
    $COMMON_ARGS

echo "=== [$(date)] Group fi-g1 complete ==="
