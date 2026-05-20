#!/bin/bash
#SBATCH --job-name=libero-obj-fi-task-id
#SBATCH --output=logs/libero_object_suite_image_flow_intent_task_id_%j.out
#SBATCH --error=logs/libero_object_suite_image_flow_intent_task_id_%j.err
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
    task=libero_object_suite_image_flow_intent_task_id \
    network=mlp_flow_intent \
    network.rgb_model_name=resnet18 \
    optimization.gradient_steps=100000 \
    optimization.batch_size=128 \
    optimization.loss_type=flow \
    optimization.weight_decay=1e-4 \
    optimization.auto_resume=false \
    log.eval_episodes=20 \
    log.eval_freq=25000 \
    log.eval_nsteps=9 \
    log.save_freq=25000 \
    log.project=libero-experiments \
    log.group=libero-object-suite-image \
    log.exp_name=flow_intent_resnet18_bs128_task_id \
    log.log_dir="/data/user_data/ldahiya/libero-train/libero-object-suite-image/flow_intent_resnet18_bs128_task_id" \
    task.num_envs=5 \
    log.save_video=false
