#!/bin/bash
#SBATCH --job-name=libero-10-img-fi-s
#SBATCH --output=logs/libero_10_suite_image_flow_intent_sampled_%j.out
#SBATCH --error=logs/libero_10_suite_image_flow_intent_sampled_%j.err
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
    task=libero_10_suite_image_flow_intent \
    network=mlp_flow_intent \
    optimization.gradient_steps=1000000 \
    optimization.batch_size=64 \
    optimization.loss_type=flow \
    optimization.weight_decay=1e-4 \
    log.eval_episodes=10 \
    log.eval_freq=100000 \
    log.eval_nsteps=9 \
    log.save_freq=20000 \
    log.project=libero-experiments \
    log.group=libero-10-suite-image \
    log.exp_name=flow_intent_sampled \
    log.log_dir="/data/user_data/ldahiya/libero-train/libero-10-suite-image/flow_intent_sampled" \
    task.num_envs=5 \
    task.decoder_uses_sampled_intent=true \
    optimization.auto_resume=true \
    log.save_video=false
