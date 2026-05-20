#!/bin/bash
#SBATCH --job-name=libero-obj-rparl-img
#SBATCH --output=logs/libero_object_residual_parl_multitask_image_%j.out
#SBATCH --error=logs/libero_object_residual_parl_multitask_image_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
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

FLOW_INTENT_CKPT="/data/user_data/ldahiya/libero-train/libero-object-suite-image/flow_intent/models/model_latest.pt"

python examples/train_residual_parl_multitask_image.py \
    task=libero_object_suite_image_flow_intent \
    network=mlp_flow_intent \
    residual_parl.flow_intent_ckpt="${FLOW_INTENT_CKPT}" \
    residual_parl.residual_alpha=1.0 \
    residual_parl.action_magnitude=0.1 \
    residual_parl.actor_lr=3e-4 \
    residual_parl.critic_lr=3e-4 \
    residual_parl.discount=0.99 \
    residual_parl.batch_size=256 \
    residual_parl.warmup_steps=5000 \
    residual_parl.total_env_steps=2000000 \
    residual_parl.num_updates_per_step=1 \
    residual_parl.parl_num_samples=16 \
    residual_parl.parl_num_elites=4 \
    residual_parl.parl_num_grad_steps=5 \
    residual_parl.parl_step_size=0.01 \
    residual_parl.num_critic_updates=2 \
    residual_parl.num_actor_updates=4 \
    +residual_parl.flow_num_steps=9 \
    log.eval_episodes=10 \
    log.eval_freq=100000 \
    log.save_freq=20000 \
    log.project=libero-experiments \
    log.group=libero-object-suite-image \
    log.exp_name=residual_parl_multitask_image \
    log.log_dir="/data/user_data/ldahiya/libero-train/libero-object-suite-image/residual_parl_multitask" \
    task.num_envs=5 \
    log.save_video=false
