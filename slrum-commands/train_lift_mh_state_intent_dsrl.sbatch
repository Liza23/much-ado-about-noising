#!/bin/bash
#SBATCH --job-name=lift-mh-intent-dsrl
#SBATCH --output=logs/lift_mh_state_intent_dsrl_%j.out
#SBATCH --error=logs/lift_mh_state_intent_dsrl_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail

mkdir -p logs

cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

export NUMBA_CACHE_DIR=/tmp
export MPLCONFIGDIR=/tmp/matplotlib
export MUJOCO_GL=egl
export XDG_RUNTIME_DIR=/tmp

# Exp 3: DSRL on FlowIntentAgent (Config A)
# SAC learns to pick the 7D intent noise x_0 that steers the intent ODE
# Base: frozen FlowIntentAgent trained on lift_mh_state_flow_intent
FLOW_INTENT_CKPT="/data/user_data/ldahiya/mip_render/lift-mh-state/flow_intent/models/model_best.pt"

python examples/train_dsrl.py \
    task=lift_mh_state_flow_intent \
    network=mlp_flow_intent \
    dsrl.flow_ckpt="${FLOW_INTENT_CKPT}" \
    dsrl.noise_mag=2.0 \
    dsrl.actor_lr=3e-4 \
    dsrl.critic_lr=3e-4 \
    dsrl.discount=0.99 \
    dsrl.reward_scale=0.1 \
    dsrl.q_target_clip=10.0 \
    dsrl.batch_size=256 \
    dsrl.warmup_steps=1000 \
    dsrl.total_env_steps=1000000 \
    dsrl.num_updates_per_step=1 \
    task.num_envs=5 \
    log.eval_episodes=50 \
    log.eval_freq=10000 \
    log.save_freq=50000 \
    log.project=lift-experiments \
    log.group=lift-mh-state \
    log.exp_name=flow_intent_dsrl
