#!/bin/bash
#SBATCH --job-name=lift-mh-rsac
#SBATCH --output=logs/lift_mh_state_residual_sac_%j.out
#SBATCH --error=logs/lift_mh_state_residual_sac_%j.err
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

# Exp 4: Intent diffusioned flow + SAC on action
# Base: frozen FlowIntentAgent (Config A) trained on lift_mh_state_flow_intent
# SAC: learns residual delta_action in action space on top of the frozen base policy
FLOW_INTENT_CKPT="/data/user_data/ldahiya/mip_render/lift-mh-state/flow_intent/models/model_best.pt"

python examples/train_residual_sac.py \
    task=lift_mh_state_flow_intent \
    network=mlp_flow_intent \
    residual_sac.flow_intent_ckpt="${FLOW_INTENT_CKPT}" \
    residual_sac.residual_alpha=0.1 \
    residual_sac.actor_lr=3e-4 \
    residual_sac.critic_lr=3e-4 \
    residual_sac.discount=0.99 \
    residual_sac.batch_size=256 \
    residual_sac.warmup_steps=5000 \
    residual_sac.total_env_steps=1000000 \
    residual_sac.num_updates_per_step=1 \
    task.num_envs=5 \
    log.eval_episodes=50 \
    log.eval_freq=10000 \
    log.save_freq=50000 \
    log.project=lift-experiments \
    log.group=lift-mh-state \
    log.exp_name=intent_flow_residual_sac
