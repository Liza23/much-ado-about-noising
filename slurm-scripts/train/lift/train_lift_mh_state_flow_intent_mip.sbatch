#!/bin/bash
#SBATCH --job-name=lift-mh-fi-mip
#SBATCH --output=logs/lift_mh_state_flow_intent_mip_%j.out
#SBATCH --error=logs/lift_mh_state_flow_intent_mip_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

mkdir -p logs

cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

export NUMBA_CACHE_DIR=/tmp
export MPLCONFIGDIR=/tmp/matplotlib
export MUJOCO_GL=egl
export XDG_RUNTIME_DIR=/tmp

# flow_intent_mip: hierarchical MIP — flow intent + flow action, both MIP-sampled.
# Intent trained with mip_loss; action flow trained with flow_loss (GT intent conditioning).
python examples/train_robomimic.py \
    task=lift_mh_state_flow_intent_mip \
    network=mlp_flow_intent_mip \
    optimization.gradient_steps=1000000 \
    optimization.batch_size=256 \
    optimization.loss_type=flow \
    optimization.t_two_step=0.9 \
    log.eval_episodes=50 \
    log.eval_freq=20000 \
    log.save_freq=50000 \
    log.project=intent-conditioning-comparison \
    log.group=lift-mh-state \
    log.exp_name=flow_intent_mip \
    task.num_envs=5
