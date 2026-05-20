#!/bin/bash
#SBATCH --job-name=lift-mh-slot-intent
#SBATCH --output=logs/lift_mh_slot_intent_%j.out
#SBATCH --error=logs/lift_mh_slot_intent_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

mkdir -p logs

cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

export MUJOCO_GL=egl
export NUMBA_CACHE_DIR=/tmp
export MPLCONFIGDIR=/tmp/matplotlib
export XDG_RUNTIME_DIR=/tmp

python examples/train_robomimic.py \
    task=lift_mh_image_slot_intent \
    network=mlp_flow_intent \
    optimization.gradient_steps=300000 \
    optimization.batch_size=16 \
    optimization.loss_type=flow \
    optimization.auto_resume=false \
    log.eval_episodes=50 \
    log.eval_freq=20000 \
    log.save_freq=50000 \
    log.project=slot-intent \
    log.group=lift-mh-image \
    log.exp_name=slot_intent_seed0_v5 \
    task.num_envs=4
