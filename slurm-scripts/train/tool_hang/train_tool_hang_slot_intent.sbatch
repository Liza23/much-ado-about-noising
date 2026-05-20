#!/bin/bash
#SBATCH --job-name=tool-hang-slot-intent
#SBATCH --output=logs/tool_hang_slot_intent_%j.out
#SBATCH --error=logs/tool_hang_slot_intent_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00

mkdir -p logs

cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

export MUJOCO_GL=egl
export NUMBA_CACHE_DIR=/tmp
export MPLCONFIGDIR=/tmp/matplotlib
export XDG_RUNTIME_DIR=/tmp

python examples/train_robomimic.py \
    task=tool_hang_ph_image_slot_intent \
    network=mlp_flow_intent \
    optimization.gradient_steps=400000 \
    optimization.batch_size=64 \
    optimization.loss_type=flow \
    optimization.auto_resume=false \
    task.rgb_model=resnet50 \
    log.eval_episodes=50 \
    log.eval_freq=20000 \
    log.save_freq=50000 \
    log.project=tool-hang \
    log.group=tool-hang-image \
    log.exp_name=slot_intent_resnet50_bs64_seed0 \
    task.num_envs=4
