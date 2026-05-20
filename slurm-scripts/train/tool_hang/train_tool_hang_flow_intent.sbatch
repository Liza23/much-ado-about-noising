#!/bin/bash
#SBATCH --job-name=tool-hang-flow-intent
#SBATCH --output=slrum-commands/logs/tool_hang_flow_intent_%j.out
#SBATCH --error=slrum-commands/logs/tool_hang_flow_intent_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00

mkdir -p slrum-commands/logs

cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

export MUJOCO_GL=egl
export NUMBA_CACHE_DIR=/tmp
export MPLCONFIGDIR=/tmp/matplotlib
export XDG_RUNTIME_DIR=/tmp
export HF_HOME=/data/group_data/maxlab/common_datasets/pandaliza/cache

python examples/train_robomimic.py \
    task=tool_hang_ph_image_flow_intent \
    network=mlp_flow_intent \
    optimization.gradient_steps=600000 \
    optimization.batch_size=64 \
    optimization.loss_type=flow \
    optimization.auto_resume=false \
    task.rgb_model=resnet50 \
    log.eval_episodes=25 \
    log.eval_freq=20000 \
    log.save_freq=50000 \
    log.project=tool-hang \
    log.group=tool-hang-image \
    log.exp_name=flow_intent_resnet50_bs64_seed0
