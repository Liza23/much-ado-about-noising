#!/bin/bash
#SBATCH --job-name=can-mh-img-base
#SBATCH --output=logs/can_mh_image_baseline_%j.out
#SBATCH --error=logs/can_mh_image_baseline_%j.err
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
    task=can_mh_image \
    network=mlp \
    optimization.gradient_steps=300000 \
    optimization.batch_size=64 \
    optimization.loss_type=flow \
    optimization.auto_resume=false \
    log.eval_episodes=50 \
    log.eval_freq=20000 \
    log.save_freq=50000 \
    log.project=can-mh \
    log.group=can-mh-image \
    log.exp_name=baseline_seed0 \
    task.num_envs=4
