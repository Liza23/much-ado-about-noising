#!/bin/bash
#SBATCH --job-name=kitchen-base-chi
#SBATCH --output=logs/kitchen_baseline_chiunet_%j.out
#SBATCH --error=logs/kitchen_baseline_chiunet_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

mkdir -p logs
source /home/ldahiya/max_vla/much-ado-about-noising/.venv/bin/activate
cd /home/ldahiya/max_vla/much-ado-about-noising

NUMBA_CACHE_DIR=/tmp MUJOCO_GL=egl python examples/train_kitchen.py \
    task=kitchen_state \
    network=chiunet \
    optimization.gradient_steps=300000 \
    optimization.batch_size=256 \
    optimization.loss_type=flow \
    log.eval_episodes=50 \
    log.eval_freq=20000 \
    log.save_freq=50000 \
    log.project=intent-conditioning-comparison \
    log.group=kitchen-state \
    log.exp_name=baseline_chiunet \
    task.num_envs=5 \
    optimization.auto_resume=false \
    log.save_video=false
