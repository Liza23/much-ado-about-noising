#!/bin/bash
#SBATCH --job-name=lift-mh-dual-intent
#SBATCH --output=logs/lift_mh_dual_intent_%j.out
#SBATCH --error=logs/lift_mh_dual_intent_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

mkdir -p logs
source /home/ldahiya/max_vla/much-ado-about-noising/.venv/bin/activate
cd /home/ldahiya/max_vla/much-ado-about-noising

python examples/train_robomimic.py \
    task=lift_mh_state_dual_intent \
    network=mlp \
    optimization.gradient_steps=300000 \
    optimization.batch_size=256 \
    optimization.loss_type=flow \
    log.eval_episodes=50 \
    log.eval_freq=20000 \
    log.save_freq=50000 \
    log.project=intent-conditioning-comparison \
    log.group=lift-mh-state \
    log.exp_name=dual_intent \
    task.num_envs=5 \
    optimization.auto_resume=false
