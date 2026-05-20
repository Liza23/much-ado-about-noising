#!/bin/bash
#SBATCH --job-name=lift-baseline
#SBATCH --output=logs/lift_baseline_%j.out
#SBATCH --error=logs/lift_baseline_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

mkdir -p logs

cd /home/ldahiya/max_vla/much-ado-about-noising

# Baseline: standard flow policy, no intent conditioning
# Uses lift_ph_state task with default MLP network
# Evaluates every 20k steps on 50 rollouts (5 envs x 10 episodes)
python examples/train_robomimic.py \
    task=lift_ph_state \
    network=mlp \
    optimization.gradient_steps=1000000 \
    optimization.batch_size=256 \
    optimization.loss_type=flow \
    log.eval_episodes=50 \
    log.eval_freq=20000 \
    log.save_freq=50000 \
    log.project=intent-conditioning-comparison \
    log.group=lift-ph-state \
    log.exp_name=baseline_no_intent \
    task.num_envs=5
