#!/bin/bash
#SBATCH --job-name=lift-intent
#SBATCH --output=logs/lift_intent_%j.out
#SBATCH --error=logs/lift_intent_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

mkdir -p logs

cd /home/ldahiya/max_vla/much-ado-about-noising

# Intent-conditioned policy: augments obs with mean future eef (pos+quat, 7D)
# over the next act_steps=8 steps from ground truth demonstration data.
# At inference, uses current eef from obs as intent proxy.
# Effective obs_dim increases by 7 (intent_dim); all else is identical to baseline.
python examples/train_robomimic.py \
    task=lift_ph_state_intent \
    network=mlp \
    optimization.gradient_steps=1000000 \
    optimization.batch_size=256 \
    optimization.loss_type=flow \
    log.eval_episodes=50 \
    log.eval_freq=20000 \
    log.save_freq=50000 \
    log.project=intent-conditioning-comparison \
    log.group=lift-ph-state \
    log.exp_name=intent_conditioned \
    task.num_envs=5
