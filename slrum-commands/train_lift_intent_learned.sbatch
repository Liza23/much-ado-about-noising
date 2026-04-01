#!/bin/bash
#SBATCH --job-name=lift-intent-learned
#SBATCH --output=logs/lift_intent_learned_%j.out
#SBATCH --error=logs/lift_intent_learned_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

mkdir -p logs

cd /home/ldahiya/max_vla/much-ado-about-noising

# Intent-conditioned policy with a learned intent predictor at inference.
# At training time: intent = mean future eef from demo data (identical to intent_cv).
# At inference time: a jointly-trained MLP predicts the intent from current obs,
# replacing the constant-velocity proxy (no train/inference distribution mismatch).
python examples/train_robomimic.py \
    task=lift_ph_state_intent_learned \
    network=mlp \
    optimization.gradient_steps=300000 \
    optimization.batch_size=256 \
    optimization.loss_type=flow \
    log.eval_episodes=50 \
    log.eval_freq=20000 \
    log.save_freq=50000 \
    log.project=intent-conditioning-comparison \
    log.group=lift-ph-state \
    log.exp_name=intent_learned \
    task.num_envs=5
