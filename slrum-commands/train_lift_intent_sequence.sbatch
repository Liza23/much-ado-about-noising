#!/bin/bash
#SBATCH --job-name=lift-intent-seq
#SBATCH --output=logs/lift_intent_sequence_%j.out
#SBATCH --error=logs/lift_intent_sequence_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

mkdir -p logs

cd /home/ldahiya/max_vla/much-ado-about-noising

# Full intent sequence: intent = [eef[t+1],...,eef[t+8]] flattened to (56,)
# Joint training: flow policy trains on predictor output (not GT) to close train/inference gap.
python examples/train_robomimic.py \
    task=lift_ph_state_intent_sequence \
    network=mlp \
    optimization.gradient_steps=1000000 \
    optimization.batch_size=256 \
    optimization.loss_type=flow \
    log.eval_episodes=50 \
    log.eval_freq=20000 \
    log.save_freq=50000 \
    log.project=intent-conditioning-comparison \
    log.group=lift-ph-state \
    log.exp_name=hierarchical_joint_sequence \
    task.num_envs=5 \
    optimization.auto_resume=false
