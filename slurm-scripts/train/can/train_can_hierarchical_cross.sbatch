#!/bin/bash
#SBATCH --job-name=can-hier-cross
#SBATCH --output=logs/can_hierarchical_cross_%j.out
#SBATCH --error=logs/can_hierarchical_cross_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

mkdir -p logs

cd /home/ldahiya/max_vla/much-ado-about-noising

# Cross condition: act_steps=4 (short execution), intent_horizon=16 (long HL lookahead).
# Dataset horizon auto-set to max(6, 2+16)=18.
python examples/train_robomimic.py \
    task=can_ph_state_hierarchical \
    task.act_steps=4 \
    task.horizon=6 \
    task.intent_horizon=16 \
    network=mlp \
    optimization.gradient_steps=300000 \
    optimization.batch_size=256 \
    optimization.loss_type=flow \
    log.eval_episodes=50 \
    log.eval_freq=20000 \
    log.save_freq=50000 \
    log.project=intent-conditioning-comparison \
    log.group=can-ph-state \
    log.exp_name=hierarchical_joint_cross \
    task.num_envs=5
