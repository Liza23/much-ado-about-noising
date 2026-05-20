#!/bin/bash
#SBATCH --job-name=can-hier-c16
#SBATCH --output=logs/can_hierarchical_chunk16_%j.out
#SBATCH --error=logs/can_hierarchical_chunk16_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

mkdir -p logs

cd /home/ldahiya/max_vla/much-ado-about-noising

# act_steps=16, intent_horizon=16, joint training
python examples/train_robomimic.py \
    task=can_ph_state_hierarchical \
    task.act_steps=16 \
    task.horizon=18 \
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
    log.exp_name=hierarchical_joint_chunk16 \
    task.num_envs=5
