#!/bin/bash
#SBATCH --job-name=lift-hier-final
#SBATCH --output=logs/lift_hierarchical_final_%j.out
#SBATCH --error=logs/lift_hierarchical_final_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

mkdir -p logs

cd /home/ldahiya/max_vla/much-ado-about-noising

# Intent = eef[t+N] (endpoint) rather than mean(eef[t+1..t+N]).
# Everything else identical to hierarchical_joint (act_steps=8, intent_horizon=8, joint training).
python examples/train_robomimic.py \
    task=lift_ph_state_hierarchical_final \
    network=mlp \
    optimization.gradient_steps=300000 \
    optimization.batch_size=256 \
    optimization.loss_type=flow \
    log.eval_episodes=50 \
    log.eval_freq=20000 \
    log.save_freq=50000 \
    log.project=intent-conditioning-comparison \
    log.group=lift-ph-state \
    log.exp_name=hierarchical_joint_final \
    task.num_envs=5 \
    optimization.auto_resume=false
