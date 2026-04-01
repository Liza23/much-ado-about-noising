#!/bin/bash
#SBATCH --job-name=lift-hier-cross
#SBATCH --output=logs/lift_hierarchical_cross_%j.out
#SBATCH --error=logs/lift_hierarchical_cross_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

mkdir -p logs

cd /home/ldahiya/max_vla/much-ado-about-noising

# Cross condition: act_steps=4 (short execution chunks), intent_horizon=16 (long HL lookahead).
# HL policy plans 16 steps ahead; LL re-executes every 4 steps with a fresh HL plan.
# Dataset horizon is automatically set to max(6, 2+16)=18 by make_dataset.
# This tests whether a longer-horizon intent signal helps short-chunk execution.
python examples/train_robomimic.py \
    task=lift_ph_state_hierarchical \
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
    log.group=chunk-ablation \
    log.exp_name=hierarchical_joint_cross \
    task.num_envs=5
