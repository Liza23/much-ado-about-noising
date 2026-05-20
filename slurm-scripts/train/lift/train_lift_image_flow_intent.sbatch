#!/bin/bash
#SBATCH --job-name=lift-image-flow-intent
#SBATCH --output=logs/image/lift_image_flow_intent_%j.out
#SBATCH --error=logs/image/lift_image_flow_intent_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

mkdir -p logs/image

cd /home/ldahiya/max_vla/much-ado-about-noising

# Lift image Config A (flow intent) — raw 7D intent via flow model.
# FlowIntentAgent: shared encoder → flow intent model (7D) → MLP action decoder.
python examples/train_robomimic.py \
    task=lift_ph_image_flow_intent \
    network=mlp_flow_intent \
    optimization.gradient_steps=300000 \
    optimization.batch_size=64 \
    optimization.loss_type=flow \
    log.eval_episodes=50 \
    log.eval_freq=20000 \
    log.save_freq=50000 \
    log.project=intent-conditioning-comparison \
    log.group=lift-ph-image \
    log.exp_name=lift-ph-image-flow-intent \
    task.num_envs=5 \
    optimization.auto_resume=false
