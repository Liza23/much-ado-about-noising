#!/bin/bash
#SBATCH --job-name=sq-image-flow-intent
#SBATCH --output=logs/image/square_image_flow_intent_%j.out
#SBATCH --error=logs/image/square_image_flow_intent_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00

mkdir -p logs/image

cd /home/ldahiya/max_vla/much-ado-about-noising

# Square image Config A (flow intent) — raw 7D intent via flow model.
python examples/train_robomimic.py \
    task=square_ph_image_flow_intent \
    network=mlp_flow_intent \
    optimization.gradient_steps=1000000 \
    optimization.batch_size=64 \
    optimization.loss_type=flow \
    log.eval_episodes=50 \
    log.eval_freq=50000 \
    log.save_freq=100000 \
    log.project=intent-conditioning-comparison \
    log.group=square-ph-image \
    log.exp_name=square-ph-image-flow-intent \
    task.num_envs=5 \
    optimization.auto_resume=false
