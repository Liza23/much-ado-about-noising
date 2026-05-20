#!/bin/bash
#SBATCH --job-name=sq-image-hier-emb
#SBATCH --output=logs/image/square_image_hierarchical_emb_%j.out
#SBATCH --error=logs/image/square_image_hierarchical_emb_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00

mkdir -p logs/image

cd /home/ldahiya/max_vla/much-ado-about-noising

# Square image Config B (hierarchical) with encoded-mean intent.
python examples/train_robomimic.py \
    task=square_ph_image_hierarchical_emb \
    optimization.gradient_steps=1000000 \
    optimization.batch_size=64 \
    optimization.loss_type=flow \
    log.eval_episodes=50 \
    log.eval_freq=50000 \
    log.save_freq=100000 \
    log.project=intent-conditioning-comparison \
    log.group=square-ph-image \
    log.exp_name=square-ph-image-hierarchical-emb \
    task.num_envs=5 \
    optimization.auto_resume=false
