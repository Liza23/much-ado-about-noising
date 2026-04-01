#!/bin/bash
#SBATCH --job-name=lift-image-hier-emb
#SBATCH --output=logs/image/lift_image_hierarchical_emb_%j.out
#SBATCH --error=logs/image/lift_image_hierarchical_emb_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

mkdir -p logs/image

cd /home/ldahiya/max_vla/much-ado-about-noising

# Lift image Config B (hierarchical) with encoded-mean intent.
# IntentEncoder (7→64D) + IntentPredictor co-trained jointly via MSE.
# Action flow policy conditions on 64D embedded intent via shape_meta "intent" key.
python examples/train_robomimic.py \
    task=lift_ph_image_hierarchical_emb \
    optimization.gradient_steps=300000 \
    optimization.batch_size=64 \
    optimization.loss_type=flow \
    log.eval_episodes=50 \
    log.eval_freq=20000 \
    log.save_freq=50000 \
    log.project=intent-conditioning-comparison \
    log.group=lift-ph-image \
    log.exp_name=lift-ph-image-hierarchical-emb \
    task.num_envs=5 \
    optimization.auto_resume=false
