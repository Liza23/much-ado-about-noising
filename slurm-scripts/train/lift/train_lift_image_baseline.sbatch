#!/bin/bash
#SBATCH --job-name=lift-image-baseline
#SBATCH --output=logs/image/lift_image_baseline_%j.out
#SBATCH --error=logs/image/lift_image_baseline_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

mkdir -p logs/image

cd /home/ldahiya/max_vla/much-ado-about-noising

# Lift image baseline — plain flow policy, no intent conditioning.
# Image obs: agentview_image + robot0_eye_in_hand_image + lowdim (eef_pos, eef_quat, gripper).
python examples/train_robomimic.py \
    task=lift_ph_image \
    optimization.gradient_steps=300000 \
    optimization.batch_size=64 \
    optimization.loss_type=flow \
    log.eval_episodes=50 \
    log.eval_freq=20000 \
    log.save_freq=50000 \
    log.project=intent-conditioning-comparison \
    log.group=lift-ph-image \
    log.exp_name=lift-ph-image-baseline \
    task.num_envs=5 \
    optimization.auto_resume=false
