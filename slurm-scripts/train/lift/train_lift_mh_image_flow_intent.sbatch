#!/bin/bash
#SBATCH --job-name=lift-mh-img-fi
#SBATCH --output=logs/image/lift_mh_image_flow_intent_%j.out
#SBATCH --error=logs/image/lift_mh_image_flow_intent_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

mkdir -p logs/image
source /home/ldahiya/max_vla/much-ado-about-noising/.venv/bin/activate
cd /home/ldahiya/max_vla/much-ado-about-noising

python examples/train_robomimic.py \
    task=lift_mh_image_flow_intent \
    network=mlp_flow_intent \
    optimization.gradient_steps=300000 \
    optimization.batch_size=64 \
    optimization.loss_type=flow \
    log.eval_episodes=50 \
    log.eval_freq=20000 \
    log.save_freq=50000 \
    log.project=intent-conditioning-comparison \
    log.group=lift-mh-image \
    log.exp_name=flow_intent \
    task.num_envs=5 \
    optimization.auto_resume=false
