#!/bin/bash
#SBATCH --job-name=square-intent-learned
#SBATCH --output=logs/square_intent_learned_%j.out
#SBATCH --error=logs/square_intent_learned_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

mkdir -p logs
source /home/ldahiya/max_vla/much-ado-about-noising/.venv/bin/activate
cd /home/ldahiya/max_vla/much-ado-about-noising

python examples/train_robomimic.py \
    task=square_ph_state_intent_learned \
    network=mlp \
    optimization.gradient_steps=1000000 \
    optimization.batch_size=256 \
    optimization.loss_type=flow \
    log.eval_episodes=50 \
    log.eval_freq=20000 \
    log.save_freq=50000 \
    log.project=intent-conditioning-comparison \
    log.group=square-ph-state \
    log.exp_name=intent_learned \
    task.num_envs=5 \
    optimization.auto_resume=false
