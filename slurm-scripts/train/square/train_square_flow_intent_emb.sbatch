#!/bin/bash
#SBATCH --job-name=sq-flow-intent-emb
#SBATCH --output=logs/square_flow_intent_emb_%j.out
#SBATCH --error=logs/square_flow_intent_emb_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

mkdir -p logs

cd /home/ldahiya/max_vla/much-ado-about-noising

# Config A + encoded-mean intent — square-ph.
python examples/train_robomimic.py \
    task=square_ph_state_flow_intent_emb \
    network=mlp_flow_intent \
    optimization.gradient_steps=1000000 \
    optimization.batch_size=256 \
    optimization.loss_type=flow \
    log.eval_episodes=50 \
    log.eval_freq=50000 \
    log.save_freq=100000 \
    log.project=intent-conditioning-comparison \
    log.group=square-ph-state \
    log.exp_name=flow_intent_emb \
    task.num_envs=5 \
    optimization.auto_resume=false
