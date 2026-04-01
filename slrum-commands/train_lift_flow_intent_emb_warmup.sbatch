#!/bin/bash
#SBATCH --job-name=lift-flow-intent-emb-warmup
#SBATCH --output=logs/lift_flow_intent_emb_warmup_%j.out
#SBATCH --error=logs/lift_flow_intent_emb_warmup_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

mkdir -p logs

cd /home/ldahiya/max_vla/much-ado-about-noising

# Config A with encoded-mean intent + encoder warmup (freeze after 50k steps).
# IntentEncoder trains freely for 50k steps, then is frozen.
# Flow intent model then learns a distribution over the stable 64D embedding space.
python examples/train_robomimic.py \
    task=lift_ph_state_flow_intent_emb_warmup \
    network=mlp_flow_intent \
    optimization.gradient_steps=300000 \
    optimization.batch_size=256 \
    optimization.loss_type=flow \
    log.eval_episodes=50 \
    log.eval_freq=20000 \
    log.save_freq=50000 \
    log.project=intent-conditioning-comparison \
    log.group=lift-ph-state \
    log.exp_name=flow_intent_emb_warmup \
    task.num_envs=5 \
    optimization.auto_resume=false
