#!/bin/bash
#SBATCH --job-name=lift-flow-intent
#SBATCH --output=logs/lift_flow_intent_%j.out
#SBATCH --error=logs/lift_flow_intent_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

mkdir -p logs

cd /home/ldahiya/max_vla/much-ado-about-noising

# Config A: flow-based intent model + deterministic MLP action decoder.
# Contrast with Config B (hierarchical, intent_learned) which uses an MLP intent predictor.
#
# Architecture:
#   - Shared context encoder (MLP)
#   - Flow intent model: samples intent from noise via ODE (Ta=1, D=7)
#   - MLP action decoder: deterministic (obs_emb, intent) → action
#
# Switch: network=mlp_flow_intent sets network.arch_variant=flow_intent
python examples/train_robomimic.py \
    task=lift_ph_state_flow_intent \
    network=mlp_flow_intent \
    optimization.gradient_steps=300000 \
    optimization.batch_size=256 \
    optimization.loss_type=flow \
    log.eval_episodes=50 \
    log.eval_freq=20000 \
    log.save_freq=50000 \
    log.project=intent-conditioning-comparison \
    log.group=lift-ph-state \
    log.exp_name=flow_intent \
    task.num_envs=5
