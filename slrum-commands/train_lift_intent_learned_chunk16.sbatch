#!/bin/bash
#SBATCH --job-name=lift-intent-learned-chunk16
#SBATCH --output=logs/lift_intent_learned_chunk16_%j.out
#SBATCH --error=logs/lift_intent_learned_chunk16_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

mkdir -p logs

cd /home/ldahiya/max_vla/much-ado-about-noising

python examples/train_robomimic.py \
    task=lift_ph_state_intent_learned \
    task.act_steps=16 \
    task.horizon=18 \
    network=mlp \
    optimization.gradient_steps=300000 \
    optimization.batch_size=256 \
    optimization.loss_type=flow \
    log.eval_episodes=50 \
    log.eval_freq=20000 \
    log.save_freq=50000 \
    log.project=intent-conditioning-comparison \
    log.group=chunk-ablation \
    log.exp_name=intent_learned_chunk16 \
    task.num_envs=5
