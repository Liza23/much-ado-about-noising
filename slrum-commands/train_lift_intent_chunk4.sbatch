#!/bin/bash
#SBATCH --job-name=lift-intent-chunk4
#SBATCH --output=logs/lift_intent_chunk4_%j.out
#SBATCH --error=logs/lift_intent_chunk4_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

mkdir -p logs

cd /home/ldahiya/max_vla/much-ado-about-noising

# Intent = mean future eef over next 4 steps.
# CV proxy at inference: eef_now + 2.5 * velocity
python examples/train_robomimic.py \
    task=lift_ph_state_intent \
    task.act_steps=4 \
    task.horizon=6 \
    network=mlp \
    optimization.gradient_steps=300000 \
    optimization.batch_size=256 \
    optimization.loss_type=flow \
    log.eval_episodes=50 \
    log.eval_freq=20000 \
    log.save_freq=50000 \
    log.project=intent-conditioning-comparison \
    log.group=chunk-ablation \
    log.exp_name=intent_cv_chunk4 \
    task.num_envs=5
