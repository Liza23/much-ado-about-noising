#!/bin/bash
#SBATCH --job-name=lift-hierarchical
#SBATCH --output=logs/lift_hierarchical_%j.out
#SBATCH --error=logs/lift_hierarchical_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

mkdir -p logs

cd /home/ldahiya/max_vla/much-ado-about-noising

# Hierarchical joint training: HL predictor and flow policy train simultaneously.
# Flow policy receives HL predictor output (detached) as conditioning — same signal as inference.
# HL predictor trains independently via MSE against GT future eef.
# Checkpoints: flow policy saved to checkpoints/*_intent_learned_joint_success*.pt
#              HL predictor saved to checkpoints/*_intent_learned_joint_hl_policy.pt
python examples/train_robomimic.py \
    task=lift_ph_state_hierarchical \
    network=mlp \
    optimization.gradient_steps=300000 \
    optimization.batch_size=256 \
    optimization.loss_type=flow \
    log.eval_episodes=50 \
    log.eval_freq=20000 \
    log.save_freq=50000 \
    log.project=intent-conditioning-comparison \
    log.group=lift-ph-state \
    log.exp_name=hierarchical_joint \
    task.num_envs=5
