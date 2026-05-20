#!/bin/bash
#SBATCH --job-name=eval-lift-ph-state-sr
#SBATCH --output=slrum-commands/logs/eval_lift_ph_state_sr_%j.out
#SBATCH --error=slrum-commands/logs/eval_lift_ph_state_sr_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3:00:00

mkdir -p slrum-commands/logs results

cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

# NOTE: lift_ph_state_flow_mlp_512_h16_seed0_success0.pt is the h16 baseline but has 0% success.
# Using lift_ph_state_flow_mlp_512_seed0_success98.pt as baseline instead — confirm with user.
python examples/eval_success_rate.py \
    --run "baseline:checkpoints/lift_ph_state_flow_mlp_512_seed0_success98.pt:task=lift_ph_state" \
    --run "flow_intent:checkpoints/lift_ph_state_flow_mlp_512_h16_seed0_intent_flow_intent_success100.pt:task=lift_ph_state_flow_intent:+network.arch_variant=flow_intent" \
    --run "hierarchical_emb:checkpoints/lift_ph_state_flow_mlp_512_h16_seed0_intent_learned_joint_emb_success98.pt:task=lift_ph_state_hierarchical_emb" \
    --n-rollouts 100 \
    --device cuda \
    --out results/lift_ph_state_success_rate.csv
