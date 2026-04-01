#!/bin/bash
#SBATCH --job-name=lift-state-diversity
#SBATCH --output=slrum-commands/logs/lift_state_diversity_%j.out
#SBATCH --error=slrum-commands/logs/lift_state_diversity_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00

mkdir -p slrum-commands/logs rollouts

cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

python examples/collect_diversity_rollouts.py \
    --task lift_ph_state \
    --run "baseline:checkpoints/lift_ph_state_flow_mlp_512_h6_seed0_success98.pt:task=lift_ph_state" \
    --run "hierarchical_emb:checkpoints/lift_ph_state_flow_mlp_512_h6_seed0_intent_learned_joint_success100.pt:task=lift_ph_state_hierarchical_emb" \
    --run "intent_learned:checkpoints/lift_ph_state_flow_mlp_512_h6_seed0_intent_learned_success100.pt:task=lift_ph_state_intent_learned" \
    --n-rollouts 100 --device cuda \
    --out rollouts/lift_ph_state_diversity.pkl

python examples/analyze_diversity.py \
    --data rollouts/lift_ph_state_diversity.pkl \
    --task lift_ph_state \
    --out-dir rollouts/lift_ph_state_figs
