#!/bin/bash
#SBATCH --job-name=can-ph-diversity
#SBATCH --output=slrum-commands/logs/can_ph_diversity_%j.out
#SBATCH --error=slrum-commands/logs/can_ph_diversity_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00

mkdir -p slrum-commands/logs rollouts

cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

python examples/collect_diversity_rollouts.py \
    --task can_ph_state \
    --run "baseline:checkpoints/can_ph_state_flow_mlp_512_h10_seed0_success98.pt:task=can_ph_state" \
    --run "intent_learned:checkpoints/can_ph_state_flow_mlp_512_h10_seed0_intent_learned_success100.pt:task=can_ph_state_intent_learned" \
    --run "hierarchical_emb:checkpoints/can_ph_state_flow_mlp_512_h10_seed0_intent_learned_joint_success96.pt:task=can_ph_state_hierarchical" \
    --n-rollouts 100 --device cuda \
    --out rollouts/can_ph_state_diversity.pkl

python examples/analyze_diversity.py \
    --data rollouts/can_ph_state_diversity.pkl \
    --task can_ph_state \
    --out-dir rollouts/can_ph_state_figs
