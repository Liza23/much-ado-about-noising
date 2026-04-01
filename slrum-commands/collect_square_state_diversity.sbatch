#!/bin/bash
#SBATCH --job-name=sq-state-diversity
#SBATCH --output=slrum-commands/logs/square_state_diversity_%j.out
#SBATCH --error=slrum-commands/logs/square_state_diversity_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00

mkdir -p slrum-commands/logs rollouts

cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

python examples/collect_diversity_rollouts.py \
    --task square_ph_state \
    --run "baseline:checkpoints/square_ph_state_flow_mlp_512_h10_seed0_success88.pt:task=square_ph_state" \
    --run "hierarchical_emb:checkpoints/square_ph_state_flow_mlp_512_h10_seed0_intent_learned_joint_emb_success92.pt:task=square_ph_state_hierarchical_emb" \
    --run "flow_intent:checkpoints/square_ph_state_flow_mlp_512_h10_seed0_intent_flow_intent_success86.pt:task=square_ph_state_flow_intent:+network.arch_variant=flow_intent" \
    --n-rollouts 100 --device cuda \
    --out rollouts/square_ph_state_diversity.pkl

python examples/analyze_diversity.py \
    --data rollouts/square_ph_state_diversity.pkl \
    --task square_ph_state \
    --out-dir rollouts/square_ph_state_figs
