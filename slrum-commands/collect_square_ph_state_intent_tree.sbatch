#!/bin/bash
#SBATCH --job-name=square-ph-intent-tree
#SBATCH --output=slrum-commands/logs/collect_square_ph_state_intent_tree_%j.out
#SBATCH --error=slrum-commands/logs/collect_square_ph_state_intent_tree_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00

set -euo pipefail

mkdir -p slrum-commands/logs rollouts/square_ph_state_intent_tree_figs

cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

python examples/collect_intent_tree.py \
    --run "flow_intent:checkpoints/square_ph_state_flow_mlp_512_h10_seed0_intent_flow_intent_success98.pt:task=square_ph_state_flow_intent:+network.arch_variant=flow_intent" \
    --n-rollouts 5 \
    --diversity-threshold 0.001 \
    --k-branches 3 \
    --max-depth 2 \
    --n-probe 20 \
    --render \
    --device cuda \
    --out rollouts/square_ph_state_intent_tree.pkl \
    --out-dir rollouts/square_ph_state_intent_tree_figs
