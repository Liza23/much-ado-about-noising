#!/bin/bash
#SBATCH --job-name=lift-mh-intent-tree
#SBATCH --output=slrum-commands/logs/collect_lift_mh_state_intent_tree_%j.out
#SBATCH --error=slrum-commands/logs/collect_lift_mh_state_intent_tree_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00

set -euo pipefail

mkdir -p slrum-commands/logs rollouts/lift_mh_state_intent_tree_figs

cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

python examples/collect_intent_tree.py \
    --run "flow_intent:checkpoints/lift_mh_state_flow_mlp_512_h10_seed0_intent_flow_intent_success100.pt:task=lift_mh_state_flow_intent:+network.arch_variant=flow_intent" \
    --n-rollouts 5 \
    --diversity-threshold 0.05 \
    --k-branches 3 \
    --max-depth 3 \
    --n-probe 20 \
    --render \
    --plot-mode raw-intents \
    --device cuda \
    --out rollouts/lift_mh_state_intent_tree.pkl \
    --out-dir rollouts/lift_mh_state_intent_tree_figs
