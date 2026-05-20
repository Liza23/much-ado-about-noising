#!/bin/bash
#SBATCH --job-name=kitchen-intent-outcomes
#SBATCH --output=slrum-commands/logs/collect_kitchen_state_intent_outcomes_%j.out
#SBATCH --error=slrum-commands/logs/collect_kitchen_state_intent_outcomes_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00

set -euo pipefail

mkdir -p slrum-commands/logs rollouts/kitchen_state_intent_outcomes_figs

cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

NUMBA_CACHE_DIR=/tmp MPLCONFIGDIR=/tmp/matplotlib MUJOCO_GL=egl python examples/collect_intent_outcomes.py \
    --run "flow_intent:/data/user_data/ldahiya/mip_render/kitchen-state/flow_intent/models/model_best.pt:task=kitchen_state_flow_intent:++network.arch_variant=flow_intent" \
    --n-rollouts 10 \
    --n-critical 5 \
    --n-intents 50 \
    --k-clusters 3 \
    --n-future-steps 80 \
    --n-trials 20 \
    --render \
    --plot-intent-xyz \
    --device cuda \
    --out rollouts/kitchen_state_intent_outcomes.pkl \
    --out-dir rollouts/kitchen_state_intent_outcomes_figs
