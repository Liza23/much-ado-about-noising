#!/bin/bash
#SBATCH --job-name=lift-mh-intent-outcomes
#SBATCH --output=slrum-commands/logs/collect_lift_mh_state_intent_outcomes_%j.out
#SBATCH --error=slrum-commands/logs/collect_lift_mh_state_intent_outcomes_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00

set -euo pipefail

mkdir -p slrum-commands/logs rollouts/lift_mh_state_intent_outcomes_figs

cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

python examples/collect_intent_outcomes.py \
    --run "flow_intent:checkpoints/lift_mh_state_flow_mlp_512_h10_seed0_intent_flow_intent_success100.pt:task=lift_mh_state_flow_intent:+network.arch_variant=flow_intent" \
    --n-rollouts 10 \
    --n-critical 1 \
    --max-timestep 80 \
    --n-intents 50 \
    --k-clusters 3 \
    --n-future-steps 80 \
    --n-trials 20 \
    --render \
    --plot-intent-xyz \
    --device cuda \
    --out rollouts/lift_mh_state_intent_outcomes.pkl \
    --out-dir rollouts/tmp-lift_mh_state_intent_outcomes-1critical
