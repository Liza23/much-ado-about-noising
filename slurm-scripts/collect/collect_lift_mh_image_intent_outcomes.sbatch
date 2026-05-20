#!/bin/bash
#SBATCH --job-name=lift-mh-image-intent-outcomes
#SBATCH --output=slrum-commands/logs/lift_mh_image_intent_outcomes_%j.out
#SBATCH --error=slrum-commands/logs/lift_mh_image_intent_outcomes_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=6:00:00

set -euo pipefail

mkdir -p slrum-commands/logs rollouts/lift_mh_image_intent_outcomes_figs

cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

NUMBA_CACHE_DIR=/tmp MUJOCO_GL=egl python examples/collect_intent_outcomes_image.py \
    --run "flow_intent:/data/user_data/ldahiya/mip_render/lift-mh-image/flow_intent/models/model_best.pt:task=lift_mh_image_flow_intent:+network.arch_variant=flow_intent" \
    --n-rollouts 10 \
    --n-critical 5 \
    --n-future-steps 120 \
    --max-timestep 80 \
    --n-trials 10 \
    --k-clusters 3 \
    --render \
    --device cuda \
    --out rollouts/lift_mh_image_intent_outcomes.pkl \
    --out-dir rollouts/lift_mh_image_intent_outcomes_figs
