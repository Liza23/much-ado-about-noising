#!/bin/bash
#SBATCH --job-name=coverage-lift-mh-state
#SBATCH --output=slrum-commands/logs/coverage_lift_mh_state_%j.out
#SBATCH --error=slrum-commands/logs/coverage_lift_mh_state_%j.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00

set -euo pipefail

mkdir -p slrum-commands/logs rollouts figs/coverage/lift_mh

cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

ulimit -n 65536

export NUMBA_CACHE_DIR=/tmp
export MPLCONFIGDIR=/tmp/matplotlib
export MUJOCO_GL=egl
export XDG_RUNTIME_DIR=/tmp

VARIANT=${VARIANT:-baseline}
CKPT_BL="checkpoints/lift_mh_state_flow_mlp_512_h10_seed0_success100.pt"
CKPT_FI="checkpoints/lift_mh_state_flow_mlp_512_h10_seed0_intent_flow_intent_success100.pt"

echo "============================================================"
echo "Collecting 1k rollouts: ${VARIANT}"
echo "============================================================"

if [ "${VARIANT}" = "baseline" ]; then
    python examples/collect_diversity_rollouts.py \
        --task lift_mh \
        --run "baseline:${CKPT_BL}:task=lift_mh_state" \
        --n-rollouts 1000 \
        --device cuda \
        --out "rollouts/coverage_lift_mh_${VARIANT}.pkl"
else
    python examples/collect_diversity_rollouts.py \
        --task lift_mh \
        --run "flow_intent:${CKPT_FI}:task=lift_mh_state_flow_intent:+network.arch_variant=flow_intent" \
        --n-rollouts 1000 \
        --device cuda \
        --out "rollouts/coverage_lift_mh_${VARIANT}.pkl"
fi

echo "Done: rollouts/coverage_lift_mh_${VARIANT}.pkl"