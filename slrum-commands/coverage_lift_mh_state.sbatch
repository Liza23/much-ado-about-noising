#!/bin/bash
#SBATCH --job-name=coverage-lift-mh-state
#SBATCH --output=slrum-commands/logs/coverage_lift_mh_state_%j.out
#SBATCH --error=slrum-commands/logs/coverage_lift_mh_state_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00

set -euo pipefail

mkdir -p slrum-commands/logs rollouts figs/coverage/lift_mh

cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

export NUMBA_CACHE_DIR=/tmp
export MPLCONFIGDIR=/tmp/matplotlib
export MUJOCO_GL=egl
export XDG_RUNTIME_DIR=/tmp

CKPT_BL="checkpoints/lift_mh_state_flow_mlp_512_h10_seed0_success100.pt"
CKPT_FI="checkpoints/lift_mh_state_flow_mlp_512_h10_seed0_intent_flow_intent_success100.pt"

OUT_PKL="rollouts/coverage_lift_mh_state.pkl"
OUT_DIR="figs/coverage/lift_mh"

echo "============================================================"
echo "Stage 1: collecting 1k rollouts per variant"
echo "  baseline:    ${CKPT_BL}"
echo "  flow_intent: ${CKPT_FI}"
echo "============================================================"

python examples/collect_diversity_rollouts.py \
    --task lift_mh \
    --run "baseline:${CKPT_BL}:task=lift_mh_state" \
    --run "flow_intent:${CKPT_FI}:task=lift_mh_state_flow_intent:network.arch_variant=flow_intent" \
    --n-rollouts 1000 \
    --device cuda \
    --out "${OUT_PKL}"

echo "============================================================"
echo "Stage 2: analyzing state-space coverage"
echo "  output: ${OUT_DIR}"
echo "============================================================"

python examples/analyze_coverage.py \
    --data "${OUT_PKL}" \
    --task lift_mh \
    --out-dir "${OUT_DIR}"

echo "Done. Figures saved to: ${OUT_DIR}"
