#!/bin/bash
#SBATCH --job-name=ghost-lift-mh-state
#SBATCH --output=slrum-commands/logs/ghost_lift_mh_state_%j.out
#SBATCH --error=slrum-commands/logs/ghost_lift_mh_state_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00

set -euo pipefail

mkdir -p slrum-commands/logs rollouts

cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

export NUMBA_CACHE_DIR=/tmp
export MPLCONFIGDIR=/tmp/matplotlib
export MUJOCO_GL=egl
export XDG_RUNTIME_DIR=/tmp

CKPT_FI="checkpoints/lift_mh_state_flow_mlp_512_h10_seed0_intent_flow_intent_success100.pt"
CKPT_FI_SAMPLED="/data/user_data/ldahiya/mip_render/lift-mh-state/flow_intent_sampled/models/model_latest.pt"
CKPT_FI_CURR="/data/user_data/ldahiya/mip_render/lift-mh-state/flow_intent_curriculum/models/model_best.pt"
CKPT_BL="checkpoints/lift_mh_state_flow_mlp_512_h10_seed0_success100.pt"

OUT_PKL="rollouts/ghost_lift_mh_state.pkl"
OUT_PLOT="rollouts/ghost_lift_mh_state.png"

echo "============================================================"
echo "Ghost rollouts: lift-mh-state all variants"
echo "flow_intent          ckpt: ${CKPT_FI}"
echo "flow_intent_sampled  ckpt: ${CKPT_FI_SAMPLED}"
echo "flow_intent_curr     ckpt: ${CKPT_FI_CURR}"
echo "baseline             ckpt: ${CKPT_BL}"
echo "============================================================"

python examples/collect_ghost_rollouts.py \
    --run "flow_intent:${CKPT_FI}:task=lift_mh_state_flow_intent:network=mlp_flow_intent" \
    --run "flow_intent_sampled:${CKPT_FI_SAMPLED}:task=lift_mh_state_flow_intent:network=mlp_flow_intent:+task.decoder_uses_sampled_intent=true" \
    --run "flow_intent_curriculum:${CKPT_FI_CURR}:task=lift_mh_state_flow_intent:network=mlp_flow_intent:+task.decoder_uses_sampled_intent=true:+task.decoder_curriculum_steps=100000" \
    --run "baseline_stochastic:${CKPT_BL}:task=lift_mh_state:network=mlp:optimization.sample_mode=stochastic" \
    --run "baseline_zero:${CKPT_BL}:task=lift_mh_state:network=mlp:optimization.sample_mode=zero" \
    --n-rollouts 10 \
    --n-critical 5 \
    --n-probe 20 \
    --k-ghost 8 \
    --n-future-steps 80 \
    --max-timestep 100 \
    --device cuda \
    --out "${OUT_PKL}"

echo "============================================================"
echo "Generating ghost plot: ${OUT_PLOT}"
echo "============================================================"

python examples/plot_ghost.py \
    --in "${OUT_PKL}" \
    --out "${OUT_PLOT}" \
    --n-ghost-plot 8 \
    --k 4 \
    --tag lift_mh_state

echo "Done. Plot saved to: ${OUT_PLOT}"
