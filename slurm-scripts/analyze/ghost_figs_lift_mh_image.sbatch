#!/bin/bash
#SBATCH --job-name=ghost-lift-mh-img
#SBATCH --output=slrum-commands/logs/ghost_lift_mh_image_%j.out
#SBATCH --error=slrum-commands/logs/ghost_lift_mh_image_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=6:00:00

set -euo pipefail

mkdir -p slrum-commands/logs rollouts/ghost_lift_mh_image

cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

export NUMBA_CACHE_DIR="$(mktemp -d /tmp/numba_${SLURM_JOB_ID:-manual}_XXXXXX)"
export MPLCONFIGDIR="$(mktemp -d /tmp/mpl_${SLURM_JOB_ID:-manual}_XXXXXX)"
export XDG_RUNTIME_DIR="$(mktemp -d /tmp/xdg_${SLURM_JOB_ID:-manual}_XXXXXX)"
export MUJOCO_GL=egl
trap 'rm -rf "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_RUNTIME_DIR"' EXIT

BASE="/data/user_data/ldahiya/mip_render/lift-mh-image"

CKPT_BL="${BASE}/baseline/models/model_best.pt"
CKPT_FI="${BASE}/flow_intent/models/model_best.pt"
CKPT_FI_H16="${BASE}/flow_intent_h16/models/model_best.pt"
CKPT_FI_H32="${BASE}/flow_intent_h32/models/model_best.pt"
CKPT_FI_H64="${BASE}/flow_intent_h64/models/model_best.pt"
CKPT_FI_H128="${BASE}/flow_intent_h128/models/model_best.pt"
CKPT_HIEREMB="${BASE}/hierarchical_emb/models/model_best.pt"
CKPT_SLOT_V8="${BASE}/slot_intent_seed0_v8/models/model_best.pt"
CKPT_CNN="${BASE}/cnn_intent_seed0/models/model_best.pt"

# ── Comparison 1: key model variants ─────────────────────────────────────────
echo "=== [$(date)] Ghost rollouts: key model comparison ==="

python examples/collect_ghost_rollouts.py \
    --run "flow_intent:${CKPT_FI}:task=lift_mh_image_flow_intent:network=mlp_flow_intent" \
    --run "baseline:${CKPT_BL}:task=lift_mh_image" \
    --run "slot_intent_v8:${CKPT_SLOT_V8}:task=lift_mh_image_slot_intent_v8:network=mlp_flow_intent" \
    --run "hierarchical_emb:${CKPT_HIEREMB}:task=lift_mh_image_hierarchical_emb:network=mlp" \
    --run "cnn_intent:${CKPT_CNN}:task=lift_mh_image_cnn_intent:network=mlp_flow_intent" \
    --n-rollouts 10 \
    --n-critical 5 \
    --n-probe 20 \
    --k-ghost 10 \
    --n-future-steps 80 \
    --device cuda \
    --out rollouts/ghost_lift_mh_image/key_models.pkl

echo "=== [$(date)] Plotting: key model comparison ==="
python examples/plot_ghost.py \
    --in rollouts/ghost_lift_mh_image/key_models.pkl \
    --out rollouts/ghost_lift_mh_image/key_models.png \
    --n-critical 5 \
    --n-ghost-plot 10 \
    --k 3 \
    --tag "lift-mh-image key models"

# ── Comparison 2: intent horizon ablation ────────────────────────────────────
echo "=== [$(date)] Ghost rollouts: horizon ablation ==="

python examples/collect_ghost_rollouts.py \
    --run "flow_intent_h8:${CKPT_FI}:task=lift_mh_image_flow_intent:network=mlp_flow_intent" \
    --run "flow_intent_h16:${CKPT_FI_H16}:task=lift_mh_image_flow_intent_h16:network=mlp_flow_intent" \
    --run "flow_intent_h32:${CKPT_FI_H32}:task=lift_mh_image_flow_intent_h32:network=mlp_flow_intent" \
    --run "flow_intent_h64:${CKPT_FI_H64}:task=lift_mh_image_flow_intent_h64:network=mlp_flow_intent" \
    --run "flow_intent_h128:${CKPT_FI_H128}:task=lift_mh_image_flow_intent_h128:network=mlp_flow_intent" \
    --n-rollouts 10 \
    --n-critical 5 \
    --n-probe 20 \
    --k-ghost 10 \
    --n-future-steps 80 \
    --device cuda \
    --out rollouts/ghost_lift_mh_image/horizon_ablation.pkl

echo "=== [$(date)] Plotting: horizon ablation ==="
python examples/plot_ghost.py \
    --in rollouts/ghost_lift_mh_image/horizon_ablation.pkl \
    --out rollouts/ghost_lift_mh_image/horizon_ablation.png \
    --n-critical 5 \
    --n-ghost-plot 10 \
    --k 3 \
    --tag "lift-mh-image horizon ablation"

echo "=== [$(date)] Done. Figures in rollouts/ghost_lift_mh_image/ ==="
