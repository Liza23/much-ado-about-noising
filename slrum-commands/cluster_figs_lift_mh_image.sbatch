#!/bin/bash
#SBATCH --job-name=cluster-lift-mh-image
#SBATCH --output=slrum-commands/logs/cluster_lift_mh_image_%j.out
#SBATCH --error=slrum-commands/logs/cluster_lift_mh_image_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00

set -euo pipefail

mkdir -p slrum-commands/logs rollouts/lift_mh_image_cluster_figs

cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

export NUMBA_CACHE_DIR=/tmp
export MPLCONFIGDIR=/tmp/matplotlib
export MUJOCO_GL=egl
export XDG_RUNTIME_DIR=/tmp

BASE="/data/user_data/ldahiya/mip_render/lift-mh-image"

CKPT_BL="${BASE}/baseline/models/model_best.pt"
CKPT_FI="${BASE}/flow_intent/models/model_best.pt"
CKPT_FI_H16="${BASE}/flow_intent_h16/models/model_best.pt"
CKPT_FI_H32="${BASE}/flow_intent_h32/models/model_best.pt"
CKPT_FI_H64="${BASE}/flow_intent_h64/models/model_best.pt"
CKPT_FI_H128="${BASE}/flow_intent_h128/models/model_best.pt"
CKPT_HIEREMB="${BASE}/hierarchical_emb/models/model_best.pt"
CKPT_CNN="${BASE}/cnn_intent_seed0/models/model_best.pt"
CKPT_SLOT_V0="${BASE}/slot_intent_seed0/models/model_best.pt"
CKPT_SLOT_V2="${BASE}/slot_intent_seed0_v2/models/model_best.pt"
CKPT_SLOT_V3="${BASE}/slot_intent_seed0_v3/models/model_best.pt"
CKPT_SLOT_V4="${BASE}/slot_intent_seed0_v4/models/model_best.pt"
CKPT_SLOT_V5="${BASE}/slot_intent_seed0_v5/models/model_best.pt"
CKPT_SLOT_V6="${BASE}/slot_intent_seed0_v6/models/model_best.pt"
CKPT_SLOT_V7="${BASE}/slot_intent_seed0_v7/models/model_latest.pt"
CKPT_SLOT_V8="${BASE}/slot_intent_seed0_v8/models/model_best.pt"

OUT_PKL="rollouts/lift_mh_image_multimodal.pkl"
OUT_DIR="rollouts/lift_mh_image_cluster_figs"

echo "============================================================"
echo "Step 1: Collecting multimodal viz data"
echo "============================================================"

python examples/collect_multimodal_viz.py \
    --task lift_mh_image \
    --run "baseline:${CKPT_BL}:task=lift_mh_image" \
    --run "flow_intent:${CKPT_FI}:task=lift_mh_image_flow_intent:network=mlp_flow_intent" \
    --run "flow_intent_h16:${CKPT_FI_H16}:task=lift_mh_image_flow_intent_h16:network=mlp_flow_intent" \
    --run "flow_intent_h32:${CKPT_FI_H32}:task=lift_mh_image_flow_intent_h32:network=mlp_flow_intent" \
    --run "flow_intent_h64:${CKPT_FI_H64}:task=lift_mh_image_flow_intent_h64:network=mlp_flow_intent" \
    --run "flow_intent_h128:${CKPT_FI_H128}:task=lift_mh_image_flow_intent_h128:network=mlp_flow_intent" \
    --run "hierarchical_emb:${CKPT_HIEREMB}:task=lift_mh_image_hierarchical_emb:network=mlp" \
    --run "cnn_intent:${CKPT_CNN}:task=lift_mh_image_cnn_intent:network=mlp_flow_intent" \
    --run "slot_intent_v0:${CKPT_SLOT_V0}:task=lift_mh_image_slot_intent:network=mlp_flow_intent" \
    --run "slot_intent_v2:${CKPT_SLOT_V2}:task=lift_mh_image_slot_intent:network=mlp_flow_intent" \
    --run "slot_intent_v3:${CKPT_SLOT_V3}:task=lift_mh_image_slot_intent:network=mlp_flow_intent" \
    --run "slot_intent_v4:${CKPT_SLOT_V4}:task=lift_mh_image_slot_intent:network=mlp_flow_intent" \
    --run "slot_intent_v5:${CKPT_SLOT_V5}:task=lift_mh_image_slot_intent:network=mlp_flow_intent" \
    --run "slot_intent_v6:${CKPT_SLOT_V6}:task=lift_mh_image_slot_intent:network=mlp_flow_intent" \
    --run "slot_intent_v7:${CKPT_SLOT_V7}:task=lift_mh_image_slot_intent_v7:network=mlp_flow_intent" \
    --run "slot_intent_v8:${CKPT_SLOT_V8}:task=lift_mh_image_slot_intent_v8:network=mlp_flow_intent" \
    --n-rollouts 20 \
    --n-samples 50 \
    --n-critical 10 \
    --n-probe 10 \
    --device cuda \
    --out "${OUT_PKL}"

echo "============================================================"
echo "Step 2: Spectral clustering analysis (all variants) → ${OUT_DIR}"
echo "============================================================"

python examples/analyze_multimodal_clustering.py \
    --data "${OUT_PKL}" \
    --task lift_mh_image \
    --out-dir "${OUT_DIR}" \
    --k-min 2 \
    --k-max 6 \
    --n-cols 3

echo "Done. Figures saved to: ${OUT_DIR}"

echo "============================================================"
echo "Step 3: Filtered clustering analysis (slot_intent comparison) → ${OUT_DIR}_slot_cmp"
echo "============================================================"

python examples/analyze_multimodal_clustering.py \
    --data "${OUT_PKL}" \
    --task lift_mh_image \
    --out-dir "${OUT_DIR}_slot_cmp" \
    --k-min 2 \
    --k-max 6 \
    --n-cols 3 \
    --labels baseline flow_intent \
             slot_intent_v0 slot_intent_v2 slot_intent_v3 slot_intent_v4 \
             slot_intent_v5 slot_intent_v6 slot_intent_v7 slot_intent_v8 \
             cnn_intent

echo "Done. Filtered figures saved to: ${OUT_DIR}_slot_cmp"
