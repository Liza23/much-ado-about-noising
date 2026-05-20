#!/bin/bash
#SBATCH --job-name=cluster-lift-mh-state
#SBATCH --output=slrum-commands/logs/cluster_lift_mh_state_%j.out
#SBATCH --error=slrum-commands/logs/cluster_lift_mh_state_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00

set -euo pipefail

mkdir -p slrum-commands/logs rollouts/lift_mh_state_cluster_figs

cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

export NUMBA_CACHE_DIR=/tmp
export MPLCONFIGDIR=/tmp/matplotlib
export MUJOCO_GL=egl
export XDG_RUNTIME_DIR=/tmp

CKPT_FI="checkpoints/lift_mh_state_flow_mlp_512_h10_seed0_intent_flow_intent_success100.pt"
CKPT_FI_CURR="/data/user_data/ldahiya/mip_render/lift-mh-state/flow_intent_curriculum/models/model_best.pt"
CKPT_FI_CURR_50K="/data/user_data/ldahiya/mip_render/lift-mh-state/flow_intent_curriculum_50k/models/model_best.pt"
CKPT_BL="checkpoints/lift_mh_state_flow_mlp_512_h10_seed0_success100.pt"
CKPT_MIP_RENDER="checkpoints/lift_mh_state_mip_render_mlp_512_h10_seed0_success100.pt"
CKPT_PARL="/data/user_data/ldahiya/mip_render/lift-mh-state/residual_parl/residual_parl_checkpoints/best.pt"
CKPT_FI_FOR_PARL="/data/user_data/ldahiya/mip_render/lift-mh-state/flow_intent/models/model_best.pt"
CKPT_POSTBC="/data/user_data/ldahiya/mip_render/lift-mh-state/postbc_flow_bc/models/model_best.pt"
CKPT_DSRL="/data/user_data/ldahiya/mip_render/lift-mh-state/flow_intent_dsrl/dsrl_checkpoints/best.pt"
CKPT_SAC="/data/user_data/ldahiya/mip_render/lift-mh-state/intent_flow_residual_sac/residual_sac_checkpoints/best.pt"
CKPT_PLAIN_DSRL="/data/user_data/ldahiya/mip_render/lift-mh-state/plain_flow_dsrl/dsrl_checkpoints/best.pt"
CKPT_DUAL="checkpoints/lift_mh_state_flow_mlp_512_h10_seed0_intent_learned_dual_success100.pt"

OUT_PKL="rollouts/lift_mh_state_multimodal.pkl"
OUT_DIR="rollouts/lift_mh_state_cluster_figs"

echo "============================================================"
echo "Step 1: Collecting multimodal viz data"
echo "============================================================"

python examples/collect_multimodal_viz.py \
    --task lift_mh_state \
    --run "flow_intent:${CKPT_FI}:task=lift_mh_state_flow_intent:network=mlp_flow_intent" \
    --run "curriculum_100k:${CKPT_FI_CURR}:task=lift_mh_state_flow_intent:network=mlp_flow_intent:+task.decoder_uses_sampled_intent=true:+task.decoder_curriculum_steps=100000" \
    --run "curriculum_50k:${CKPT_FI_CURR_50K}:task=lift_mh_state_flow_intent:network=mlp_flow_intent:+task.decoder_uses_sampled_intent=true:+task.decoder_curriculum_steps=50000" \
    --run "baseline_stochastic:${CKPT_BL}:task=lift_mh_state:network=mlp:optimization.sample_mode=stochastic" \
    --run "baseline_zero:${CKPT_BL}:task=lift_mh_state:network=mlp:optimization.sample_mode=zero" \
    --run "mip_render:${CKPT_MIP_RENDER}:task=lift_mh_state:network=mlp:optimization.loss_type=mip_render:optimization.sample_mode=stochastic" \
    --run "residual_parl:${CKPT_PARL}:task=lift_mh_state_flow_intent:network=mlp_flow_intent:flow_intent_ckpt=${CKPT_FI_FOR_PARL}" \
    --run "intent_dsrl:${CKPT_DSRL}:task=lift_mh_state_flow_intent:network=mlp_flow_intent:flow_intent_ckpt=${CKPT_FI_FOR_PARL}:agent_type=dsrl" \
    --run "residual_sac:${CKPT_SAC}:task=lift_mh_state_flow_intent:network=mlp_flow_intent:flow_intent_ckpt=${CKPT_FI_FOR_PARL}:agent_type=residual_sac" \
    --run "plain_dsrl:${CKPT_PLAIN_DSRL}:task=lift_mh_state:network=mlp:flow_intent_ckpt=${CKPT_BL}:agent_type=plain_dsrl" \
    --run "postbc:${CKPT_POSTBC}:task=lift_mh_state_postbc:network=mlp" \
    --run "learned_dual:${CKPT_DUAL}:task=lift_mh_state_dual_intent:network=mlp" \
    --n-rollouts 20 \
    --n-samples 50 \
    --n-critical 10 \
    --n-probe 10 \
    --device cuda \
    --out "${OUT_PKL}"

echo "============================================================"
echo "Step 2: Spectral clustering analysis → ${OUT_DIR}"
echo "============================================================"

python examples/analyze_multimodal_clustering.py \
    --data "${OUT_PKL}" \
    --task lift_mh_state \
    --out-dir "${OUT_DIR}" \
    --k-min 2 \
    --k-max 6 \
    --n-cols 3

echo "Done. Figures saved to: ${OUT_DIR}"

echo "============================================================"
echo "Step 3: Filtered clustering analysis (RL comparison) → ${OUT_DIR}_rl_cmp"
echo "============================================================"

python examples/analyze_multimodal_clustering.py \
    --data "${OUT_PKL}" \
    --task lift_mh_state \
    --out-dir "${OUT_DIR}_rl_cmp" \
    --k-min 2 \
    --k-max 6 \
    --n-cols 3 \
    --labels flow_intent curriculum_100k curriculum_50k baseline_stochastic \
             plain_dsrl postbc intent_dsrl residual_sac

echo "Done. Filtered figures saved to: ${OUT_DIR}_rl_cmp"
