#!/bin/bash
#SBATCH --job-name=mm-kitchen-state
#SBATCH --output=slrum-commands/logs/multimodal_kitchen_state_%j.out
#SBATCH --error=slrum-commands/logs/multimodal_kitchen_state_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00

set -euo pipefail

mkdir -p slrum-commands/logs rollouts

cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

NUMBA_CACHE_DIR=/tmp MPLCONFIGDIR=/tmp/matplotlib MUJOCO_GL=egl python examples/collect_multimodal_viz.py \
    --task kitchen_state \
    --run "baseline:/data/user_data/ldahiya/mip_render/kitchen-state/baseline/models/model_best.pt:task=kitchen_state" \
    --run "baseline_chiunet:/data/user_data/ldahiya/mip_render/kitchen-state/baseline_chiunet/models/model_best.pt:task=kitchen_state:network=chiunet" \
    --run "flow_intent:/data/user_data/ldahiya/mip_render/kitchen-state/flow_intent/models/model_best.pt:task=kitchen_state_flow_intent:++network.arch_variant=flow_intent" \
    --run "flow_intent_chiunet:/data/user_data/ldahiya/mip_render/kitchen-state/flow_intent_chiunet_action/models/model_best.pt:task=kitchen_state_flow_intent:network=mlp_flow_intent_chiunet:++network.arch_variant=flow_intent_chiunet" \
    --run "hierarchical_emb:checkpoints/kitchen-all-v0_multitask_state_flow_mlp_512_seed0_intent_learned_joint_emb_success88.pt:task=kitchen_state_hierarchical_emb" \
    --run "hierarchical_emb_chiunet:checkpoints/kitchen-all-v0_multitask_state_flow_chiunet_256_seed0_intent_learned_joint_emb_success100.pt:task=kitchen_state_hierarchical_emb:network=chiunet" \
    --n-rollouts 20 --n-samples 50 --n-critical 10 \
    --device cuda \
    --out rollouts/kitchen_state_multimodal.pkl

NUMBA_CACHE_DIR=/tmp MPLCONFIGDIR=/tmp/matplotlib MUJOCO_GL=egl python examples/analyze_multimodal_viz.py \
    --data rollouts/kitchen_state_multimodal.pkl \
    --task kitchen_state \
    --out-dir rollouts/kitchen_state_multimodal_figs

NUMBA_CACHE_DIR=/tmp MPLCONFIGDIR=/tmp/matplotlib MUJOCO_GL=egl python examples/analyze_multimodal_clustering.py \
    --data rollouts/kitchen_state_multimodal.pkl \
    --task kitchen_state \
    --out-dir rollouts/kitchen_state_cluster_figs \
    --k-min 2 --k-max 6 --n-ghost-states 5
