#!/bin/bash
#SBATCH --job-name=kitchen-diversity
#SBATCH --output=slrum-commands/logs/kitchen_state_diversity_%j.out
#SBATCH --error=slrum-commands/logs/kitchen_state_diversity_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00

mkdir -p slrum-commands/logs rollouts

cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

NUMBA_CACHE_DIR=/tmp MUJOCO_GL=egl python examples/collect_diversity_rollouts.py \
    --task kitchen_state \
    --run "baseline:/data/user_data/ldahiya/mip_render/kitchen-state/baseline/models/model_best.pt:task=kitchen_state" \
    --run "baseline_chiunet:/data/user_data/ldahiya/mip_render/kitchen-state/baseline_chiunet/models/model_best.pt:task=kitchen_state:network=chiunet" \
    --run "flow_intent:/data/user_data/ldahiya/mip_render/kitchen-state/flow_intent/models/model_best.pt:task=kitchen_state_flow_intent:++network.arch_variant=flow_intent" \
    --run "flow_intent_chiunet_action:/data/user_data/ldahiya/mip_render/kitchen-state/flow_intent_chiunet_action/models/model_best.pt:task=kitchen_state_flow_intent:network=mlp_flow_intent_chiunet" \
    --n-rollouts 100 --device cuda \
    --out rollouts/kitchen_state_diversity.pkl

NUMBA_CACHE_DIR=/tmp MUJOCO_GL=egl python examples/analyze_diversity.py \
    --data rollouts/kitchen_state_diversity.pkl \
    --task kitchen_state \
    --out-dir rollouts/kitchen_state_figs
