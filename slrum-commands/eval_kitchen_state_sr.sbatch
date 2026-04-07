#!/bin/bash
#SBATCH --job-name=eval-kitchen-sr
#SBATCH --output=slrum-commands/logs/eval_kitchen_state_sr_%j.out
#SBATCH --error=slrum-commands/logs/eval_kitchen_state_sr_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00

mkdir -p slrum-commands/logs results

cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

NUMBA_CACHE_DIR=/tmp MUJOCO_GL=egl python examples/eval_success_rate.py \
    --run "baseline:/data/user_data/ldahiya/mip_render/kitchen-state/baseline/models/model_best.pt:task=kitchen_state" \
    --run "baseline_chiunet:/data/user_data/ldahiya/mip_render/kitchen-state/baseline_chiunet/models/model_best.pt:task=kitchen_state:network=chiunet" \
    --run "flow_intent:/data/user_data/ldahiya/mip_render/kitchen-state/flow_intent/models/model_best.pt:task=kitchen_state_flow_intent:++network.arch_variant=flow_intent" \
    --run "flow_intent_chiunet_action:/data/user_data/ldahiya/mip_render/kitchen-state/flow_intent_chiunet_action/models/model_best.pt:task=kitchen_state_flow_intent:network=mlp_flow_intent_chiunet" \
    --n-rollouts 100 \
    --device cuda \
    --out results/kitchen_state_success_rate.csv
# NOTE: hierarchical_emb and hierarchical_emb_chiunet omitted — their checkpoints
# lack intent_predictor_state, needed for intent_type=encoded_mean at eval time.
