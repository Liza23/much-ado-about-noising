#!/bin/bash
#SBATCH --job-name=eval-lift-mh-image-sr
#SBATCH --output=slrum-commands/logs/eval_lift_mh_image_sr_%j.out
#SBATCH --error=slrum-commands/logs/eval_lift_mh_image_sr_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3:00:00

mkdir -p slrum-commands/logs results

cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

python examples/eval_success_rate.py \
    --run "baseline:/data/user_data/ldahiya/mip_render/lift-mh-image/baseline/models/model_best.pt:task=lift_mh_image" \
    --run "flow_intent:/data/user_data/ldahiya/mip_render/lift-mh-image/flow_intent/models/model_best.pt:task=lift_mh_image_flow_intent:+network.arch_variant=flow_intent" \
    --run "flow_intent_h32:/data/user_data/ldahiya/mip_render/lift-mh-image/flow_intent_h32/models/model_best.pt:task=lift_mh_image_flow_intent_h32:+network.arch_variant=flow_intent" \
    --run "hierarchical_emb:/data/user_data/ldahiya/mip_render/lift-mh-image/hierarchical_emb/models/model_best.pt:task=lift_mh_image_hierarchical_emb" \
    --run "flow_intent_h16:/data/user_data/ldahiya/mip_render/lift-mh-image/flow_intent_h16/models/model_best.pt:task=lift_mh_image_flow_intent_h16:+network.arch_variant=flow_intent" \
    --run "flow_intent_h64:/data/user_data/ldahiya/mip_render/lift-mh-image/flow_intent_h64/models/model_best.pt:task=lift_mh_image_flow_intent_h64:+network.arch_variant=flow_intent" \
    --run "flow_intent_h128:/data/user_data/ldahiya/mip_render/lift-mh-image/flow_intent_h128/models/model_best.pt:task=lift_mh_image_flow_intent_h128:+network.arch_variant=flow_intent" \
    --n-rollouts 100 \
    --device cuda \
    --out results/lift_mh_image_success_rate.csv
