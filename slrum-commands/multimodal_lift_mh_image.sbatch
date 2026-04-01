#!/bin/bash
#SBATCH --job-name=mm-lift-mh-image
#SBATCH --output=slrum-commands/logs/multimodal_lift_mh_image_%j.out
#SBATCH --error=slrum-commands/logs/multimodal_lift_mh_image_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00

mkdir -p slrum-commands/logs rollouts

cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

python examples/collect_multimodal_viz.py \
    --task lift_mh_image \
    --run "baseline:/data/user_data/ldahiya/mip_render/lift-mh-image/baseline/models/model_best.pt:task=lift_mh_image" \
    --run "flow_intent:/data/user_data/ldahiya/mip_render/lift-mh-image/flow_intent/models/model_best.pt:task=lift_mh_image_flow_intent:+network.arch_variant=flow_intent" \
    --run "flow_intent_h32:/data/user_data/ldahiya/mip_render/lift-mh-image/flow_intent_h32/models/model_best.pt:task=lift_mh_image_flow_intent_h32:+network.arch_variant=flow_intent" \
    --run "hierarchical_emb:/data/user_data/ldahiya/mip_render/lift-mh-image/hierarchical_emb/models/model_best.pt:task=lift_mh_image_hierarchical_emb" \
    --n-rollouts 20 --n-samples 50 --n-critical 10 \
    --device cuda \
    --out rollouts/lift_mh_image_multimodal.pkl

python examples/analyze_multimodal_viz.py \
    --data rollouts/lift_mh_image_multimodal.pkl \
    --task lift_mh_image \
    --out-dir rollouts/lift_mh_image_multimodal_figs

python examples/analyze_multimodal_clustering.py \
    --data rollouts/lift_mh_image_multimodal.pkl \
    --task lift_mh_image \
    --out-dir rollouts/lift_mh_image_cluster_figs \
    --k-min 2 --k-max 6 --n-ghost-states 5
