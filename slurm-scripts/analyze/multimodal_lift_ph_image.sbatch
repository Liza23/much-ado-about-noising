#!/bin/bash
#SBATCH --job-name=mm-lift-ph-image
#SBATCH --output=slrum-commands/logs/multimodal_lift_ph_image_%j.out
#SBATCH --error=slrum-commands/logs/multimodal_lift_ph_image_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00

mkdir -p slrum-commands/logs rollouts

cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

python examples/collect_multimodal_viz.py \
    --task lift_ph_image \
    --run "baseline:checkpoints/lift_ph_image_flow_mlp_512_h16_seed0_success100.pt:task=lift_ph_image" \
    --run "flow_intent:checkpoints/lift_ph_image_flow_mlp_512_h16_seed0_intent_flow_intent_success100.pt:task=lift_ph_image_flow_intent:+network.arch_variant=flow_intent" \
    --run "hierarchical_emb:checkpoints/lift_ph_image_flow_mlp_512_h16_seed0_intent_learned_joint_emb_success100.pt:task=lift_ph_image_hierarchical_emb" \
    --n-rollouts 20 --n-samples 50 --n-critical 10 \
    --device cuda \
    --out rollouts/lift_ph_image_multimodal.pkl

python examples/analyze_multimodal_viz.py \
    --data rollouts/lift_ph_image_multimodal.pkl \
    --task lift_ph_image \
    --out-dir rollouts/lift_ph_image_multimodal_figs

python examples/analyze_multimodal_clustering.py \
    --data rollouts/lift_ph_image_multimodal.pkl \
    --task lift_ph_image \
    --out-dir rollouts/lift_ph_image_cluster_figs \
    --k-min 2 --k-max 6 --n-ghost-states 5
