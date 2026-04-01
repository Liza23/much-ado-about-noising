#!/bin/bash
#SBATCH --job-name=mm-sq-ph-image
#SBATCH --output=slrum-commands/logs/multimodal_square_ph_image_%j.out
#SBATCH --error=slrum-commands/logs/multimodal_square_ph_image_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00

mkdir -p slrum-commands/logs rollouts

cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

python examples/collect_multimodal_viz.py \
    --task square_ph_image \
    --run "baseline:checkpoints/square_ph_image_flow_mlp_512_h10_seed0_success96.pt:task=square_ph_image" \
    --run "flow_intent:checkpoints/square_ph_image_flow_mlp_512_h10_seed0_intent_flow_intent_success90.pt:task=square_ph_image_flow_intent:+network.arch_variant=flow_intent" \
    --run "hierarchical_emb:checkpoints/square_ph_image_flow_mlp_512_h10_seed0_intent_learned_joint_emb_success94.pt:task=square_ph_image_hierarchical_emb" \
    --n-rollouts 20 --n-samples 50 --n-critical 10 \
    --device cuda \
    --out rollouts/square_ph_image_multimodal.pkl

python examples/analyze_multimodal_viz.py \
    --data rollouts/square_ph_image_multimodal.pkl \
    --task square_ph_image \
    --out-dir rollouts/square_ph_image_multimodal_figs

python examples/analyze_multimodal_clustering.py \
    --data rollouts/square_ph_image_multimodal.pkl \
    --task square_ph_image \
    --out-dir rollouts/square_ph_image_cluster_figs \
    --k-min 2 --k-max 6 --n-ghost-states 5
