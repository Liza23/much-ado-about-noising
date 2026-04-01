#!/bin/bash
#SBATCH --job-name=mm-pusht-image
#SBATCH --output=slrum-commands/logs/multimodal_pusht_image_%j.out
#SBATCH --error=slrum-commands/logs/multimodal_pusht_image_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00

mkdir -p slrum-commands/logs rollouts

cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

python examples/collect_multimodal_viz.py \
    --task pusht_image \
    --run "baseline:checkpoints/pusht_ph_image_flow_mlp_512_seed0_success91.pt:task=pusht_image" \
    --run "flow_intent:checkpoints/pusht_ph_image_flow_mlp_512_seed0_flow_intent_success75.pt:task=pusht_image_flow_intent:+network.arch_variant=flow_intent" \
    --run "hierarchical_emb:checkpoints/pusht_ph_image_flow_mlp_512_seed0_hierarchical_emb_learned_success88.pt:task=pusht_image_hierarchical_emb" \
    --n-rollouts 20 --n-samples 50 --n-critical 10 \
    --device cuda \
    --out rollouts/pusht_image_multimodal.pkl

python examples/analyze_multimodal_viz.py \
    --data rollouts/pusht_image_multimodal.pkl \
    --task pusht_image \
    --out-dir rollouts/pusht_image_multimodal_figs

python examples/analyze_multimodal_clustering.py \
    --data rollouts/pusht_image_multimodal.pkl \
    --task pusht_image \
    --out-dir rollouts/pusht_image_cluster_figs \
    --k-min 2 --k-max 6 --n-ghost-states 5 \
    --vel-dims 2 --abs-action
