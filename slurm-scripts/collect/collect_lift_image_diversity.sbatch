#!/bin/bash
#SBATCH --job-name=lift-image-diversity
#SBATCH --output=slrum-commands/logs/lift_image_diversity_%j.out
#SBATCH --error=slrum-commands/logs/lift_image_diversity_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00

mkdir -p slrum-commands/logs rollouts

cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

python examples/collect_diversity_rollouts.py \
    --task lift_ph_image \
    --run "baseline:checkpoints/lift_ph_image_flow_mlp_512_h16_seed0_success100.pt:task=lift_ph_image" \
    --run "hierarchical_emb:checkpoints/lift_ph_image_flow_mlp_512_h16_seed0_intent_learned_joint_emb_success100.pt:task=lift_ph_image_hierarchical_emb" \
    --run "flow_intent:checkpoints/lift_ph_image_flow_mlp_512_h16_seed0_intent_flow_intent_success100.pt:task=lift_ph_image_flow_intent:+network.arch_variant=flow_intent" \
    --n-rollouts 100 --device cuda \
    --out rollouts/lift_ph_image_diversity.pkl

python examples/analyze_diversity.py \
    --data rollouts/lift_ph_image_diversity.pkl \
    --task lift_ph_image \
    --out-dir rollouts/lift_ph_image_figs
