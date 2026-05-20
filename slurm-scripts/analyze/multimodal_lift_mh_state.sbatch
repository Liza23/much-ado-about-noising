#!/bin/bash
#SBATCH --job-name=mm-lift-mh-state
#SBATCH --output=slrum-commands/logs/multimodal_lift_mh_state_%j.out
#SBATCH --error=slrum-commands/logs/multimodal_lift_mh_state_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00

mkdir -p slrum-commands/logs rollouts

cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

python examples/collect_multimodal_viz.py \
    --task lift_mh_state \
    --run "baseline:checkpoints/lift_mh_state_flow_mlp_512_h10_seed0_success100.pt:task=lift_mh_state" \
    --run "flow_intent:checkpoints/lift_mh_state_flow_mlp_512_h10_seed0_intent_flow_intent_success100.pt:task=lift_mh_state_flow_intent:+network.arch_variant=flow_intent" \
    --run "hierarchical_emb:checkpoints/lift_mh_state_flow_mlp_512_h10_seed0_intent_learned_joint_emb_success98.pt:task=lift_mh_state_hierarchical_emb" \
    --n-rollouts 20 --n-samples 50 --n-critical 10 \
    --device cuda \
    --out rollouts/lift_mh_state_multimodal.pkl

python examples/analyze_multimodal_viz.py \
    --data rollouts/lift_mh_state_multimodal.pkl \
    --task lift_mh_state \
    --out-dir rollouts/lift_mh_state_multimodal_figs

python examples/analyze_multimodal_clustering.py \
    --data rollouts/lift_mh_state_multimodal.pkl \
    --task lift_mh_state \
    --out-dir rollouts/lift_mh_state_cluster_figs \
    --k-min 2 --k-max 6 --n-ghost-states 5
