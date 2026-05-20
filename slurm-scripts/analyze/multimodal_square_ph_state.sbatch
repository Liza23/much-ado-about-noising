#!/bin/bash
#SBATCH --job-name=mm-sq-ph-state
#SBATCH --output=slrum-commands/logs/multimodal_square_ph_state_%j.out
#SBATCH --error=slrum-commands/logs/multimodal_square_ph_state_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00

mkdir -p slrum-commands/logs rollouts

cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

python examples/collect_multimodal_viz.py \
    --task square_ph_state \
    --run "baseline:checkpoints/square_ph_state_flow_mlp_512_h10_seed0_success88.pt:task=square_ph_state" \
    --run "flow_intent:checkpoints/square_ph_state_flow_mlp_512_h10_seed0_intent_flow_intent_success98.pt:task=square_ph_state_flow_intent:+network.arch_variant=flow_intent" \
    --run "hierarchical_emb:checkpoints/square_ph_state_flow_mlp_512_h10_seed0_intent_learned_joint_emb_success92.pt:task=square_ph_state_hierarchical_emb" \
    --n-rollouts 20 --n-samples 50 --n-critical 10 \
    --device cuda \
    --out rollouts/square_ph_state_multimodal.pkl

python examples/analyze_multimodal_viz.py \
    --data rollouts/square_ph_state_multimodal.pkl \
    --task square_ph_state \
    --out-dir rollouts/square_ph_state_multimodal_figs

python examples/analyze_multimodal_clustering.py \
    --data rollouts/square_ph_state_multimodal.pkl \
    --task square_ph_state \
    --out-dir rollouts/square_ph_state_cluster_figs \
    --k-min 2 --k-max 6 --n-ghost-states 5
