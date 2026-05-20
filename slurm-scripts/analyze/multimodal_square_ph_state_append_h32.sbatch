#!/bin/bash
#SBATCH --job-name=mm-sq-ph-h32
#SBATCH --output=slrum-commands/logs/multimodal_square_ph_state_append_h32_%j.out
#SBATCH --error=slrum-commands/logs/multimodal_square_ph_state_append_h32_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00

mkdir -p slrum-commands/logs rollouts

cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

# Append flow_intent_h32 to existing square_ph_state pkl
python examples/collect_multimodal_viz.py \
    --task square_ph_state \
    --run "flow_intent_h32:/data/user_data/ldahiya/mip_render/square-ph-state/flow_intent_h32/models/model_best.pt:task=square_ph_state_flow_intent_h32:+network.arch_variant=flow_intent" \
    --n-rollouts 20 --n-samples 50 --n-critical 10 \
    --device cuda \
    --out rollouts/square_ph_state_multimodal.pkl \
    --append

# Regenerate figures
python examples/analyze_multimodal_viz.py \
    --data rollouts/square_ph_state_multimodal.pkl \
    --task square_ph_state \
    --out-dir rollouts/square_ph_state_multimodal_figs

python examples/analyze_multimodal_clustering.py \
    --data rollouts/square_ph_state_multimodal.pkl \
    --task square_ph_state \
    --out-dir rollouts/square_ph_state_cluster_figs \
    --k-min 2 --k-max 6 --n-ghost-states 5