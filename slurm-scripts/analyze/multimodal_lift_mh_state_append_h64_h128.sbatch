#!/bin/bash
#SBATCH --job-name=mm-lift-mh-h64-h128
#SBATCH --output=slrum-commands/logs/multimodal_lift_mh_state_append_h64_h128_%j.out
#SBATCH --error=slrum-commands/logs/multimodal_lift_mh_state_append_h64_h128_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3:00:00

mkdir -p slrum-commands/logs rollouts

cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

# Append flow_intent_h16
python examples/collect_multimodal_viz.py \
    --task lift_mh_state \
    --run "flow_intent_h16:/data/user_data/ldahiya/mip_render/lift-mh-state/flow_intent_h16/models/model_best.pt:task=lift_mh_state_flow_intent_emb_h16:+network.arch_variant=flow_intent" \
    --n-rollouts 20 --n-samples 50 --n-critical 10 \
    --device cuda \
    --out rollouts/lift_mh_state_multimodal.pkl \
    --append

# Append flow_intent_h64
python examples/collect_multimodal_viz.py \
    --task lift_mh_state \
    --run "flow_intent_h64:/data/user_data/ldahiya/mip_render/lift-mh-state/flow_intent_h64/models/model_best.pt:task=lift_mh_state_flow_intent_emb_h64:+network.arch_variant=flow_intent" \
    --n-rollouts 20 --n-samples 50 --n-critical 10 \
    --device cuda \
    --out rollouts/lift_mh_state_multimodal.pkl \
    --append

# Append flow_intent_h128
python examples/collect_multimodal_viz.py \
    --task lift_mh_state \
    --run "flow_intent_h128:/data/user_data/ldahiya/mip_render/lift-mh-state/flow_intent_h128/models/model_best.pt:task=lift_mh_state_flow_intent_emb_h128:+network.arch_variant=flow_intent" \
    --n-rollouts 20 --n-samples 50 --n-critical 10 \
    --device cuda \
    --out rollouts/lift_mh_state_multimodal.pkl \
    --append

# Regenerate figures
python examples/analyze_multimodal_viz.py \
    --data rollouts/lift_mh_state_multimodal.pkl \
    --task lift_mh_state \
    --out-dir rollouts/lift_mh_state_multimodal_figs

python examples/analyze_multimodal_clustering.py \
    --data rollouts/lift_mh_state_multimodal.pkl \
    --task lift_mh_state \
    --out-dir rollouts/lift_mh_state_cluster_figs \
    --k-min 2 --k-max 6 --n-ghost-states 5
