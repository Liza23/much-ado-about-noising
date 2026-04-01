#!/bin/bash
#SBATCH --job-name=mm-pusht-state
#SBATCH --output=slrum-commands/logs/multimodal_pusht_state_%j.out
#SBATCH --error=slrum-commands/logs/multimodal_pusht_state_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00

mkdir -p slrum-commands/logs rollouts

cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

python examples/collect_multimodal_viz.py \
    --task pusht_state \
    --run "baseline:/data/user_data/ldahiya/pusht_baselines/state/models/model_best.pt:task=pusht_state" \
    --run "flow_intent:/data/user_data/ldahiya/pusht_baselines/state_flow_intent/models/model_best.pt:task=pusht_state_flow_intent:+network.arch_variant=flow_intent" \
    --run "hierarchical_emb:/data/user_data/ldahiya/pusht_baselines/state_hierarchical_emb/models/model_best.pt:task=pusht_state_hierarchical_emb" \
    --n-rollouts 20 --n-samples 50 --n-critical 10 \
    --device cuda \
    --out rollouts/pusht_state_multimodal.pkl

python examples/analyze_multimodal_viz.py \
    --data rollouts/pusht_state_multimodal.pkl \
    --task pusht_state \
    --out-dir rollouts/pusht_state_multimodal_figs

python examples/analyze_multimodal_clustering.py \
    --data rollouts/pusht_state_multimodal.pkl \
    --task pusht_state \
    --out-dir rollouts/pusht_state_cluster_figs \
    --k-min 2 --k-max 6 --n-ghost-states 5 \
    --vel-dims 2 --abs-action
