#!/bin/bash
#SBATCH --job-name=eval-pusht-state-sr
#SBATCH --output=slrum-commands/logs/eval_pusht_state_sr_%j.out
#SBATCH --error=slrum-commands/logs/eval_pusht_state_sr_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3:00:00

mkdir -p slrum-commands/logs results

cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

python examples/eval_success_rate.py \
    --run "baseline:/data/user_data/ldahiya/pusht_baselines/state/models/model_best.pt:task=pusht_state" \
    --run "flow_intent:/data/user_data/ldahiya/pusht_baselines/state_flow_intent/models/model_best.pt:task=pusht_state_flow_intent:+network.arch_variant=flow_intent" \
    --run "hierarchical_emb:/data/user_data/ldahiya/pusht_baselines/state_hierarchical_emb/models/model_best.pt:task=pusht_state_hierarchical_emb" \
    --n-rollouts 100 \
    --device cuda \
    --out results/pusht_state_success_rate.csv
