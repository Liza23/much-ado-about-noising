#!/bin/bash
#SBATCH --job-name=eval-pusht-image-sr
#SBATCH --output=slrum-commands/logs/eval_pusht_image_sr_%j.out
#SBATCH --error=slrum-commands/logs/eval_pusht_image_sr_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3:00:00

mkdir -p slrum-commands/logs results

cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

python examples/eval_success_rate.py \
    --run "baseline:checkpoints/pusht_ph_image_flow_mlp_512_seed0_success91.pt:task=pusht_image" \
    --run "flow_intent:checkpoints/pusht_ph_image_flow_mlp_512_seed0_flow_intent_success75.pt:task=pusht_image_flow_intent:+network.arch_variant=flow_intent" \
    --run "hierarchical_emb:checkpoints/pusht_ph_image_flow_mlp_512_seed0_hierarchical_emb_learned_success88.pt:task=pusht_image_hierarchical_emb" \
    --n-rollouts 100 \
    --device cuda \
    --out results/pusht_image_success_rate.csv
