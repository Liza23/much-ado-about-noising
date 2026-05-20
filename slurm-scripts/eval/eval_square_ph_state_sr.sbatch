#!/bin/bash
#SBATCH --job-name=eval-sq-ph-sr
#SBATCH --output=slrum-commands/logs/eval_square_ph_state_sr_%j.out
#SBATCH --error=slrum-commands/logs/eval_square_ph_state_sr_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3:00:00

mkdir -p slrum-commands/logs results

cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

python examples/eval_success_rate.py \
    --run "baseline:checkpoints/square_ph_state_flow_mlp_512_h10_seed0_success88.pt:task=square_ph_state" \
    --run "flow_intent:checkpoints/square_ph_state_flow_mlp_512_h10_seed0_intent_flow_intent_success98.pt:task=square_ph_state_flow_intent:+network.arch_variant=flow_intent" \
    --run "hierarchical_emb:checkpoints/square_ph_state_flow_mlp_512_h10_seed0_intent_learned_joint_emb_success92.pt:task=square_ph_state_hierarchical_emb" \
    --run "flow_intent_h32:/data/user_data/ldahiya/mip_render/square-ph-state/flow_intent_h32/models/model_best.pt:task=square_ph_state_flow_intent_h32:+network.arch_variant=flow_intent" \
    --n-rollouts 100 \
    --device cuda \
    --out results/square_ph_state_success_rate.csv
