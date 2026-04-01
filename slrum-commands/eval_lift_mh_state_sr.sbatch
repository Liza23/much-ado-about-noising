#!/bin/bash
#SBATCH --job-name=eval-lift-mh-sr
#SBATCH --output=slrum-commands/logs/eval_lift_mh_state_sr_%j.out
#SBATCH --error=slrum-commands/logs/eval_lift_mh_state_sr_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3:00:00

mkdir -p slrum-commands/logs results

cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

python examples/eval_success_rate.py \
    --run "baseline:checkpoints/lift_mh_state_flow_mlp_512_h10_seed0_success100.pt:task=lift_mh_state" \
    --run "flow_intent:checkpoints/lift_mh_state_flow_mlp_512_h10_seed0_intent_flow_intent_success100.pt:task=lift_mh_state_flow_intent:+network.arch_variant=flow_intent" \
    --run "hierarchical_emb:checkpoints/lift_mh_state_flow_mlp_512_h10_seed0_intent_learned_joint_emb_success98.pt:task=lift_mh_state_hierarchical_emb" \
    --run "flow_intent_h32:/data/user_data/ldahiya/mip_render/lift-mh-state/flow_intent_h32/models/model_best.pt:task=lift_mh_state_flow_intent_emb_h32:+network.arch_variant=flow_intent" \
    --run "flow_intent_h64:/data/user_data/ldahiya/mip_render/lift-mh-state/flow_intent_h64/models/model_best.pt:task=lift_mh_state_flow_intent_emb_h64:+network.arch_variant=flow_intent" \
    --run "flow_intent_h128:/data/user_data/ldahiya/mip_render/lift-mh-state/flow_intent_h128/models/model_best.pt:task=lift_mh_state_flow_intent_emb_h128:+network.arch_variant=flow_intent" \
    --run "flow_intent_h16:/data/user_data/ldahiya/mip_render/lift-mh-state/flow_intent_h16/models/model_best.pt:task=lift_mh_state_flow_intent_emb_h16:+network.arch_variant=flow_intent" \
    --n-rollouts 100 \
    --device cuda \
    --out results/lift_mh_state_success_rate.csv
