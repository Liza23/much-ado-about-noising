#!/bin/bash
#SBATCH --job-name=intent-sensitivity
#SBATCH --output=slrum-commands/logs/intent_sensitivity_%j.out
#SBATCH --error=slrum-commands/logs/intent_sensitivity_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00

set -euo pipefail

mkdir -p slrum-commands/logs

cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

export NUMBA_CACHE_DIR=/tmp
export MPLCONFIGDIR=/tmp/matplotlib

CKPT_FI="checkpoints/lift_mh_state_flow_mlp_512_h10_seed0_intent_flow_intent_success100.pt"
CKPT_FI_SAMPLED="/data/user_data/ldahiya/mip_render/lift-mh-state/flow_intent_sampled/models/model_latest.pt"

python examples/intent_sensitivity.py \
    --run "flow_intent:${CKPT_FI}:task=lift_mh_state_flow_intent:network=mlp_flow_intent" \
    --run "flow_intent_sampled:${CKPT_FI_SAMPLED}:task=lift_mh_state_flow_intent:network=mlp_flow_intent:+task.decoder_uses_sampled_intent=true" \
    --n-obs 256 \
    --n-pairs 16 \
    --num-steps 9 \
    --device cuda
