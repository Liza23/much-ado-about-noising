#!/bin/bash
#SBATCH --job-name=pusht-intent-shift
#SBATCH --output=slrum-commands/logs/plot_pusht_intent_shift_%j.out
#SBATCH --error=slrum-commands/logs/plot_pusht_intent_shift_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

mkdir -p slrum-commands/logs plots

cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

export MPLCONFIGDIR=/tmp/matplotlib
export XDG_RUNTIME_DIR=/tmp

RUN_SPEC="flow_intent:/data/user_data/ldahiya/pusht_baselines/state_flow_intent/models/model_best.pt:task=pusht_state_flow_intent:+network.arch_variant=flow_intent"
OUT_FIG="plots/pusht_intent_shift.png"
OUT_DATA="plots/pusht_intent_shift.pkl"

echo "============================================================"
echo "Running PushT intent-shift plot"
echo "Run spec: ${RUN_SPEC}"
echo "Figure: ${OUT_FIG}"
echo "Data:   ${OUT_DATA}"
echo "============================================================"

python examples/plot_pusht_intent_shift.py \
    --run "${RUN_SPEC}" \
    --env-seed 0 \
    --perturb-dx=-50 \
    --perturb-dy=-50 \
    --perturb-dtheta=-10.0 \
    --n-intent-samples 32 \
    --out "${OUT_FIG}" \
    --out-data "${OUT_DATA}"
