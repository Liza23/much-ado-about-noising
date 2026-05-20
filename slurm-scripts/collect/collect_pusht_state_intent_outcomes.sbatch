#!/bin/bash
#SBATCH --job-name=pusht-state-intent-outcomes
#SBATCH --output=slrum-commands/logs/collect_pusht_state_intent_outcomes_%j.out
#SBATCH --error=slrum-commands/logs/collect_pusht_state_intent_outcomes_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail

mkdir -p slrum-commands/logs rollouts

cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

export MPLCONFIGDIR=/tmp/matplotlib
export XDG_RUNTIME_DIR=/tmp

declare -a LABELS=(
    "flow_intent"
)

declare -a RUN_SPECS=(
    "flow_intent:/data/user_data/ldahiya/pusht_baselines/state_flow_intent/models/model_best.pt:task=pusht_state_flow_intent:+network.arch_variant=flow_intent"
)

for idx in "${!LABELS[@]}"; do
    label="${LABELS[$idx]}"
    run_spec="${RUN_SPECS[$idx]}"
    out_path="rollouts/pusht_state_intent_outcomes_${label}.pkl"
    out_dir="rollouts/pusht_state_intent_outcomes_${label}_figs_2d"

    mkdir -p "$out_dir"

    echo "============================================================"
    echo "Running intent-outcomes collection for ${label}"
    echo "Run spec: ${run_spec}"
    echo "Output: ${out_path}"
    echo "Figures: ${out_dir}"
    echo "============================================================"

    python examples/collect_intent_outcomes.py \
        --run "${run_spec}" \
        --n-rollouts 1 \
        --n-critical 5 \
        --max-timestep 80 \
        --n-intents 50 \
        --k-clusters 2 \
        --n-future-steps 200 \
        --n-trials 20 \
        --render \
        --device cuda \
        --out "${out_path}" \
        --out-dir "${out_dir}"
done
