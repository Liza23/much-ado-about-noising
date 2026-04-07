#!/bin/bash
#SBATCH --job-name=square-ph-image-intent-outcomes
#SBATCH --output=slrum-commands/logs/collect_square_ph_image_intent_outcomes_%j.out
#SBATCH --error=slrum-commands/logs/collect_square_ph_image_intent_outcomes_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=6:00:00

set -euo pipefail

mkdir -p slrum-commands/logs rollouts

cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

export NUMBA_CACHE_DIR=/tmp
export MPLCONFIGDIR=/tmp/matplotlib
export MUJOCO_GL=egl
export XDG_RUNTIME_DIR=/tmp

declare -a LABELS=(
    "flow_intent"
)

declare -a RUN_SPECS=(
    "flow_intent:checkpoints/square_ph_image_flow_mlp_512_h10_seed0_intent_flow_intent_success90.pt:task=square_ph_image_flow_intent:+network.arch_variant=flow_intent"
)

for idx in "${!LABELS[@]}"; do
    label="${LABELS[$idx]}"
    run_spec="${RUN_SPECS[$idx]}"
    out_path="rollouts/square_ph_image_intent_outcomes_${label}.pkl"
    out_dir="rollouts/square_ph_image_intent_outcomes_${label}_figs"

    mkdir -p "$out_dir"

    echo "============================================================"
    echo "Running image intent-outcomes collection for ${label}"
    echo "Run spec: ${run_spec}"
    echo "Output: ${out_path}"
    echo "Figures: ${out_dir}"
    echo "============================================================"

    python examples/collect_intent_outcomes_image.py \
        --run "${run_spec}" \
        --n-rollouts 10 \
        --n-critical 5 \
        --max-timestep 100 \
        --n-intents 50 \
        --k-clusters 3 \
        --n-future-steps 80 \
        --n-trials 20 \
        --render \
        --plot-intent-xyz \
        --device cuda \
        --out "${out_path}" \
        --out-dir "${out_dir}"
done
