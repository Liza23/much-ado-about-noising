#!/bin/bash
#SBATCH --job-name=lift-ph-state-intent-outcomes
#SBATCH --output=slrum-commands/logs/collect_lift_ph_state_intent_outcomes_%j.out
#SBATCH --error=slrum-commands/logs/collect_lift_ph_state_intent_outcomes_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail

mkdir -p slrum-commands/logs rollouts

cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

export NUMBA_CACHE_DIR=/tmp
export MPLCONFIGDIR=/tmp/matplotlib
export MUJOCO_GL=egl
export XDG_RUNTIME_DIR=/tmp

declare -a LABELS=(
    "baseline"
    "flow_intent"
    "hierarchical_emb"
)

declare -a RUN_SPECS=(
    "baseline:checkpoints/lift_ph_state_flow_mlp_512_h16_seed0_success0.pt:task=lift_ph_state"
    "flow_intent:checkpoints/lift_ph_state_flow_mlp_512_h16_seed0_intent_flow_intent_success100.pt:task=lift_ph_state_flow_intent:+network.arch_variant=flow_intent"
    "hierarchical_emb:checkpoints/lift_ph_state_flow_mlp_512_h16_seed0_intent_learned_joint_emb_success98.pt:task=lift_ph_state_hierarchical_emb"
)

for idx in "${!LABELS[@]}"; do
    label="${LABELS[$idx]}"
    run_spec="${RUN_SPECS[$idx]}"
    out_path="rollouts/lift_ph_state_intent_outcomes_${label}.pkl"
    out_dir="rollouts/lift_ph_state_intent_outcomes_${label}_figs"

    mkdir -p "$out_dir"

    echo "============================================================"
    echo "Running intent-outcomes collection for ${label}"
    echo "Run spec: ${run_spec}"
    echo "Output: ${out_path}"
    echo "Figures: ${out_dir}"
    echo "============================================================"

    python examples/collect_intent_outcomes.py \
        --run "${run_spec}" \
        --n-rollouts 10 \
        --n-critical 5 \
        --max-timestep 80 \
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
