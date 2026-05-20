#!/bin/bash
#SBATCH --job-name=diversity-states-lift-mh
#SBATCH --output=slrum-commands/logs/diversity_states_lift_mh_%j.out
#SBATCH --error=slrum-commands/logs/diversity_states_lift_mh_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00

set -euo pipefail

mkdir -p slrum-commands/logs rollouts

cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

export NUMBA_CACHE_DIR=/tmp
export MPLCONFIGDIR=/tmp/matplotlib
export MUJOCO_GL=egl
export XDG_RUNTIME_DIR=/tmp

CKPT_BL="checkpoints/lift_mh_state_flow_mlp_512_h10_seed0_success100.pt"
CKPT_FI="checkpoints/lift_mh_state_flow_mlp_512_h10_seed0_intent_flow_intent_success100.pt"

N_ROLLOUTS=20        # reference episodes
N_STOCHASTIC=10      # stochastic re-rollouts per state
SUBSAMPLE=2          # keep every 2nd state

echo "============================================================"
echo "Behavioral diversity states: lift-mh-state"
echo "n_rollouts=${N_ROLLOUTS}  n_stochastic=${N_STOCHASTIC}"
echo "============================================================"

# ── Baseline ─────────────────────────────────────────────────────────────────
echo ""
echo ">>> Baseline"
python examples/identify_diversity_states.py \
    --run "baseline:${CKPT_BL}:task=lift_mh_state:network=mlp:optimization.sample_mode=stochastic" \
    --n-rollouts ${N_ROLLOUTS} \
    --n-stochastic ${N_STOCHASTIC} \
    --subsample-every ${SUBSAMPLE} \
    --device cuda \
    --seed 42 \
    --out rollouts/diversity_lift_mh_baseline.pkl \
    --out-dir rollouts/diversity_lift_mh_baseline_figs

# ── Flow-intent ───────────────────────────────────────────────────────────────
echo ""
echo ">>> Flow-intent"
python examples/identify_diversity_states.py \
    --run "flow_intent:${CKPT_FI}:task=lift_mh_state_flow_intent:network=mlp_flow_intent" \
    --n-rollouts ${N_ROLLOUTS} \
    --n-stochastic ${N_STOCHASTIC} \
    --subsample-every ${SUBSAMPLE} \
    --device cuda \
    --seed 42 \
    --out rollouts/diversity_lift_mh_fi.pkl \
    --out-dir rollouts/diversity_lift_mh_fi_figs

echo ""
echo "============================================================"
echo "Done. Outputs:"
echo "  rollouts/diversity_lift_mh_{baseline,fi}.pkl"
echo "  rollouts/diversity_lift_mh_*_figs/"
echo "============================================================"
