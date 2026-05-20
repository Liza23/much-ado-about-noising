#!/bin/bash
#SBATCH --job-name=critical-states-lift-mh
#SBATCH --output=slrum-commands/logs/critical_states_lift_mh_%j.out
#SBATCH --error=slrum-commands/logs/critical_states_lift_mh_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00

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

N_ROLLOUTS=20        # reference episodes to collect states from
N_NOISY=10           # perturbed re-rollouts per state
# noise_std in radians on arm joint angles:
#   0.05 rad ≈ 3°  (subtle)
#   0.10 rad ≈ 6°  (moderate)
SUBSAMPLE=2          # keep every 2nd state to reduce Phase 2 cost

echo "============================================================"
echo "Critical-state identification: lift-mh-state (physical perturbation)"
echo "n_rollouts=${N_ROLLOUTS}  n_noisy=${N_NOISY}"
echo "============================================================"

# ── Baseline — noise 0.05 rad ────────────────────────────────────────────────
echo ""
echo ">>> Baseline  noise=0.05 rad"
python examples/identify_critical_states.py \
    --run "baseline:${CKPT_BL}:task=lift_mh_state:network=mlp:optimization.sample_mode=stochastic" \
    --n-rollouts ${N_ROLLOUTS} \
    --n-noisy ${N_NOISY} \
    --noise-std 0.05 \
    --subsample-every ${SUBSAMPLE} \
    --device cuda \
    --seed 42 \
    --out rollouts/critical_states_lift_mh_baseline_n005.pkl \
    --out-dir rollouts/critical_states_lift_mh_baseline_n005_figs

# ── Baseline — noise 0.1 rad ─────────────────────────────────────────────────
echo ""
echo ">>> Baseline  noise=0.10 rad"
python examples/identify_critical_states.py \
    --run "baseline:${CKPT_BL}:task=lift_mh_state:network=mlp:optimization.sample_mode=stochastic" \
    --n-rollouts ${N_ROLLOUTS} \
    --n-noisy ${N_NOISY} \
    --noise-std 0.10 \
    --subsample-every ${SUBSAMPLE} \
    --device cuda \
    --seed 42 \
    --out rollouts/critical_states_lift_mh_baseline_n010.pkl \
    --out-dir rollouts/critical_states_lift_mh_baseline_n010_figs

# ── Flow-intent — noise 0.05 rad ──────────────────────────────────────────────
echo ""
echo ">>> Flow-intent  noise=0.05 rad"
python examples/identify_critical_states.py \
    --run "flow_intent:${CKPT_FI}:task=lift_mh_state_flow_intent:network=mlp_flow_intent" \
    --n-rollouts ${N_ROLLOUTS} \
    --n-noisy ${N_NOISY} \
    --noise-std 0.05 \
    --subsample-every ${SUBSAMPLE} \
    --device cuda \
    --seed 42 \
    --out rollouts/critical_states_lift_mh_fi_n005.pkl \
    --out-dir rollouts/critical_states_lift_mh_fi_n005_figs

# ── Flow-intent — noise 0.1 rad ───────────────────────────────────────────────
echo ""
echo ">>> Flow-intent  noise=0.10 rad"
python examples/identify_critical_states.py \
    --run "flow_intent:${CKPT_FI}:task=lift_mh_state_flow_intent:network=mlp_flow_intent" \
    --n-rollouts ${N_ROLLOUTS} \
    --n-noisy ${N_NOISY} \
    --noise-std 0.10 \
    --subsample-every ${SUBSAMPLE} \
    --device cuda \
    --seed 42 \
    --out rollouts/critical_states_lift_mh_fi_n010.pkl \
    --out-dir rollouts/critical_states_lift_mh_fi_n010_figs

echo ""
echo "============================================================"
echo "Done. Outputs:"
echo "  rollouts/critical_states_lift_mh_{baseline,fi}_{n005,n010}.pkl"
echo "  rollouts/critical_states_lift_mh_*_figs/"
echo "============================================================"
