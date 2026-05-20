#!/bin/bash
#SBATCH --job-name=div-vs-crit-fi
#SBATCH --output=slrum-commands/logs/div_vs_crit_fi_%j.out
#SBATCH --error=slrum-commands/logs/div_vs_crit_fi_%j.err
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

CKPT_FI="checkpoints/lift_mh_state_flow_mlp_512_h10_seed0_intent_flow_intent_success100.pt"

N_ROLLOUTS=20
N_NOISY=10
K_STEER=10
SUBSAMPLE=2
N_ARM_JOINTS=2   # only wrist-side joints 6+7 (qpos[5:7]) — avoids compounding 7-joint displacement
NOISE_STD=0.02   # ~1 deg per joint; small enough to get recovery variation

echo "============================================================"
echo "Diversity vs Criticality: lift-mh flow-intent"
echo "n_rollouts=${N_ROLLOUTS}  n_noisy=${N_NOISY}  k_steer=${K_STEER}"
echo "n_arm_joints=${N_ARM_JOINTS}  noise_std=${NOISE_STD} rad"
echo "============================================================"

python -u examples/critical_vs_diverse_states.py \
    --run "flow_intent:${CKPT_FI}:task=lift_mh_state_flow_intent:network=mlp_flow_intent" \
    --n-rollouts ${N_ROLLOUTS} \
    --n-noisy ${N_NOISY} \
    --k-steer ${K_STEER} \
    --noise-std ${NOISE_STD} \
    --n-arm-joints ${N_ARM_JOINTS} \
    --subsample-every ${SUBSAMPLE} \
    --device cuda \
    --seed 42 \
    --out rollouts/div_vs_crit_lift_mh_fi_j2_n002.pkl \
    --out-dir rollouts/div_vs_crit_lift_mh_fi_j2_n002_figs

echo ""
echo "============================================================"
echo "Done. Output:"
echo "  rollouts/div_vs_crit_lift_mh_fi_j2_n002.pkl"
echo "  rollouts/div_vs_crit_lift_mh_fi_j2_n002_figs/"
echo "============================================================"
