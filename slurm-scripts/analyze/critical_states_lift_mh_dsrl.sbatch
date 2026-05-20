#!/bin/bash
#SBATCH --job-name=critical-states-dsrl
#SBATCH --output=slrum-commands/logs/critical_states_dsrl_%j.out
#SBATCH --error=slrum-commands/logs/critical_states_dsrl_%j.err
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

CKPT_FI="checkpoints/lift_mh_state_flow_mlp_512_h10_seed0_intent_flow_intent_success100.pt"
CKPT_DSRL="/data/user_data/ldahiya/mip_render/lift-mh-state/flow_intent_dsrl/dsrl_checkpoints/best.pt"

N_ROLLOUTS=20
N_NOISY=10
SUBSAMPLE=2

echo "============================================================"
echo "Critical-state identification: lift-mh DSRL (flow-intent)"
echo "n_rollouts=${N_ROLLOUTS}  n_noisy=${N_NOISY}"
echo "============================================================"

echo ""
echo ">>> DSRL  noise=0.05 rad"
python examples/identify_critical_states.py \
    --run "dsrl:${CKPT_DSRL}:task=lift_mh_state_flow_intent:network=mlp_flow_intent" \
    --agent-type dsrl \
    --flow-intent-ckpt ${CKPT_FI} \
    --n-rollouts ${N_ROLLOUTS} \
    --n-noisy ${N_NOISY} \
    --noise-std 0.05 \
    --subsample-every ${SUBSAMPLE} \
    --device cuda \
    --seed 42 \
    --out rollouts/critical_states_lift_mh_dsrl_n005.pkl \
    --out-dir rollouts/critical_states_lift_mh_dsrl_n005_figs

echo ""
echo "============================================================"
echo "Done. Output:"
echo "  rollouts/critical_states_lift_mh_dsrl_n005.pkl"
echo "  rollouts/critical_states_lift_mh_dsrl_n005_figs/"
echo "============================================================"
