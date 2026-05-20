#!/bin/bash
#SBATCH --job-name=critical-states-rl
#SBATCH --output=slrum-commands/logs/critical_states_rl_%j.out
#SBATCH --error=slrum-commands/logs/critical_states_rl_%j.err
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
CKPT_SAC="/data/user_data/ldahiya/mip_render/lift-mh-state/intent_flow_residual_sac/residual_sac_checkpoints/best.pt"

N_ROLLOUTS=20
N_NOISY=10
SUBSAMPLE=2

echo "============================================================"
echo "Critical-state identification: lift-mh ResidualSAC"
echo "n_rollouts=${N_ROLLOUTS}  n_noisy=${N_NOISY}"
echo "============================================================"

echo ""
echo ">>> ResidualSAC  noise=0.05 rad"
python examples/identify_critical_states.py \
    --run "residual_sac:${CKPT_SAC}:task=lift_mh_state_flow_intent:network=mlp_flow_intent" \
    --agent-type residual_sac \
    --flow-intent-ckpt ${CKPT_FI} \
    --n-rollouts ${N_ROLLOUTS} \
    --n-noisy ${N_NOISY} \
    --noise-std 0.05 \
    --subsample-every ${SUBSAMPLE} \
    --device cuda \
    --seed 42 \
    --out rollouts/critical_states_lift_mh_sac_n005.pkl \
    --out-dir rollouts/critical_states_lift_mh_sac_n005_figs

echo ""
echo ">>> ResidualSAC  noise=0.10 rad"
python examples/identify_critical_states.py \
    --run "residual_sac:${CKPT_SAC}:task=lift_mh_state_flow_intent:network=mlp_flow_intent" \
    --agent-type residual_sac \
    --flow-intent-ckpt ${CKPT_FI} \
    --n-rollouts ${N_ROLLOUTS} \
    --n-noisy ${N_NOISY} \
    --noise-std 0.10 \
    --subsample-every ${SUBSAMPLE} \
    --device cuda \
    --seed 42 \
    --out rollouts/critical_states_lift_mh_sac_n010.pkl \
    --out-dir rollouts/critical_states_lift_mh_sac_n010_figs

echo ""
echo "============================================================"
echo "Done. Outputs:"
echo "  rollouts/critical_states_lift_mh_sac_{n005,n010}.pkl"
echo "  rollouts/critical_states_lift_mh_sac_*_figs/"
echo "============================================================"
