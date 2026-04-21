#!/bin/bash
#SBATCH --job-name=libero-10-ghost
#SBATCH --output=slrum-commands/logs/collect_libero_10_ghost_%j.out
#SBATCH --error=slrum-commands/logs/collect_libero_10_ghost_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00

set -euo pipefail

mkdir -p slrum-commands/logs rollouts

cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

export NUMBA_CACHE_DIR=/tmp
export MPLCONFIGDIR=/tmp/matplotlib
export MUJOCO_GL=egl
export XDG_RUNTIME_DIR=/tmp

TASK_NAME="KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"
DATASET_PATH="/home/ldahiya/LIBERO/libero/datasets/libero_10/${TASK_NAME}_demo.hdf5"
BDDL_FILE="/home/ldahiya/LIBERO/libero/libero/bddl_files/libero_10/${TASK_NAME}.bddl"

CKPT_FI="/data/user_data/ldahiya/libero-train/libero-10-state/flow_intent/models/model_best.pt"
CKPT_BL="/data/user_data/ldahiya/libero-train/libero-10-state/baseline/models/model_best.pt"

OUT_PKL="rollouts/libero_10_ghost.pkl"
OUT_PLOT="rollouts/libero_10_ghost.png"

SHARED_OVERRIDES="task.dataset_path=${DATASET_PATH}:task.bddl_file=${BDDL_FILE}"

echo "============================================================"
echo "Collecting ghost rollouts: flow_intent vs baseline (libero-10)"
echo "flow_intent ckpt: ${CKPT_FI}"
echo "baseline ckpt:    ${CKPT_BL}"
echo "============================================================"

python examples/collect_ghost_rollouts.py \
    --run "flow_intent:${CKPT_FI}:task=libero_10_state_flow_intent:network=mlp_flow_intent:${SHARED_OVERRIDES}" \
    --run "baseline_stochastic:${CKPT_BL}:task=libero_10_state:${SHARED_OVERRIDES}:optimization.sample_mode=stochastic" \
    --run "baseline_zero:${CKPT_BL}:task=libero_10_state:${SHARED_OVERRIDES}:optimization.sample_mode=zero" \
    --n-rollouts 10 \
    --n-critical 5 \
    --n-probe 20 \
    --k-ghost 8 \
    --n-future-steps 80 \
    --max-timestep 100 \
    --device cuda \
    --out "${OUT_PKL}"

echo "============================================================"
echo "Generating ghost plot: ${OUT_PLOT}"
echo "============================================================"

python examples/plot_ghost.py \
    --in "${OUT_PKL}" \
    --out "${OUT_PLOT}" \
    --n-ghost-plot 8 \
    --k 4 \
    --tag libero_10

echo "Done. Plot saved to: ${OUT_PLOT}"
