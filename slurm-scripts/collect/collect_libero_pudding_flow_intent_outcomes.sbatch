#!/bin/bash
#SBATCH --job-name=libero-pudding-fi-outcomes
#SBATCH --output=slrum-commands/logs/collect_libero_pudding_flow_intent_outcomes_%j.out
#SBATCH --error=slrum-commands/logs/collect_libero_pudding_flow_intent_outcomes_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00

set -euo pipefail

mkdir -p slrum-commands/logs rollouts/libero_pudding_flow_intent_figs

cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

export NUMBA_CACHE_DIR=/tmp
export MPLCONFIGDIR=/tmp/matplotlib
export MUJOCO_GL=egl
export XDG_RUNTIME_DIR=/tmp

CKPT="/data/user_data/ldahiya/libero-train/libero-object-state-pudding/flow_intent/models/model_best.pt"
TASK_NAME="pick_up_the_chocolate_pudding_and_place_it_in_the_basket"
DATASET_PATH="/home/ldahiya/LIBERO/libero/datasets/libero_object/${TASK_NAME}_demo.hdf5"
BDDL_FILE="/home/ldahiya/LIBERO/libero/libero/bddl_files/libero_object/${TASK_NAME}.bddl"

OUT_PKL="rollouts/libero_pudding_flow_intent_outcomes.pkl"
OUT_DIR="rollouts/libero_pudding_flow_intent_figs"
OUT_GRID="rollouts/libero_pudding_flow_intent_grid.png"

echo "============================================================"
echo "Collecting intent-outcomes for libero pudding flow-intent"
echo "Checkpoint: ${CKPT}"
echo "============================================================"

python examples/collect_intent_outcomes_image.py \
    --run "flow_intent:${CKPT}:task=libero_spatial_state_flow_intent:network=mlp_flow_intent:task.dataset_path=${DATASET_PATH}:task.bddl_file=${BDDL_FILE}" \
    --n-rollouts 10 \
    --n-critical 5 \
    --n-intents 50 \
    --k-clusters 3 \
    --n-future-steps 40 \
    --render \
    --device cuda \
    --out "${OUT_PKL}" \
    --out-dir "${OUT_DIR}"

echo "============================================================"
echo "Generating grid figure: ${OUT_GRID}"
echo "============================================================"

python examples/plot_libero_intent_outcome_grid.py \
    --row "flow_intent:${OUT_PKL}" \
    --style v2 \
    --out "${OUT_GRID}"

echo "Done. Figure saved to: ${OUT_GRID}"
