#!/bin/bash
#SBATCH --job-name=pudding-perturb-grid
#SBATCH --output=slrum-commands/logs/collect_libero_pudding_perturb_grid_%j.out
#SBATCH --error=slrum-commands/logs/collect_libero_pudding_perturb_grid_%j.err
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

# ─── Paths ────────────────────────────────────────────────────────────────────
TASK_NAME="pick_up_the_chocolate_pudding_and_place_it_in_the_basket"
DATASET_PATH="/home/ldahiya/LIBERO/libero/datasets/libero_object/${TASK_NAME}_demo.hdf5"

BDDL_NONE="/home/ldahiya/LIBERO/libero/libero/bddl_files/libero_object/${TASK_NAME}.bddl"
BDDL_BOOK="/home/ldahiya/LIBERO-PRO/libero/libero/bddl_files/libero_object_with_yellow_book/FLOOR_SCENE1_pick_up_the_chocolate_pudding_and_place_it_in_the_basket.bddl"
BDDL_MUG="/home/ldahiya/LIBERO-PRO/libero/libero/bddl_files/libero_object_with_mug/FLOOR_SCENE10_pick_the_chocolate_pudding_and_place_it_in_the_basket.bddl"
BDDL_BOX="/home/ldahiya/LIBERO-PRO/libero/libero/bddl_files/libero_object_with_red_box/FLOOR_SCENE1_pick_the_chocolate_pudding_and_place_it_in_the_basket.bddl"

CKPT_FI="/data/user_data/ldahiya/libero-train/libero-object-state-pudding/flow_intent/models/model_best.pt"

BASE_RUN="flow_intent:${CKPT_FI}:task=libero_object_state_flow_intent:network=mlp_flow_intent:task.dataset_path=${DATASET_PATH}"

N_ROLLOUTS=10
N_CRITICAL=5
N_PROBE=20
N_INTENTS=50
K_CLUSTERS=3
N_FUTURE=80
MAX_TS=100

COLLECT="python examples/collect_intent_outcomes_image.py \
    --n-rollouts ${N_ROLLOUTS} \
    --n-critical ${N_CRITICAL} \
    --n-probe ${N_PROBE} \
    --n-intents ${N_INTENTS} \
    --k-clusters ${K_CLUSTERS} \
    --n-future-steps ${N_FUTURE} \
    --max-timestep ${MAX_TS} \
    --render \
    --device cuda"

# ─── Collect: No Perturbation ─────────────────────────────────────────────────
echo "=== [1/4] No perturbation ==="
${COLLECT} \
    --run "${BASE_RUN}:task.bddl_file=${BDDL_NONE}" \
    --out rollouts/pudding_perturb_none.pkl

# ─── Collect: Yellow Book ─────────────────────────────────────────────────────
echo "=== [2/4] Yellow Book ==="
${COLLECT} \
    --run "${BASE_RUN}:task.bddl_file=${BDDL_BOOK}" \
    --out rollouts/pudding_perturb_book.pkl

# ─── Collect: Mug ─────────────────────────────────────────────────────────────
echo "=== [3/4] Mug ==="
${COLLECT} \
    --run "${BASE_RUN}:task.bddl_file=${BDDL_MUG}" \
    --out rollouts/pudding_perturb_mug.pkl

# ─── Collect: Red Box ─────────────────────────────────────────────────────────
echo "=== [4/4] Red Box ==="
${COLLECT} \
    --run "${BASE_RUN}:task.bddl_file=${BDDL_BOX}" \
    --out rollouts/pudding_perturb_box.pkl

# ─── Plot grid ────────────────────────────────────────────────────────────────
echo "=== Plotting perturbation grid ==="
python examples/plot_libero_intent_outcome_grid.py \
    --row "No Perturb:rollouts/pudding_perturb_none.pkl" \
    --row "Yellow Book:rollouts/pudding_perturb_book.pkl" \
    --row "Mug:rollouts/pudding_perturb_mug.pkl" \
    --row "Red Box:rollouts/pudding_perturb_box.pkl" \
    --style v1 \
    --select-by variance \
    --frame-crop full \
    --out rollouts/libero_pudding_perturb_grid.png

echo "Done. Grid saved to: rollouts/libero_pudding_perturb_grid.png"
