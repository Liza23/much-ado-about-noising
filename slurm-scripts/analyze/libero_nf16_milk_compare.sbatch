#!/bin/bash
#SBATCH --job-name=libero-nf16-milk
#SBATCH --output=/home/ldahiya/max_vla/much-ado-about-noising/slrum-commands/logs/libero_nf16_milk_%j.out
#SBATCH --error=/home/ldahiya/max_vla/much-ado-about-noising/slrum-commands/logs/libero_nf16_milk_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00

set -euo pipefail

cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

export NUMBA_CACHE_DIR="$(mktemp -d /tmp/numba_${SLURM_JOB_ID:-manual}_XXXXXX)"
export MPLCONFIGDIR="$(mktemp -d /tmp/matplotlib_${SLURM_JOB_ID:-manual}_XXXXXX)"
export MUJOCO_GL=egl
export XDG_RUNTIME_DIR="$(mktemp -d /tmp/xdg_${SLURM_JOB_ID:-manual}_XXXXXX)"
trap 'rm -rf "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_RUNTIME_DIR"' EXIT

OUT_ROOT="rollouts/libero_spatial_image_flow_intent_success68_perturb_grid_nf16_milk_10-rollouts_16-steps"
mkdir -p "$OUT_ROOT"

CKPT="checkpoints/libero_spatial_image_image_flow_mlp_512_seed0_intent_flow_intent_success68.pt"
DATASET="/home/ldahiya/LIBERO/libero/datasets/libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_demo.hdf5"
BASE_BDDL="/home/ldahiya/LIBERO/libero/libero/bddl_files/libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate.bddl"
YELLOW_BDDL="/home/ldahiya/LIBERO-PRO/libero/libero/bddl_files/libero_spatial_with_yellow_book/MAIN_TABLE_SCENE1_pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate.bddl"
MUG_BDDL="/home/ldahiya/LIBERO-PRO/libero/libero/bddl_files/libero_spatial_with_mug/MAIN_TABLE_SCENE1_pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate.bddl"
MILK_BDDL="/home/ldahiya/LIBERO-PRO/libero/libero/bddl_files/libero_spatial_with_milk/FLOOR_SCENE1_pick_the_akita_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate.bddl"
RED_BDDL="/home/ldahiya/LIBERO-PRO/libero/libero/bddl_files/libero_spatial_with_red_box/FLOOR_SCENE1_pick_the_akita_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate.bddl"

run_case() {
  local out_name="$1"
  local bddl_file="$2"
  python examples/collect_intent_outcomes_image.py \
    --run "flow_intent:${CKPT}:task=libero_spatial_image_flow_intent:network=mlp_flow_intent:task.dataset_path=${DATASET}:task.bddl_file=${bddl_file}" \
    --n-rollouts 10 \
    --n-critical 5 \
    --n-probe 20 \
    --n-intents 50 \
    --k-clusters 3 \
    --n-future-steps 16 \
    --render \
    --device cuda \
    --out "${OUT_ROOT}/${out_name}_intent_outcomes.pkl" \
    --out-dir "${OUT_ROOT}/${out_name}_figs"
}

run_case "base" "$BASE_BDDL"
run_case "yellow_book" "$YELLOW_BDDL"
run_case "mug" "$MUG_BDDL"
run_case "milk" "$MILK_BDDL"
run_case "red_box" "$RED_BDDL"

python examples/plot_libero_intent_outcome_grid.py \
  --style v1 \
  --select-by endpoint_spread \
  --row "No Perturb:${OUT_ROOT}/base_intent_outcomes.pkl" \
  --row "Yellow Book:${OUT_ROOT}/yellow_book_intent_outcomes.pkl" \
  --row "Mug:${OUT_ROOT}/mug_intent_outcomes.pkl" \
  --row "Milk:${OUT_ROOT}/milk_intent_outcomes.pkl" \
  --row "Red Block:${OUT_ROOT}/red_box_intent_outcomes.pkl" \
  --out "${OUT_ROOT}/libero_perturb_intent_outcome_grid_nf16_milk.png"
