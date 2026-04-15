#!/bin/bash
#SBATCH --job-name=eval-bl-warmup
#SBATCH --output=logs/eval_libero_after_warmup_%j.out
#SBATCH --error=logs/eval_libero_after_warmup_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00

set -euo pipefail

mkdir -p logs
source /home/ldahiya/max_vla/much-ado-about-noising/.venv/bin/activate
cd /home/ldahiya/max_vla/much-ado-about-noising

export NUMBA_CACHE_DIR="$(mktemp -d /tmp/numba_${SLURM_JOB_ID:-manual}_XXXXXX)"
export MPLCONFIGDIR="$(mktemp -d /tmp/mpl_${SLURM_JOB_ID:-manual}_XXXXXX)"
export XDG_RUNTIME_DIR="$(mktemp -d /tmp/xdg_${SLURM_JOB_ID:-manual}_XXXXXX)"
export MUJOCO_GL=egl
trap 'rm -rf "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_RUNTIME_DIR"' EXIT

# --- Paths (must match log.log_dir in training sbatch files) ---
SPATIAL_TASK="pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate"
SPATIAL_DS="/home/ldahiya/LIBERO/libero/datasets/libero_spatial/${SPATIAL_TASK}_demo.hdf5"
SPATIAL_BDDL="/home/ldahiya/LIBERO/libero/libero/bddl_files/libero_spatial/${SPATIAL_TASK}.bddl"
SPATIAL_CKPT_BEST="/data/user_data/ldahiya/libero-train/libero-spatial-state/baseline/models/model_best.pt"
SPATIAL_CKPT_LATEST="/data/user_data/ldahiya/libero-train/libero-spatial-state/baseline/models/model_latest.pt"

PUDDING_TASK="pick_up_the_chocolate_pudding_and_place_it_in_the_basket"
PUDDING_DS="/home/ldahiya/LIBERO/libero/datasets/libero_object/${PUDDING_TASK}_demo.hdf5"
PUDDING_BDDL="/home/ldahiya/LIBERO/libero/libero/bddl_files/libero_object/${PUDDING_TASK}.bddl"
PUDDING_CKPT_BEST="/data/user_data/ldahiya/libero-train/libero-object-state-pudding/baseline/models/model_best.pt"
PUDDING_CKPT_LATEST="/data/user_data/ldahiya/libero-train/libero-object-state-pudding/baseline/models/model_latest.pt"

SOUP_TASK="LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket"
SOUP_DS="/home/ldahiya/LIBERO/libero/datasets/libero_10/${SOUP_TASK}_demo.hdf5"
SOUP_BDDL="/home/ldahiya/LIBERO/libero/libero/bddl_files/libero_10/${SOUP_TASK}.bddl"
SOUP_CKPT_BEST="/data/user_data/ldahiya/libero-train/libero-10-state-soup/baseline_600k/models/model_best.pt"
SOUP_CKPT_LATEST="/data/user_data/ldahiya/libero-train/libero-10-state-soup/baseline_600k/models/model_latest.pt"

# Evaluate model_latest (300k steps) with the gripper warm-up fix applied.
# Also evaluates spatial model_best (76% peak) to confirm it reproduces.
python examples/eval_success_rate.py \
    --run "spatial_best:${SPATIAL_CKPT_BEST}:task=libero_spatial_state:task.dataset_path=${SPATIAL_DS}:task.bddl_file=${SPATIAL_BDDL}" \
    --run "spatial_latest:${SPATIAL_CKPT_LATEST}:task=libero_spatial_state:task.dataset_path=${SPATIAL_DS}:task.bddl_file=${SPATIAL_BDDL}" \
    --run "pudding_best:${PUDDING_CKPT_BEST}:task=libero_spatial_state:task.dataset_path=${PUDDING_DS}:task.bddl_file=${PUDDING_BDDL}" \
    --run "pudding_latest:${PUDDING_CKPT_LATEST}:task=libero_spatial_state:task.dataset_path=${PUDDING_DS}:task.bddl_file=${PUDDING_BDDL}" \
    --run "soup_best:${SOUP_CKPT_BEST}:task=libero_spatial_state:task.dataset_path=${SOUP_DS}:task.bddl_file=${SOUP_BDDL}" \
    --run "soup_latest:${SOUP_CKPT_LATEST}:task=libero_spatial_state:task.dataset_path=${SOUP_DS}:task.bddl_file=${SOUP_BDDL}" \
    --num-steps 1 \
    --n-rollouts 50 \
    --out results/libero_baselines_eval_after_warmup.csv
