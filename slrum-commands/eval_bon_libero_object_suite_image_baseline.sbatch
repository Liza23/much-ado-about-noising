#!/bin/bash
#SBATCH --job-name=bon-obj-img-bc
#SBATCH --output=logs/bon_libero_object_suite_image_baseline_%j.out
#SBATCH --error=logs/bon_libero_object_suite_image_baseline_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --requeue
#SBATCH --time=24:00:00

set -euo pipefail

mkdir -p logs
source /home/ldahiya/max_vla/much-ado-about-noising/.venv/bin/activate
cd /home/ldahiya/max_vla/much-ado-about-noising

export NUMBA_CACHE_DIR="$(mktemp -d /tmp/numba_${SLURM_JOB_ID:-manual}_XXXXXX)"
export MPLCONFIGDIR="$(mktemp -d /tmp/mpl_${SLURM_JOB_ID:-manual}_XXXXXX)"
export XDG_RUNTIME_DIR="$(mktemp -d /tmp/xdg_${SLURM_JOB_ID:-manual}_XXXXXX)"
export MUJOCO_GL=egl
trap 'rm -rf "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_RUNTIME_DIR"' EXIT

BC_CKPT="/data/user_data/ldahiya/libero-train/libero-object-suite-image/baseline/models/model_latest.pt"
OUTPUT_DIR="/data/user_data/ldahiya/libero-eval/bon/libero-object-suite-image"

mkdir -p "${OUTPUT_DIR}"

python examples/eval_best_of_n.py \
    task=libero_object_suite_image \
    network=mlp \
    eval.checkpoint="${BC_CKPT}" \
    eval.num_rollouts=50 \
    eval.num_envs=5 \
    eval.seed=0 \
    eval.n_values="[1,10,50,100,500,1000,2000]" \
    eval.output_path="${OUTPUT_DIR}/bc_baseline_bon.json"
