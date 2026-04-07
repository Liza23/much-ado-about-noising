#!/bin/bash
#SBATCH --job-name=libero-soup-fi
#SBATCH --output=logs/libero_soup_state_flow_intent_%j.out
#SBATCH --error=logs/libero_soup_state_flow_intent_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail

mkdir -p logs
source /home/ldahiya/max_vla/much-ado-about-noising/.venv/bin/activate
cd /home/ldahiya/max_vla/much-ado-about-noising

export NUMBA_CACHE_DIR="$(mktemp -d /tmp/numba_${SLURM_JOB_ID:-manual}_XXXXXX)"
export MPLCONFIGDIR="$(mktemp -d /tmp/mpl_${SLURM_JOB_ID:-manual}_XXXXXX)"
export XDG_RUNTIME_DIR="$(mktemp -d /tmp/xdg_${SLURM_JOB_ID:-manual}_XXXXXX)"
export MUJOCO_GL=egl
trap 'rm -rf "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_RUNTIME_DIR"' EXIT

TASK_NAME="LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket"
DATASET_PATH="/home/ldahiya/LIBERO/libero/datasets/libero_10/${TASK_NAME}_demo.hdf5"
BDDL_FILE="/home/ldahiya/LIBERO/libero/libero/bddl_files/libero_10/${TASK_NAME}.bddl"

python examples/train_libero.py \
    task=libero_spatial_state_flow_intent \
    network=mlp_flow_intent \
    task.dataset_path="${DATASET_PATH}" \
    task.bddl_file="${BDDL_FILE}" \
    optimization.gradient_steps=300000 \
    optimization.batch_size=256 \
    optimization.loss_type=flow \
    log.eval_episodes=50 \
    log.eval_freq=20000 \
    log.save_freq=50000 \
    log.project=libero-experiments \
    log.group=libero-10-state-soup \
    log.exp_name=flow_intent \
    task.num_envs=5 \
    optimization.auto_resume=false \
    log.save_video=false
