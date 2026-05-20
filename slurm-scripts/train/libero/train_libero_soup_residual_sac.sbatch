#!/bin/bash
#SBATCH --job-name=libero-soup-rsac
#SBATCH --output=logs/libero_soup_residual_sac_%j.out
#SBATCH --error=logs/libero_soup_residual_sac_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
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

TASK_NAME="LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket"
DATASET_PATH="/home/ldahiya/LIBERO/libero/datasets/libero_10/${TASK_NAME}_demo.hdf5"
BDDL_FILE="/home/ldahiya/LIBERO/libero/libero/bddl_files/libero_10/${TASK_NAME}.bddl"
FLOW_INTENT_CKPT="/data/user_data/ldahiya/libero-train/libero-10-state-soup/flow_intent/models/model_best.pt"

python examples/train_residual_sac.py \
    task=libero_10_state_flow_intent \
    network=mlp_flow_intent \
    task.dataset_path="${DATASET_PATH}" \
    task.bddl_file="${BDDL_FILE}" \
    residual_sac.flow_intent_ckpt="${FLOW_INTENT_CKPT}" \
    residual_sac.residual_alpha=0.1 \
    residual_sac.actor_lr=3e-4 \
    residual_sac.critic_lr=3e-4 \
    residual_sac.discount=0.99 \
    residual_sac.batch_size=256 \
    residual_sac.warmup_steps=5000 \
    residual_sac.total_env_steps=1000000 \
    residual_sac.num_updates_per_step=1 \
    log.eval_episodes=50 \
    log.eval_freq=10000 \
    log.save_freq=50000 \
    log.project=libero-experiments \
    log.group=libero-10-state-soup \
    log.exp_name=residual_sac \
    log.log_dir="/data/user_data/ldahiya/libero-train/libero-10-state-soup/residual_sac" \
    task.num_envs=5 \
    log.save_video=false
