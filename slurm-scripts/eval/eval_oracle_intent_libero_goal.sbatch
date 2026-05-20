#!/bin/bash
#SBATCH --job-name=oracle-intent-goal
#SBATCH --output=logs/eval_oracle_intent_libero_goal_%j.out
#SBATCH --error=logs/eval_oracle_intent_libero_goal_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00

set -euo pipefail

mkdir -p logs
source /home/ldahiya/max_vla/much-ado-about-noising/.venv/bin/activate
cd /home/ldahiya/max_vla/much-ado-about-noising

export NUMBA_CACHE_DIR="$(mktemp -d /tmp/numba_${SLURM_JOB_ID:-manual}_XXXXXX)"
export MPLCONFIGDIR="$(mktemp -d /tmp/mpl_${SLURM_JOB_ID:-manual}_XXXXXX)"
export XDG_RUNTIME_DIR="$(mktemp -d /tmp/xdg_${SLURM_JOB_ID:-manual}_XXXXXX)"
export MUJOCO_GL=egl
trap 'rm -rf "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_RUNTIME_DIR"' EXIT

# Run oracle intent evaluation on the libero-goal-suite-image flow_intent_curriculum checkpoint.
# Compares:
#   (1) action MSE with GT future eef intent (oracle upper bound)
#   (2) action MSE with ODE-sampled intent   (what eval actually sees)
# Large ratio => intent predictor is the bottleneck.

python examples/eval_oracle_intent.py \
    task=libero_goal_suite_image_flow_intent \
    network=mlp_flow_intent \
    optimization.loss_type=flow \
    optimization.batch_size=64 \
    optimization.model_path="/data/user_data/ldahiya/libero-train/libero-goal-suite-image/flow_intent_curriculum/models/model_best.pt" \
    ++eval_oracle.num_batches=200 \
    ++eval_oracle.batch_size=64 \
    ++eval_oracle.num_steps=9 \
    log.wandb_mode=disabled \
    log.log_dir="/tmp/oracle_eval_goal"
