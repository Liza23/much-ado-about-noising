#!/bin/bash
#SBATCH --job-name=libero-goal-eval
#SBATCH --output=logs/eval_libero_goal_offline_%j.out
#SBATCH --error=logs/eval_libero_goal_offline_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00

set -euo pipefail

mkdir -p logs
source /home/ldahiya/max_vla/much-ado-about-noising/.venv/bin/activate
cd /home/ldahiya/max_vla/much-ado-about-noising

export NUMBA_CACHE_DIR="$(mktemp -d /tmp/numba_${SLURM_JOB_ID:-manual}_XXXXXX)"
export MPLCONFIGDIR="$(mktemp -d /tmp/mpl_${SLURM_JOB_ID:-manual}_XXXXXX)"
export XDG_RUNTIME_DIR="$(mktemp -d /tmp/xdg_${SLURM_JOB_ID:-manual}_XXXXXX)"
export MUJOCO_GL=egl
trap 'rm -rf "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_RUNTIME_DIR"' EXIT

echo "========================================"
echo "BASELINE (resnet50, bs64, 1M steps)"
echo "========================================"
python examples/eval_libero_goal_offline.py \
    task=libero_goal_suite_image \
    network.rgb_model_name=resnet50 \
    optimization.loss_type=flow \
    optimization.model_path="/data/user_data/ldahiya/libero-train/libero-goal-suite-image/resnet50/models/model_best.pt" \
    log.eval_episodes=50 \
    task.num_envs=5 \
    log.eval_nsteps=9

echo ""
echo "========================================"
echo "FLOW_INTENT (resnet50, bs64, 1M steps)"
echo "========================================"
python examples/eval_libero_goal_offline.py \
    task=libero_goal_suite_image_flow_intent \
    network=mlp_flow_intent \
    network.rgb_model_name=resnet50 \
    optimization.loss_type=flow \
    optimization.model_path="/data/user_data/ldahiya/libero-train/libero-goal-suite-image/flow_intent_resnet50/models/model_best.pt" \
    log.eval_episodes=50 \
    task.num_envs=5 \
    log.eval_nsteps=9
