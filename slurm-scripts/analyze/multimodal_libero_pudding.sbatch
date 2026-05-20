#!/bin/bash
#SBATCH --job-name=mm-pudding
#SBATCH --output=slrum-commands/logs/multimodal_libero_pudding_%j.out
#SBATCH --error=slrum-commands/logs/multimodal_libero_pudding_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00

set -euo pipefail

mkdir -p slrum-commands/logs rollouts

cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

export NUMBA_CACHE_DIR=/tmp
export MPLCONFIGDIR=/tmp/matplotlib
export MUJOCO_GL=egl
export XDG_RUNTIME_DIR=/tmp

TASK_NAME="pick_up_the_chocolate_pudding_and_place_it_in_the_basket"
DATASET_PATH="/home/ldahiya/LIBERO/libero/datasets/libero_object/${TASK_NAME}_demo.hdf5"
BDDL_FILE="/home/ldahiya/LIBERO/libero/libero/bddl_files/libero_object/${TASK_NAME}.bddl"

CKPT_BL="/data/user_data/ldahiya/libero-train/libero-object-state-pudding/baseline/models/model_best.pt"
CKPT_FI="/data/user_data/ldahiya/libero-train/libero-object-state-pudding/flow_intent/models/model_best.pt"
CKPT_PARL="/data/user_data/ldahiya/libero-train/libero-object-state-pudding/residual_parl/residual_parl_checkpoints/best.pt"

SHARED="task.dataset_path=${DATASET_PATH}:task.bddl_file=${BDDL_FILE}"

python examples/collect_multimodal_viz.py \
    --task libero_pudding \
    --run "flow_intent:${CKPT_FI}:task=libero_spatial_state_flow_intent:network=mlp_flow_intent:${SHARED}" \
    --run "baseline:${CKPT_BL}:task=libero_spatial_state:${SHARED}" \
    --n-rollouts 20 --n-samples 50 --n-critical 10 \
    --device cuda \
    --out rollouts/libero_pudding_multimodal.pkl

# Append residual_parl to existing pkl (reuses stored critical states, no env re-discovery)
python examples/collect_multimodal_viz.py \
    --task libero_pudding \
    --run "residual_parl:${CKPT_PARL}:task=libero_object_state_flow_intent:network=mlp_flow_intent:flow_intent_ckpt=${CKPT_FI}:${SHARED}" \
    --n-rollouts 20 --n-samples 50 \
    --device cuda \
    --out rollouts/libero_pudding_multimodal.pkl \
    --append

python examples/analyze_multimodal_clustering.py \
    --data rollouts/libero_pudding_multimodal.pkl \
    --task libero_pudding \
    --out-dir rollouts/libero_pudding_cluster_figs \
    --k-min 2 --k-max 6 --n-ghost-states 5
