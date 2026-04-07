#!/bin/bash
#SBATCH --job-name=libero-img-base
#SBATCH --output=logs/libero_image_baseline_%j.out
#SBATCH --error=logs/libero_image_baseline_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00

mkdir -p logs
source /home/ldahiya/max_vla/much-ado-about-noising/.venv/bin/activate
cd /home/ldahiya/max_vla/much-ado-about-noising

TASK_NAME="pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate"
DATASET_PATH="/home/ldahiya/LIBERO/libero/datasets/libero_spatial/${TASK_NAME}_demo.hdf5"
BDDL_FILE="/home/ldahiya/LIBERO/libero/libero/bddl_files/libero_spatial/${TASK_NAME}.bddl"

NUMBA_CACHE_DIR=/tmp MUJOCO_GL=egl python examples/train_libero.py \
    task=libero_spatial_image \
    task.dataset_path="${DATASET_PATH}" \
    task.bddl_file="${BDDL_FILE}" \
    optimization.gradient_steps=300000 \
    optimization.batch_size=64 \
    optimization.loss_type=flow \
    log.eval_episodes=50 \
    log.eval_freq=20000 \
    log.save_freq=50000 \
    log.project=libero-experiments \
    log.group=libero-spatial-image \
    log.exp_name=baseline \
    task.num_envs=5 \
    optimization.auto_resume=false \
    log.save_video=false
