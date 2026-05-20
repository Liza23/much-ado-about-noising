#!/bin/bash
#SBATCH --job-name=libero-goal-all-st
#SBATCH --output=logs/libero_goal_all_single_tasks_%j.out
#SBATCH --error=logs/libero_goal_all_single_tasks_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --requeue
#SBATCH --time=48:00:00

# Sequential single-task baselines for all 8 remaining LIBERO-Goal tasks.
# Timing: ~8h/task x 8 tasks = ~64h total — exceeds 48h wall limit.
# ~5-6 tasks will complete per submission. Comment out finished tasks and resubmit for the rest.
#
# Already running separately (do NOT include here):
#   task0: put_the_bowl_on_the_plate         (job 7557585)
#   task7: put_the_wine_bottle_on_the_rack   (job 7562724)

set -euo pipefail

mkdir -p logs
source /home/ldahiya/max_vla/much-ado-about-noising/.venv/bin/activate
cd /home/ldahiya/max_vla/much-ado-about-noising

export NUMBA_CACHE_DIR="$(mktemp -d /tmp/numba_${SLURM_JOB_ID:-manual}_XXXXXX)"
export MPLCONFIGDIR="$(mktemp -d /tmp/mpl_${SLURM_JOB_ID:-manual}_XXXXXX)"
export XDG_RUNTIME_DIR="$(mktemp -d /tmp/xdg_${SLURM_JOB_ID:-manual}_XXXXXX)"
export MUJOCO_GL=egl
trap 'rm -rf "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_RUNTIME_DIR"' EXIT

COMMON_ARGS="
    network.rgb_model_name=resnet50
    optimization.gradient_steps=100000
    optimization.batch_size=128
    optimization.loss_type=flow
    optimization.weight_decay=1e-4
    optimization.auto_resume=false
    log.eval_episodes=20
    log.eval_freq=20000
    log.eval_nsteps=9
    log.save_freq=100000
    log.project=libero-experiments
    log.group=libero-goal-single-task
    log.save_video=false
    task.num_envs=5
"

# task1: open_the_middle_drawer_of_the_cabinet
echo "=== [$(date)] Starting task1: open_middle_drawer ==="
python examples/train_libero.py \
    task=libero_goal_open_middle_drawer_image \
    log.exp_name=open_middle_drawer_resnet50_bs128 \
    log.log_dir="/data/user_data/ldahiya/libero-train/libero-goal-single-task/open_middle_drawer_resnet50_bs128" \
    $COMMON_ARGS

# task2: open_the_top_drawer_and_put_the_bowl_inside
echo "=== [$(date)] Starting task2: open_top_drawer_bowl ==="
python examples/train_libero.py \
    task=libero_goal_open_top_drawer_bowl_image \
    log.exp_name=open_top_drawer_bowl_resnet50_bs128 \
    log.log_dir="/data/user_data/ldahiya/libero-train/libero-goal-single-task/open_top_drawer_bowl_resnet50_bs128" \
    $COMMON_ARGS

# task3: push_the_plate_to_the_front_of_the_stove
echo "=== [$(date)] Starting task3: push_plate_stove ==="
python examples/train_libero.py \
    task=libero_goal_push_plate_stove_image \
    log.exp_name=push_plate_stove_resnet50_bs128 \
    log.log_dir="/data/user_data/ldahiya/libero-train/libero-goal-single-task/push_plate_stove_resnet50_bs128" \
    $COMMON_ARGS

# task4: put_the_bowl_on_the_stove
echo "=== [$(date)] Starting task4: bowl_on_stove ==="
python examples/train_libero.py \
    task=libero_goal_bowl_on_stove_image \
    log.exp_name=bowl_on_stove_resnet50_bs128 \
    log.log_dir="/data/user_data/ldahiya/libero-train/libero-goal-single-task/bowl_on_stove_resnet50_bs128" \
    $COMMON_ARGS

# task5: put_the_bowl_on_top_of_the_cabinet
echo "=== [$(date)] Starting task5: bowl_on_cabinet ==="
python examples/train_libero.py \
    task=libero_goal_bowl_on_cabinet_image \
    log.exp_name=bowl_on_cabinet_resnet50_bs128 \
    log.log_dir="/data/user_data/ldahiya/libero-train/libero-goal-single-task/bowl_on_cabinet_resnet50_bs128" \
    $COMMON_ARGS

# task6: put_the_cream_cheese_in_the_bowl
echo "=== [$(date)] Starting task6: cream_cheese_bowl ==="
python examples/train_libero.py \
    task=libero_goal_cream_cheese_bowl_image \
    log.exp_name=cream_cheese_bowl_resnet50_bs128 \
    log.log_dir="/data/user_data/ldahiya/libero-train/libero-goal-single-task/cream_cheese_bowl_resnet50_bs128" \
    $COMMON_ARGS

# task8: put_the_wine_bottle_on_top_of_the_cabinet
echo "=== [$(date)] Starting task8: wine_bottle_on_cabinet ==="
python examples/train_libero.py \
    task=libero_goal_wine_bottle_on_cabinet_image \
    log.exp_name=wine_bottle_on_cabinet_resnet50_bs128 \
    log.log_dir="/data/user_data/ldahiya/libero-train/libero-goal-single-task/wine_bottle_on_cabinet_resnet50_bs128" \
    $COMMON_ARGS

# task9: turn_on_the_stove
echo "=== [$(date)] Starting task9: turn_on_stove ==="
python examples/train_libero.py \
    task=libero_goal_turn_on_stove_image \
    log.exp_name=turn_on_stove_resnet50_bs128 \
    log.log_dir="/data/user_data/ldahiya/libero-train/libero-goal-single-task/turn_on_stove_resnet50_bs128" \
    $COMMON_ARGS

echo "=== [$(date)] All tasks complete ==="
