#!/bin/bash
#SBATCH --job-name=intent-ablation-v4
#SBATCH --output=logs/intent_ablation_v4_%j.out
#SBATCH --error=logs/intent_ablation_v4_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00

mkdir -p logs

cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

export MUJOCO_GL=egl
export NUMBA_CACHE_DIR=/tmp
export MPLCONFIGDIR=/tmp/matplotlib
export XDG_RUNTIME_DIR=/tmp

python examples/eval_intent_ablation.py \
    task=lift_mh_image_slot_intent \
    network=mlp_flow_intent \
    optimization.loss_type=flow \
    optimization.model_path=/data/user_data/ldahiya/mip_render/lift-mh-image/slot_intent_seed0_v4/models/model_best.pt \
    log.eval_episodes=50 \
    task.num_envs=4
