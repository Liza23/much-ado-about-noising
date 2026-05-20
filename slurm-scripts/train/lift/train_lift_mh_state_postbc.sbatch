#!/bin/bash
#SBATCH --job-name=lift-mh-postbc
#SBATCH --output=logs/lift_mh_state_postbc_%j.out
#SBATCH --error=logs/lift_mh_state_postbc_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=36:00:00   # Phase 1 (ensemble ~2-4h) + Phase 2 (flow BC ~12h)

set -euo pipefail

mkdir -p logs

cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

export NUMBA_CACHE_DIR=/tmp
export MPLCONFIGDIR=/tmp/matplotlib
export MUJOCO_GL=egl
export XDG_RUNTIME_DIR=/tmp

python examples/train_postbc.py \
    task=lift_mh_state_postbc \
    network=mlp \
    postbc.ensemble_size=100 \
    postbc.ensemble_epochs=50 \
    postbc.alpha=1.0 \
    task.num_envs=5 \
    log.eval_episodes=50 \
    log.eval_freq=10000 \
    log.save_freq=50000 \
    log.project=lift-experiments \
    log.group=lift-mh-state \
    log.exp_name=postbc_flow_bc
