#!/bin/bash
#SBATCH --job-name=pusht-fi-s2
#SBATCH --output=slrum-commands/logs/pusht_fi_reset_dist_shard2_%j.out
#SBATCH --error=slrum-commands/logs/pusht_fi_reset_dist_shard2_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00

set -euo pipefail

mkdir -p slrum-commands/logs rollouts/bc_reset_dist_pusht_state_fi

cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

export NUMBA_CACHE_DIR=/tmp
export MPLCONFIGDIR=/tmp/matplotlib
export MUJOCO_GL=egl
export XDG_RUNTIME_DIR=/tmp

python examples/analyze_bc_reset_distribution.py \
    --task pusht_state \
    --ckpt checkpoints/pusht_ph_state_flow_mlp_512_seed0_flow_intent_success85.pt \
    --variant flow_intent \
    --n-rollouts 66 \
    --seed-start 134 \
    --seed-end 200 \
    --device cuda \
    --out-dir rollouts/bc_reset_dist_pusht_state_fi

echo "Done. Shard 134-200 saved to: rollouts/bc_reset_dist_pusht_state_fi"
