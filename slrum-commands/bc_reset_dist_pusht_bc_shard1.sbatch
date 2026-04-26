#!/bin/bash
#SBATCH --job-name=pusht-bc-s1
#SBATCH --output=slrum-commands/logs/pusht_bc_reset_dist_shard1_%j.out
#SBATCH --error=slrum-commands/logs/pusht_bc_reset_dist_shard1_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00

set -euo pipefail

mkdir -p slrum-commands/logs rollouts/bc_reset_dist_pusht_state

cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

export NUMBA_CACHE_DIR=/tmp
export MPLCONFIGDIR=/tmp/matplotlib
export MUJOCO_GL=egl
export XDG_RUNTIME_DIR=/tmp

python examples/analyze_bc_reset_distribution.py \
    --task pusht_state \
    --ckpt checkpoints/pusht_ph_state_flow_mlp_512_seed0_success68.pt \
    --variant bc \
    --n-rollouts 67 \
    --seed-start 67 \
    --seed-end 134 \
    --device cuda \
    --out-dir rollouts/bc_reset_dist_pusht_state

echo "Done. Shard 67-134 saved to: rollouts/bc_reset_dist_pusht_state"
