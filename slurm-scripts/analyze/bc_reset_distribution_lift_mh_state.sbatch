#!/bin/bash
#SBATCH --job-name=bc-reset-dist-lift-mh
#SBATCH --output=slrum-commands/logs/bc_reset_dist_lift_mh_state_%j.out
#SBATCH --error=slrum-commands/logs/bc_reset_dist_lift_mh_state_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00

set -euo pipefail

mkdir -p slrum-commands/logs rollouts/bc_reset_dist_lift_mh_state

cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

export NUMBA_CACHE_DIR=/tmp
export MPLCONFIGDIR=/tmp/matplotlib
export MUJOCO_GL=egl
export XDG_RUNTIME_DIR=/tmp

python examples/analyze_bc_reset_distribution.py \
    --ckpt checkpoints/lift_mh_state_flow_mlp_512_h10_seed0_success100.pt \
    --n-rollouts 200 \
    --device cuda \
    --out-dir rollouts/bc_reset_dist_lift_mh_state

echo "Done. Figures saved to: rollouts/bc_reset_dist_lift_mh_state"
