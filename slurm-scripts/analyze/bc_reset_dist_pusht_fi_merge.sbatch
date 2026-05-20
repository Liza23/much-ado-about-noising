#!/bin/bash
#SBATCH --job-name=pusht-fi-merge
#SBATCH --output=slrum-commands/logs/pusht_fi_reset_dist_merge_%j.out
#SBATCH --error=slrum-commands/logs/pusht_fi_reset_dist_merge_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00

set -euo pipefail

mkdir -p slrum-commands/logs

cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

export MPLCONFIGDIR=/tmp/matplotlib

python examples/analyze_bc_reset_distribution.py \
    --task pusht_state \
    --variant flow_intent \
    --merge-only \
    --out-dir rollouts/bc_reset_dist_pusht_state_fi

echo "Done. Figures saved to: rollouts/bc_reset_dist_pusht_state_fi"
