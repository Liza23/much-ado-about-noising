#!/bin/bash
#SBATCH --job-name=coverage-lift-mh-analyze
#SBATCH --output=slrum-commands/logs/coverage_lift_mh_analyze_%j.out
#SBATCH --error=slrum-commands/logs/coverage_lift_mh_analyze_%j.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00

set -euo pipefail

mkdir -p slrum-commands/logs figs/coverage/lift_mh

cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

export MPLCONFIGDIR=/tmp/matplotlib

echo "Merging pkls..."
python - <<'EOF'
import pickle
merged = {"lift_mh": {}}
for variant in ("baseline", "flow_intent"):
    path = f"rollouts/coverage_lift_mh_{variant}.pkl"
    with open(path, "rb") as f:
        data = pickle.load(f)
    task_data = data["lift_mh"]
    for k, v in task_data.items():
        if k == "gt_demos" and "gt_demos" in merged["lift_mh"]:
            continue  # keep first gt_demos
        merged["lift_mh"][k] = v
with open("rollouts/coverage_lift_mh_state.pkl", "wb") as f:
    pickle.dump(merged, f)
print("Saved: rollouts/coverage_lift_mh_state.pkl")
print("Keys:", list(merged["lift_mh"].keys()))
EOF

echo "Analyzing coverage..."
python examples/analyze_coverage.py \
    --data rollouts/coverage_lift_mh_state.pkl \
    --task lift_mh \
    --out-dir figs/coverage/lift_mh

echo "Done. Figures: figs/coverage/lift_mh/"
