#!/bin/bash
#SBATCH --job-name=lift-img-fi
#SBATCH --output=slrum-commands/logs/lift_image_flow_intent_%j.out
#SBATCH --error=slrum-commands/logs/lift_image_flow_intent_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00

mkdir -p slrum-commands/logs rollouts

cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

# Collect only flow_intent into a temp pkl
python examples/collect_diversity_rollouts.py \
    --task lift_ph_image \
    --run "flow_intent:checkpoints/lift_ph_image_flow_mlp_512_h16_seed0_intent_flow_intent_success100.pt:task=lift_ph_image_flow_intent:+network.arch_variant=flow_intent" \
    --n-rollouts 100 --device cuda \
    --out rollouts/lift_ph_image_flow_intent_only.pkl

# Merge into existing pkl
python - <<'EOF'
import pickle

with open('rollouts/lift_ph_image_diversity.pkl', 'rb') as f:
    existing = pickle.load(f)

with open('rollouts/lift_ph_image_flow_intent_only.pkl', 'rb') as f:
    new = pickle.load(f)

# Merge flow_intent entry into existing pkl
existing['lift_ph_image'].update(new.get('lift_ph_image', {}))

with open('rollouts/lift_ph_image_diversity.pkl', 'wb') as f:
    pickle.dump(existing, f)

print("Merged variants:", list(existing['lift_ph_image'].keys()))
EOF

# Re-run analysis
python examples/analyze_diversity.py \
    --data rollouts/lift_ph_image_diversity.pkl \
    --task lift_ph_image \
    --out-dir rollouts/lift_ph_image_figs
