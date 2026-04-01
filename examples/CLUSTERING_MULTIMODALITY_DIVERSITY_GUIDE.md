# Clustering Multimodality and Diversity Scores: Detailed Guide

This document explains, in detail, how the multimodality and diversity analysis pipeline works in this repository, with a focus on:

1. Ghost plots  
2. VDR violin plot  
3. VDR / silhouette CSV outputs  

It also clarifies how these differ from the "diversity metrics" in `analyze_diversity.py`.

---

## 1) End-to-End Pipeline at a Glance

There are two main stages and two analysis families:

- **Data collection stage**
  - Script: `examples/collect_multimodal_viz.py`
  - Purpose: collect rollout chunks, discover high-variance critical states, and sample many actions from each variant at those frozen states.
  - Output: one `.pkl` file.

- **Clustering multimodality stage**
  - Script: `examples/analyze_multimodal_clustering.py`
  - Purpose: cluster per-state action samples (spectral clustering), compute VDR/silhouette/CH, and generate ghost + comparison plots.
  - Outputs: `ghost` plots, `vdr_violin.png`, `clustering_results.csv`, and summary artifacts.

- **Distribution diversity stage (separate lens)**
  - Script: `examples/analyze_diversity.py`
  - Purpose: quantify action spread/coverage using simple geometric statistics (std, pairwise L2, coverage).
  - Outputs: `diversity_metrics.png` + `diversity_metrics.txt`.

Important: **VDR/silhouette and diversity metrics are not the same quantity**. They answer different questions.

---

## 2) Data Collection (`collect_multimodal_viz.py`)

## 2.1 Core idea

For each policy variant:

1. Run normal rollouts to collect baseline behavior data.
2. Discover "critical" states where repeated action sampling has high variance.
3. Freeze those states and sample many actions (`n_samples`) from each variant at each fixed state.

This enables controlled multimodality comparisons: same state, many action samples, across variants.

## 2.2 Critical states

A state is scored by a probe variance:

- At timestep `t`, sample `n_probe` action chunks from the same observation.
- Flatten each sampled chunk.
- Compute mean variance across dimensions.
- Store state metadata (`obs_raw`, `timestep`, `episode_idx`, `variance_score`).
- Sort states by `variance_score` descending.

Interpretation:

- Higher probe variance means the policy is more stochastic/diverse at that state.
- These states are likely where multimodality is easiest to observe.

## 2.3 Output `.pkl` shape (high-level)

For each task:

- `critical_states` contains selected frozen states and metadata.
- each variant contains:
  - rollout data (`rollout_chunks`, success list)
  - critical-state samples (`critical_samples`: shape `(n_critical, n_samples, act_steps, act_dim)`)
  - optional intent data for flow-intent variants.

---

## 3) Clustering Multimodality (`analyze_multimodal_clustering.py`)

This is the script that creates:

- ghost plot artifacts
- VDR violin figure
- VDR/silhouette CSV

It processes each `variant x critical_state` independently, then aggregates.

## 3.1 Per-state clustering procedure

Given sampled actions at one state:

- Input shape: `(N, T, A)` where:
  - `N`: number of samples at this frozen state
  - `T`: action horizon (`act_steps`)
  - `A`: action dim

Steps:

1. **Convert action chunks to trajectories**
   - Uses shared helper integration from `viz/clustering_shared.py`.
   - If `--abs-action` (default true in this script), first `vel_dims` are treated as absolute positions and anchored to first timestep.
   - If false, it integrates velocities with cumulative sum (`dt` scaling).

2. **Build pairwise trajectory distance matrix `D`**
   - Uses a symmetrized nearest-point Minkowski-like trajectory distance.
   - Distance is computed between every pair of sampled trajectories.

3. **Convert distance to affinity**
   - `A_aff = exp(-D^2 / (2*sigma^2))` where `sigma` is median positive distance.
   - Diagonal forced to 1.

4. **Run spectral clustering for each `k` in `[k_min, k_max]`**
   - Clusters based on precomputed affinity.
   - Returns labels for each sampled chunk.

5. **Compute metrics per `k`**
   - `total_variance` from full `D`
   - `weighted_incluster_variance` from cluster-restricted submatrices
   - `variance_drop = total_variance - weighted_incluster_variance`
   - `VDR = variance_drop / total_variance` (clipped `[0,1]`)
   - `CH` (Calinski-Harabasz style quantity)
   - `silhouette` on flattened trajectory features (`X_feat`, Euclidean), when valid.

6. **Choose best `k`**
   - Best cluster solution is selected by **max VDR**.

## 3.2 What VDR is measuring

Intuition:

- `total_variance` = how spread apart all sampled trajectories are before clustering.
- `weighted_incluster_variance` = residual spread after assigning points to clusters.
- `variance_drop` = spread explained by splitting into clusters.

So:

- `VDR ≈ 0`: clusters do not explain much structure; samples are effectively unimodal or not cleanly separated.
- `VDR → 1`: clustering explains most variance; strong multimodal structure.

Note:

- VDR depends on both genuine multi-branch behavior and how cleanly separated the branches are in the chosen trajectory metric.

## 3.3 Silhouette in this script

Silhouette here is:

- computed on `X_feat` (flattened trajectory vectors)
- Euclidean metric
- only attempted when `N >= 10` and at least 2 clusters

Meaning:

- higher silhouette suggests better separated, tighter clusters.
- low/negative silhouette suggests overlapping clusters or poor partition quality.

Why both VDR and silhouette:

- VDR measures variance explained by clustering in the custom distance geometry.
- silhouette measures compactness/separation in feature space.
- Together they provide a more robust view than either alone.

---

## 4) Artifact 1: Ghost Plots

Files:

- `ghost_plots/ghost_comparison_stateXX.png`
- optional `ghost_plots/ghost_comparison_stateXX.html` (if Plotly available)

What is shown:

- One selected critical state at a time.
- For each variant, sampled trajectories are drawn as semi-transparent 3D lines.
- Colors correspond to cluster labels (best-k clustering for that variant/state).
- All variants for that state are projected through one global PCA basis, so geometry is comparable across panels.

Why it is useful:

- Qualitative view of whether branches are truly distinct.
- Checks whether high VDR corresponds to visibly separate trajectory families.
- Helps spot artifacts (for example, one cluster being just noise tails).

How to read quickly:

- Distinct bundles in one panel -> strong multimodality for that variant/state.
- Single thick cloud with little structure -> weak multimodality.
- Different variants with same state can reveal conditioning effects (intent models often show different branch spread/shape).

---

## 5) Artifact 2: VDR Violin Plot

File:

- `vdr_violin.png`

What it aggregates:

- Per variant, collect best-state VDR values across all critical states.
- Plot one violin per variant + jittered state points + median/extrema.

Interpretation:

- **Higher center/median**: stronger average multimodality.
- **Wider violin**: multimodality varies a lot by state.
- **Narrow + low**: mostly weak or consistent low multimodality.
- **Narrow + high**: consistently strong multimodality across states.

Good practice:

- Compare this with `vdr_comparison.png` (statewise bars) to identify if gains are global or concentrated in a few states.

---

## 6) Artifact 3: VDR / Silhouette CSV

File:

- `clustering_results.csv`

Columns:

- `variant`
- `state_idx`
- `timestep`
- `probe_variance` (from critical-state discovery)
- `best_k`
- `vdr`
- `ch`
- `sil`

How rows are generated:

- One row per successful `variant x state` best-clustering result.
- If clustering fails for a state/variant, that row is skipped.

How to use in analysis:

- sort by `vdr` descending to find most multimodal states.
- compare `vdr` and `sil` jointly:
  - high VDR + high silhouette: clean branch structure.
  - high VDR + low silhouette: variance explained, but weakly separated clusters.
  - low VDR + high silhouette: rare; usually indicates small/fragile partition effects.
- inspect `best_k` distribution to understand complexity of action branching.

---

## 7) Additional Generated Clustering Outputs

Besides the three requested artifacts, the script also creates:

- `vdr_comparison.png`: state-wise grouped bar chart of VDR by variant.
- `best_k_distribution.png`: histogram of best `k` per variant.
- `clustering_metrics.txt`: markdown table with means/std summary.
- `per_variant/<variant>/stateXX.png` (+ `.html` if Plotly): per-variant state-level 3D cluster visualization.

These are useful for debugging and sanity checks.

---

## 8) Diversity Metrics (`analyze_diversity.py`) vs Clustering Metrics

This script computes spread-style metrics from `action_chunks`:

- `per_dim_std`: mean std per action dimension.
- `within_l2`: mean pairwise L2 distance within variant.
- `coverage`: occupancy fraction of a 10x10 PCA-2D grid.
- `across_l2`: average distance to pooled chunks from other variants (non-gt variants).

Key distinction:

- Diversity metrics measure **how spread out** action samples are.
- Clustering metrics (VDR/silhouette) measure **how cluster-structured** that spread is.

You can have:

- High diversity but low multimodality (broad but unimodal cloud).
- Moderate diversity but high multimodality (two clean, compact branches).

So both families are complementary, not interchangeable.

---

## 9) Recommended Interpretation Workflow

For one task:

1. Start from `vdr_violin.png` for overall ranking.
2. Use `vdr_comparison.png` to locate state-specific differences.
3. Open `clustering_results.csv` to inspect exact `vdr/sil/best_k` per state.
4. Check corresponding `ghost_comparison_stateXX.png` for qualitative confirmation.
5. Cross-check with `diversity_metrics.txt` to separate "spread increase" vs "true modal branching".

This prevents over-interpreting a single metric in isolation.

---

## 10) Practical Notes and Caveats

- `best_k` is chosen by VDR only. If you care more about compact/separated clusters, you might also consider silhouette-driven or multi-objective selection.
- Silhouette can be `NaN` in small-sample or degenerate cases.
- The trajectory metric and `abs_action` assumption strongly affect results.
- PCA projections in ghost plots are for visualization only; clustering is performed before plotting.
- Critical-state ranking depends on probe sampling and random seeds; use fixed seeds for fair comparisons.

---

## 11) Minimal Command Examples

Collect multimodality data:

```bash
python examples/collect_multimodal_viz.py \
  --task lift_ph_state \
  --run "baseline:checkpoints/...pt:task=lift_ph_state" \
  --run "flow_intent:checkpoints/...pt:task=lift_ph_state_flow_intent:+network.arch_variant=flow_intent" \
  --run "hierarchical_emb:checkpoints/...pt:task=lift_ph_state_hierarchical_emb" \
  --n-rollouts 20 \
  --n-samples 50 \
  --n-critical 10 \
  --out rollouts/lift_ph_state_multimodal.pkl
```

Run clustering multimodality analysis:

```bash
python examples/analyze_multimodal_clustering.py \
  --data rollouts/lift_ph_state_multimodal.pkl \
  --task lift_ph_state \
  --out-dir rollouts/lift_ph_state_cluster_figs \
  --k-min 2 --k-max 6
```

Run diversity analysis:

```bash
python examples/analyze_diversity.py \
  --data rollouts/lift_ph_state_multimodal.pkl \
  --task lift_ph_state \
  --out-dir rollouts/lift_ph_state_multimodal_figs
```

---

## 12) Quick Glossary

- **Critical state**: observation selected for high action-sampling variance.
- **Action chunk**: predicted action sequence segment of shape `(act_steps, act_dim)`.
- **Trajectory (for clustering)**: transformed representation from action chunk integration/anchoring.
- **VDR**: fraction of total variance explained by clustering split.
- **Silhouette**: cluster compactness/separation score in feature space.
- **Ghost plot**: overlaid 3D trajectory visualization for sampled actions at same state.

---

## 13) Rule-of-Thumb Reading Cheatsheet

- "Is model more multimodal overall?"
  - Compare median/center of `vdr_violin.png`.

- "Where does multimodality happen?"
  - Inspect `vdr_comparison.png` and CSV by `state_idx`.

- "Are clusters meaningful or noisy?"
  - Prefer states with high `vdr` and positive/high `sil`.

- "Do visuals agree with metrics?"
  - Check `ghost_comparison_stateXX.png` for those rows.

- "Is gain just spread, not modes?"
  - Compare against `diversity_metrics.txt`/`diversity_metrics.png`.

---

If needed, this guide can be extended with a section mapping each function call and variable in code to the exact equations step-by-step.
