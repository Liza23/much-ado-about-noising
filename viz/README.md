# Visualization Pipeline for Action Multimodality Analysis

This folder contains tools for analyzing and visualizing **action multimodality** — whether a policy produces diverse, meaningfully different action trajectories from the same or similar states.

---

## File Overview

### 1. `clustering_shared.py` — Shared Utilities

Low-level helpers used by both `mass_droid_clustering.py` and `analyze_task_multimodality_trajectories.py`.

**Key functions:**

| Function | Purpose |
|----------|---------|
| `LazyActionsFromNpzRanges` | Memory-efficient indexable dataset over many checkpoint NPZ files. Loads actions lazily from sharded `.npz` files without loading everything into RAM. |
| `load_actions_with_fallback` | Loads a merged `actions.npz`; falls back to lazy indexing of `episodes/*/actions.npz` or `shards/*_actions.npz` if the merged file is missing. |
| `integrate_joint_velocity_chunk` | Integrates joint velocity actions `(T, A)` into displacement trajectories `dq = cumsum(v * dt)`, anchored at zero. Used because DROID actions are joint velocities, not positions. |
| `prepare_trajectories_and_features_from_actions` | Converts raw `(K, T, A)` action chunks into: integrated trajectories (list of `(T, D)`), flattened feature matrix `(K, flat_dim)` for clustering, and per-point indexing arrays. |
| `compute_minkowski_distance_matrix` | Computes pairwise symmetrized L2 Minkowski distance between trajectory sets. This is the core distance metric for spectral clustering. |
| `minkowski_variance_from_D` / `total_variance_minkowski` | Variance proxy: `Var = 0.5 * E[d^2]` over off-diagonal entries. |
| `weighted_incluster_variance_minkowski` | Per-cluster weighted variance (analogous to within-cluster sum of squares for trajectories). |

**Distance metric:** The symmetrized L2 Minkowski distance between two trajectories `X` and `Y` is:
```
d1^2 = sum_i min_j ||x_i - y_j||^2
d2^2 = sum_j min_i ||y_j - x_i||^2
d = sqrt((d1^2 + d2^2) / 2)
```

---

### 2. `mass_droid_clustering.py` — Per-State Spectral Clustering

Runs spectral clustering on action chunks **per observation state** to quantify multimodality. Originally designed for DROID dataset analysis but applicable to any dataset with per-state action samples.

**Pipeline (per state):**

1. **Load** action chunks for state `i` — shape `(K, T, A)` where K = number of samples from this state
2. **Integrate** joint velocities into displacement trajectories using `dt = 1/fps`
3. **Compute distance matrix** — symmetrized Minkowski distance between all K trajectories
4. **Build affinity matrix** — `A = exp(-D^2 / (2 * sigma^2))` with sigma = median of positive distances
5. **Spectral clustering** — try k from `k_min` to `k_max`, evaluate each k using:
   - **Variance drop ratio (VDR):** `(total_var - within_cluster_var) / total_var` (like R^2)
   - **Calinski-Harabasz index**
   - **Silhouette score** (on Euclidean features)
6. **Select best k** by the chosen metric (default: VDR)
7. **Checkpoint** results per-state to `per_state/state_XXXXXX.npz`

**Outputs:**
- `results.csv` — all states x all k values with metrics
- `top_states.csv` — top N most multimodal states (highest VDR)
- Per-state `.npz` checkpoints with cluster labels

**Visualization:** For top states, generates 3D scatter plots (Matplotlib PNG + Plotly HTML) showing action chunks colored by cluster, with trajectory lines connecting timesteps within each chunk.

**CLI:**
```bash
python viz/mass_droid_clustering.py \
    --summary_csv data/summary.csv \
    --actions_npz data/actions.npz \
    --outdir multimodality_out \
    --k_min 2 --k_max 8 --top_n 50
```

---

### 3. `analyze_task_multimodality_trajectories.py` — Episode-Level Trajectory Visualization

Visualizes per-episode trajectories with "ghost" action chunks overlaid at each state, colored by the variance-drop-ratio from clustering results.

**Two visualization modes:**

| Mode | Anchors | Ghost trajectories |
|------|---------|-------------------|
| **Integrated q(t)** (`--lerobot_root` provided) | Joint positions loaded from LeRobot parquet files | Integrate joint velocity actions from q(t) anchor |
| **Action space** (fallback) | Start/end of action chunks | Raw action chunks in first-3-dims or PCA space |

**Pipeline:**
1. Load clustering results CSV (from `mass_droid_clustering.py`)
2. Load actions NPZ
3. For each episode, identify states along the trajectory
4. At each state, overlay "ghost" action chunk trajectories showing the K different action samples
5. Color states by VDR — high VDR = highly multimodal decision point
6. Apply global PCA(3) across all plotted points for consistent embedding
7. Output interactive Plotly HTML per episode

**Key concept:** States colored in yellow/bright (high VDR) are "critical decision points" where the policy had meaningfully different action modes. This directly shows **where** multimodality matters in the task.

---

## How These Files Work Together

```
Dataset (actions per state)
         │
         ▼
┌─────────────────────────┐
│  mass_droid_clustering   │  Per-state spectral clustering
│  (mass_droid_clustering  │  → VDR scores, cluster labels
│   .py)                   │  → top_states.csv, results.csv
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  analyze_task_           │  Episode-level trajectory viz
│  multimodality_          │  with ghost overlays at
│  trajectories.py         │  high-VDR states
└─────────────────────────┘
         │
    clustering_shared.py  ←── shared distance metrics,
                              integration, variance helpers
```

---

## Integration with MIP Policy Evaluation

The files in `examples/` provide a **separate but complementary** pipeline for comparing trained MIP policy variants:

| File | Purpose |
|------|---------|
| `collect_diversity_rollouts.py` | Roll out policy variants (baseline, flow_intent, hierarchical_emb), collect action chunks + steerability data |
| `analyze_diversity.py` | UMAP/PCA embeddings, per-dim std, pairwise L2, coverage metrics, steerability plots |
| `eval_visualize.py` | Full eval suite: 3D trajectories, intent accuracy, sensitivity analysis, success rates → wandb |

The `viz/` scripts focus on **dataset-level multimodality** (how multimodal are the actions in the training data at each state?), while `examples/` scripts focus on **policy-level diversity** (how diverse are a trained policy's outputs compared across variants?).
