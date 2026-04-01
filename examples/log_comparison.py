"""Print a summary table of final eval results across ablations.

Usage:
    python examples/log_comparison.py \
        --run baseline_chunk8:6564867 \
        --run intent_cv_chunk8:6564869 \
        --run hierarchical_chunk8:6564835 \
        ...

    Label format: <config>_chunk<N>   e.g. baseline_chunk8, hierarchical_cross_chunk4
    Job ID is used to find the log at logs/<label_prefix>_<jobid>.err

Or pass explicit paths:
    --run baseline_chunk8:path/to/file.err
"""

import argparse
import os
import re

# Canonical row order for the table
ROW_ORDER = [
    # lift
    "baseline",
    "intent_cv",
    "intent_learned",
    "hierarchical",
    "hierarchical_final",
    "hierarchical_cross",
    "intent_sequence",
    "flow_intent",
    # square
    "square_baseline",
    "square_intent_learned",
    "square_hierarchical",
    "square_intent_sequence",
    "square_flow_intent",
]

# Map config name prefix → sbatch filename prefix (for auto log path resolution)
LOG_PREFIX = {
    "baseline":              "lift_baseline",
    "intent_cv":             "lift_intent",
    "intent_learned":        "lift_intent_learned",
    "hierarchical":          "lift_hierarchical",
    "hierarchical_final":    "lift_hierarchical_final",
    "hierarchical_cross":    "lift_hierarchical_cross",
    "intent_sequence":       "lift_intent_sequence",
    "flow_intent":           "lift_flow_intent",
    "square_baseline":       "square_baseline",
    "square_intent_learned": "square_intent_learned",
    "square_hierarchical":   "square_hierarchical",
    "square_intent_sequence":"square_intent_sequence",
    "square_flow_intent":    "square_flow_intent",
}


def resolve_path(label: str, job_id: str) -> str:
    """Turn a job ID into a log path, or return as-is if it looks like a path."""
    if os.path.exists(job_id):
        return job_id
    # Strip _chunkN to get config name
    m = re.match(r"^(.+?)(?:_chunk\d+)?$", label)
    config = m.group(1) if m else label
    prefix = LOG_PREFIX.get(config, config)
    # Try chunk-specific prefix first
    m2 = re.match(r"^(.+)_chunk(\d+)$", label)
    if m2:
        chunk_prefix = f"lift_{m2.group(1).replace('_', '_')}_chunk{m2.group(2)}"
        candidate = f"logs/{chunk_prefix}_{job_id}.err"
        if os.path.exists(candidate):
            return candidate
    candidate = f"logs/{prefix}_{job_id}.err"
    return candidate


def parse_log(path: str):
    """Return (steps, losses, evals) from a training .err log."""
    steps, losses, evals = [], [], []
    if not os.path.exists(path):
        return steps, losses, evals
    with open(path) as f:
        for line in f:
            m = re.search(r"\[Step (\d+)\].*? loss: ([\d.e+\-]+)", line)
            if m and "mean_success" not in line:
                steps.append(int(m.group(1)) + 1)
                losses.append(float(m.group(2)))
            m2 = re.search(r"\[Step (\d+)\].*?mean_success_9: ([\d.e+\-]+)", line)
            if m2:
                evals.append((int(m2.group(1)) + 1, float(m2.group(2))))
    return steps, losses, evals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", default=[],
                        metavar="LABEL:JOBID_OR_PATH",
                        help="e.g. baseline_chunk8:6564867")
    args = parser.parse_args()

    if not args.run:
        parser.print_help()
        return

    # Parse all runs
    records = []
    for entry in args.run:
        label, job_or_path = entry.split(":", 1)
        path = resolve_path(label, job_or_path)
        steps, losses, evals = parse_log(path)

        m = re.match(r"^(.+)_chunk(\d+)$", label)
        config = m.group(1) if m else label
        chunk  = int(m.group(2)) if m else "?"

        final_sr   = evals[-1][1]  if evals  else None
        best_sr    = max(e[1] for e in evals) if evals else None
        last_step  = steps[-1]     if steps  else None
        n_evals    = len(evals)
        job_id     = job_or_path if not os.path.exists(job_or_path) else "—"
        found      = os.path.exists(path)

        records.append(dict(
            label=label, config=config, chunk=chunk, job_id=job_id,
            path=path, found=found,
            final_sr=final_sr, best_sr=best_sr,
            last_step=last_step, n_evals=n_evals,
        ))

    # ── Per-run status ────────────────────────────────────────────────────
    print("\n── Run status ──────────────────────────────────────────────────────")
    fmt = "{:<35s}  {:>8s}  {:>7s}  {:>7s}  {:>7s}  {:>8s}  {}"
    print(fmt.format("label", "job", "steps", "evals", "final", "best", "path"))
    print("─" * 95)
    for r in records:
        print(fmt.format(
            r["label"],
            str(r["job_id"]),
            str(r["last_step"] // 1000) + "k" if r["last_step"] else "—",
            str(r["n_evals"]),
            f"{r['final_sr']:.0%}" if r["final_sr"] is not None else "—",
            f"{r['best_sr']:.0%}"  if r["best_sr"]  is not None else "—",
            r["path"] if r["found"] else f"NOT FOUND: {r['path']}",
        ))

    # ── Chunk-size table ──────────────────────────────────────────────────
    chunk_sizes = sorted({r["chunk"] for r in records if isinstance(r["chunk"], int)})
    table_data  = {}  # config -> chunk -> (final_sr, job_id)
    for r in records:
        table_data.setdefault(r["config"], {})[r["chunk"]] = (r["final_sr"], r["job_id"])

    known   = [c for c in ROW_ORDER if c in table_data]
    unknown = sorted(c for c in table_data if c not in ROW_ORDER)
    configs = known + unknown

    col_w = 14
    header = f"\n── Final success rate (chunk size →) ───────────────────────────────\n"
    header += f"{'config':<30s}" + "".join(f"{'chunk='+str(c):>{col_w}s}" for c in chunk_sizes)
    print(header)
    print("─" * (30 + col_w * len(chunk_sizes)))
    for cfg in configs:
        row = f"{cfg:<30s}"
        for c in chunk_sizes:
            entry = table_data[cfg].get(c)
            if entry is None:
                cell = "—"
            else:
                sr, jid = entry
                cell = f"{sr:.0%}({jid})" if sr is not None else f"…({jid})"
            row += f"{cell:>{col_w}s}"
        print(row)

    print()


if __name__ == "__main__":
    main()
