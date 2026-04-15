"""Collect free-sampling ghost rollouts for trajectory diversity comparison.

Two modes:

**Model-comparison mode** (default, use --run):
  For each critical state (selected by intent/action variance on the reference model):
    For each model variant (each --run):
      Run K independent free-sampling rollouts from the same starting state
      Record EEF trajectories + optional rendered frames

  The reference model is the FIRST --run.

**Perturb mode** (use --run for the model + --perturb-bddl for env variants):
  Fix one model; vary the BDDL file (env) across variants.
  For each env variant (each --perturb-bddl):
    Independently find critical states using action/intent variance in THAT env
    Run K independent free-sampling rollouts per critical state
  Output has per-env critical states so plot_ghost.py can label them correctly.

Usage (model comparison):
    python examples/collect_ghost_rollouts.py \\
        --run "flow_intent:{CKPT_FI}:task=libero_spatial_state_flow_intent:network=mlp_flow_intent" \\
        --run "baseline:{CKPT_BL}:task=libero_spatial_state" \\
        --n-rollouts 10 --n-critical 5 --k-ghost 8 --n-future-steps 80 \\
        --device cuda --out rollouts/ghost_model_cmp.pkl

Usage (perturb mode):
    python examples/collect_ghost_rollouts.py \\
        --run "flow_intent:{CKPT_FI}:task=libero_spatial_state_flow_intent:network=mlp_flow_intent" \\
        --perturb-bddl "normal:/path/to/P0.bddl" \\
        --perturb-bddl "perturb_mug:/path/to/P1_mug.bddl" \\
        --perturb-bddl "perturb_diffpos:/path/to/P2.bddl" \\
        --n-rollouts 10 --n-critical 3 --k-ghost 8 --n-future-steps 80 \\
        --device cuda --out rollouts/ghost_perturb.pkl

Output pkl (model-comparison mode)::

    {
        "mode": "model_comparison",
        "critical_states": [...],
        "runs": {
            "flow_intent": {"per_state": [[...], ...]},
            "baseline":    {"per_state": [[...], ...]},
        },
    }

Output pkl (perturb mode)::

    {
        "mode": "perturb",
        "model_label": "flow_intent",
        "runs": {
            "normal":       {"critical_states": [...], "per_state": [[...], ...]},
            "perturb_mug":  {"critical_states": [...], "per_state": [[...], ...]},
        },
    }
"""

import argparse
import os
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))
os.chdir(ROOT)

warnings.filterwarnings("ignore")

from collect_intent_outcomes_image import (
    BaselineAgentAdapter,
    make_vec_env,
    make_dataset,
    to_fi_obs,
    get_sim_state,
    restore_env_obs,
    update_obs_buf,
    get_eef_from_envs,
    render_frame,
    undo_action,
    _get_inner_step_env,
    _step_inner,
    maybe_register_libero_pro_objects,
)
from collect_diversity_rollouts import (
    load_config,
    parse_run_spec,
    setup_config_for_env,
    load_model,
)
from mip.agent import TrainingAgent
from mip.flow_intent_agent import FlowIntentAgent
from mip.torch_utils import set_seed


# ─────────────────────────────────────────────────────────────────────────────
# Rollout helpers
# ─────────────────────────────────────────────────────────────────────────────

def run_free_rollout(
    envs, obs_buf, agent, config, dataset, device, n_future_steps, capture_render,
    num_steps=9, fixed_intent=None,
):
    """Step env for n_future_steps.

    For FlowIntentAgent: if fixed_intent is provided, commits to it for the
    entire rollout (sample_given_intent at each chunk). Otherwise re-samples
    intent each chunk (free sampling).
    For baseline: calls agent.sample() normally.
    """
    s = config.task.obs_steps - 1
    act_steps = config.task.act_steps
    inner = _get_inner_step_env(envs, config)
    eef_traj = [get_eef_from_envs(envs, config)]
    frames = []
    if capture_render:
        f0 = render_frame(envs, config)
        if f0 is not None:
            frames.append(f0)
    done = False
    info = {}

    for step_i in range(0, n_future_steps, act_steps):
        fi_obs, _ = to_fi_obs(obs_buf, config, dataset, device)
        with torch.no_grad():
            if fixed_intent is not None and isinstance(agent, FlowIntentAgent):
                act_norm = agent.sample_given_intent(
                    obs=fi_obs, intent_vec=fixed_intent, use_ema=True, num_steps=num_steps
                )
            else:
                act_norm = agent.sample(obs=fi_obs, use_ema=True, num_steps=num_steps)
        act_un = dataset.normalizer["action"].unnormalize(act_norm.cpu().numpy())
        remaining = n_future_steps - step_i
        for a_i in range(min(act_steps, remaining)):
            act = undo_action(act_un[0, s + a_i], config, dataset)
            _, done, info = _step_inner(inner, act)
            eef_traj.append(get_eef_from_envs(envs, config))
            if capture_render:
                f = render_frame(envs, config)
                if f is not None:
                    frames.append(f)
            if done:
                break
        obs_buf = update_obs_buf(obs_buf, envs, config)
        if done:
            break

    return {
        "eef_trajectory": np.array(eef_traj),
        "frames": frames,
        "success": bool(info.get("success", False)),
        "n_steps": len(eef_traj) - 1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Critical state discovery (reuses Phase 1 logic from collect_intent_outcomes)
# ─────────────────────────────────────────────────────────────────────────────

def find_critical_states(config, agent, dataset, envs, args, device):
    """Run probing rollouts with the reference model; return top-k states by variance.

    For FlowIntentAgent: probes intent-space variance (goal spread).
    For any other agent: probes action-chunk variance (output diversity).
    """
    num_steps = 9
    n_probe = args.n_probe
    is_fi = isinstance(agent, FlowIntentAgent)
    print(
        f"Probing {args.n_rollouts} rollouts (n_probe={n_probe} per state, "
        f"variance={'intent' if is_fi else 'action'}) ..."
    )
    all_candidates = []
    n_done = 0

    while n_done < args.n_rollouts:
        obs, _ = envs.reset()
        t = 0

        while t < config.task.max_episode_steps:
            fi_obs, _ = to_fi_obs(obs, config, dataset, device)

            probe_values = []
            with torch.no_grad():
                for _ in range(n_probe):
                    if is_fi:
                        _, iv = agent.sample(
                            obs=fi_obs, use_ema=True, num_steps=num_steps, return_intent=True
                        )
                        probe_values.append(iv[0].cpu().numpy())
                    else:
                        act_norm = agent.sample(obs=fi_obs, use_ema=True, num_steps=num_steps)
                        probe_values.append(act_norm[0].cpu().numpy().ravel())
            variance = float(np.stack(probe_values).var(axis=0).mean())

            sim_state = get_sim_state(envs, config)
            eef_pos = get_eef_from_envs(envs, config)
            obs_snap = (
                obs.copy() if config.task.obs_type == "state"
                else {k: v.copy() for k, v in obs.items()}
            )
            all_candidates.append({
                "obs_raw": obs_snap,
                "sim_state": sim_state,
                "eef_pos": eef_pos,
                "timestep": t,
                "episode_idx": n_done,
                "variance": variance,
            })

            # Advance with one free-sampling step (reuse probe_values[0] to avoid extra sample)
            with torch.no_grad():
                act_norm = agent.sample(obs=fi_obs, use_ema=True, num_steps=num_steps)
            s = config.task.obs_steps - 1
            act_un = dataset.normalizer["action"].unnormalize(act_norm.cpu().numpy())
            act_chunk = act_un[:, s:s + config.task.act_steps]
            obs, _, terminated, truncated, _ = envs.step(act_chunk)
            t += config.task.act_steps
            if terminated.any() or truncated.any():
                break

        n_done += config.task.num_envs
        print(f"  Episode {n_done}/{args.n_rollouts}")

    # Filter: only states with enough remaining steps
    min_remaining = args.n_future_steps
    all_candidates = [
        c for c in all_candidates
        if c["timestep"] + min_remaining < config.task.max_episode_steps
    ]
    if args.max_timestep is not None:
        early = [c for c in all_candidates if c["timestep"] <= args.max_timestep]
        if early:
            all_candidates = early

    if not all_candidates:
        raise RuntimeError(
            f"No candidates with >= {min_remaining} steps remaining. "
            "Try reducing --n-future-steps or increasing --n-rollouts."
        )

    all_candidates.sort(key=lambda x: x["variance"], reverse=True)
    critical = all_candidates[: args.n_critical]
    print(f"Selected {len(critical)} critical states.")
    return critical


# ─────────────────────────────────────────────────────────────────────────────
# Per-model ghost rollouts
# ─────────────────────────────────────────────────────────────────────────────

def collect_ghost_for_model(
    label, ckpt_path, user_overrides, critical_states, args, device, ref_envs,
):
    """Load a model and run k_ghost free-sampling rollouts per critical state.

    Reuses the already-open ref_envs (same task, same env) — we just restore
    the sim state before each rollout so the model runs from the same initial
    configuration.
    """
    overrides = user_overrides + [
        f"optimization.device={device}",
        "task.num_envs=1",
        f"optimization.seed={args.seed}",
    ]
    config = load_config(overrides)
    setup_config_for_env(config, ref_envs)
    dataset = make_dataset(config)
    maybe_register_libero_pro_objects(config)

    agent, _ = load_model(ckpt_path, config, dataset, device)
    agent.eval()
    if isinstance(agent, TrainingAgent):
        agent = BaselineAgentAdapter(agent, config)

    num_steps = 9
    is_fi = isinstance(agent, FlowIntentAgent)
    per_state = []

    for ci, state in enumerate(critical_states):
        print(
            f"  [{label}] critical state {ci + 1}/{len(critical_states)} "
            f"t={state['timestep']} var={state['variance']:.5f}"
            + (" [committed-intent]" if is_fi else " [free-sampling]")
        )
        ghosts = []
        for k in range(args.k_ghost):
            obs_buf = restore_env_obs(
                ref_envs,
                state["sim_state"],
                config,
                ensure_fresh_reset=bool(args.render),
            )
            # For FlowIntentAgent: sample a fresh intent once and commit to it.
            fixed_intent = None
            if is_fi:
                fi_obs, _ = to_fi_obs(obs_buf, config, dataset, device)
                fixed_intent = agent.sample_intent(obs=fi_obs, use_ema=True, num_steps=num_steps)

            result = run_free_rollout(
                ref_envs,
                obs_buf,
                agent,
                config,
                dataset,
                device,
                n_future_steps=args.n_future_steps,
                capture_render=args.render,
                num_steps=num_steps,
                fixed_intent=fixed_intent,
            )
            ghosts.append(result)
            print(
                f"    ghost {k + 1}/{args.k_ghost}: "
                f"{result['n_steps']} steps, success={result['success']}, "
                f"end={result['eef_trajectory'][-1].round(3)}"
            )
        per_state.append(ghosts)

    return per_state


# ─────────────────────────────────────────────────────────────────────────────
# Perturb-mode: same model, different env BDDL variants
# ─────────────────────────────────────────────────────────────────────────────

def parse_perturb_bddl_spec(spec: str):
    """Parse 'label:bddl_file_path' → (label, bddl_path)."""
    parts = spec.split(":", 1)
    if len(parts) != 2:
        raise ValueError(
            f"--perturb-bddl must be 'label:bddl_file_path', got: {spec!r}"
        )
    return parts[0].strip(), parts[1].strip()


def collect_perturb_variant(
    label, bddl_path, ref_overrides, ref_ckpt, args, device
):
    """Run ghost rollouts for one env variant (different BDDL) using the reference model.

    Creates a fresh env with `bddl_path`, independently finds critical states
    via variance probing, then runs k_ghost free-sampling rollouts per state.
    Returns (critical_states_meta, per_state_data).
    """
    overrides = ref_overrides + [
        f"optimization.device={device}",
        "task.num_envs=1",
        f"optimization.seed={args.seed}",
        f"task.bddl_file={bddl_path}",
    ]
    config = load_config(overrides)

    envs = make_vec_env(config, seed=args.seed)
    setup_config_for_env(config, envs)
    dataset = make_dataset(config)
    maybe_register_libero_pro_objects(config)

    agent, _ = load_model(ref_ckpt, config, dataset, device)
    agent.eval()
    if isinstance(agent, TrainingAgent):
        agent = BaselineAgentAdapter(agent, config)

    # Find critical states in this env variant
    critical_states = find_critical_states(config, agent, dataset, envs, args, device)
    for state in critical_states:
        state.pop("obs_raw", None)

    # Ghost rollouts per critical state
    is_fi = isinstance(agent, FlowIntentAgent)
    num_steps = 9
    per_state = []
    for ci, state in enumerate(critical_states):
        print(
            f"  [{label}] critical state {ci + 1}/{len(critical_states)} "
            f"t={state['timestep']} var={state['variance']:.5f}"
            + (" [committed-intent]" if is_fi else " [free-sampling]")
        )
        ghosts = []
        for k in range(args.k_ghost):
            obs_buf = restore_env_obs(
                envs,
                state["sim_state"],
                config,
                ensure_fresh_reset=bool(args.render),
            )
            fixed_intent = None
            if is_fi:
                fi_obs, _ = to_fi_obs(obs_buf, config, dataset, device)
                fixed_intent = agent.sample_intent(obs=fi_obs, use_ema=True, num_steps=num_steps)

            result = run_free_rollout(
                envs, obs_buf, agent, config, dataset, device,
                n_future_steps=args.n_future_steps,
                capture_render=args.render,
                num_steps=num_steps,
                fixed_intent=fixed_intent,
            )
            ghosts.append(result)
            print(
                f"    ghost {k + 1}/{args.k_ghost}: "
                f"{result['n_steps']} steps, success={result['success']}, "
                f"end={result['eef_trajectory'][-1].round(3)}"
            )
        per_state.append(ghosts)

    envs.close()

    critical_states_meta = [
        {"eef_pos": s["eef_pos"], "timestep": s["timestep"], "variance": s["variance"]}
        for s in critical_states
    ]
    return critical_states_meta, per_state


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Collect free-sampling ghost rollouts for diversity comparison.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        dest="runs",
        metavar="label:ckpt:override...",
        help=(
            "Run spec (repeatable). Format: label:ckpt_path[:hydra_override ...]. "
            "In model-comparison mode, all --run variants are evaluated in the same env. "
            "In perturb mode (--perturb-bddl given), only the FIRST --run is used "
            "as the model; each --perturb-bddl defines an env variant."
        ),
    )
    parser.add_argument(
        "--perturb-bddl",
        action="append",
        default=None,
        dest="perturb_bddls",
        metavar="label:bddl_file_path",
        help=(
            "Perturb mode: fix the model from --run[0] and vary the env BDDL. "
            "Format: label:path/to/task.bddl (repeatable). "
            "Each variant gets its own critical-state discovery + ghost rollouts."
        ),
    )
    parser.add_argument("--n-rollouts", type=int, default=10,
                        help="Probe rollouts for critical state discovery (reference model)")
    parser.add_argument("--n-critical", type=int, default=5,
                        help="Number of critical states to collect ghost rollouts for")
    parser.add_argument("--n-probe", type=int, default=20,
                        help="Intent/action samples per state for variance probing")
    parser.add_argument("--k-ghost", type=int, default=8,
                        help="Free-sampling rollouts per model per critical state")
    parser.add_argument("--n-future-steps", type=int, default=80,
                        help="Steps to run forward per ghost rollout")
    parser.add_argument("--max-timestep", type=int, default=None,
                        help="Only pick critical states at or before this episode timestep")
    parser.add_argument("--render", action="store_true",
                        help="Capture rendered RGB frames during rollouts")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True, help="Output .pkl path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    ref_label, ref_ckpt, ref_overrides = parse_run_spec(args.runs[0])

    # ══════════════════════════════════════════════════════════════════════════
    # PERTURB MODE: fix model, vary BDDL env variant
    # ══════════════════════════════════════════════════════════════════════════
    if args.perturb_bddls:
        print(f"\n{'=' * 60}")
        print(f"PERTURB MODE — model: {ref_label}  |  ckpt: {ref_ckpt}")
        print(f"Env variants: {[p.split(':')[0] for p in args.perturb_bddls]}")
        print(f"{'=' * 60}")

        runs_data = {}
        for spec in args.perturb_bddls:
            env_label, bddl_path = parse_perturb_bddl_spec(spec)
            print(f"\n{'─' * 60}")
            print(f"Env variant: {env_label}  |  bddl: {bddl_path}")
            print(f"{'─' * 60}")
            try:
                crit_meta, per_state = collect_perturb_variant(
                    env_label, bddl_path, ref_overrides, ref_ckpt, args, args.device
                )
            except Exception as e:
                print(f"[ERROR] {env_label}: {e}")
                import traceback; traceback.print_exc()
                continue
            runs_data[env_label] = {
                "critical_states": crit_meta,
                "per_state": per_state,
            }

        output = {
            "mode": "perturb",
            "model_label": ref_label,
            "runs": runs_data,
        }

        with open(args.out, "wb") as f:
            pickle.dump(output, f)
        print(f"\nSaved perturb ghost rollouts to: {args.out}")
        for label, data in runs_data.items():
            n_states = len(data["per_state"])
            k = len(data["per_state"][0]) if n_states > 0 else 0
            print(f"  {label}: {n_states} states × {k} ghosts")
        return

    # ══════════════════════════════════════════════════════════════════════════
    # MODEL-COMPARISON MODE: fix env, vary model
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print(f"Reference model: {ref_label}  |  ckpt: {ref_ckpt}")
    print(f"{'=' * 60}")

    ref_overrides_full = ref_overrides + [
        f"optimization.device={args.device}",
        "task.num_envs=1",
        f"optimization.seed={args.seed}",
    ]
    ref_config = load_config(ref_overrides_full)
    ref_envs = make_vec_env(ref_config, seed=args.seed)
    setup_config_for_env(ref_config, ref_envs)
    ref_dataset = make_dataset(ref_config)
    maybe_register_libero_pro_objects(ref_config)

    ref_agent, _ = load_model(ref_ckpt, ref_config, ref_dataset, args.device)
    ref_agent.eval()
    if isinstance(ref_agent, TrainingAgent):
        ref_agent = BaselineAgentAdapter(ref_agent, ref_config)

    critical_states = find_critical_states(
        ref_config, ref_agent, ref_dataset, ref_envs, args, args.device
    )

    # Trim to keep only lightweight metadata (obs_raw can be large)
    for state in critical_states:
        _ = state.pop("obs_raw", None)

    # ── Phase 2: ghost rollouts per model ─────────────────────────────────────
    runs_data = {}
    for run_spec in args.runs:
        label, ckpt_path, user_overrides = parse_run_spec(run_spec)
        print(f"\n{'=' * 60}")
        print(f"Model: {label}  |  ckpt: {ckpt_path}")
        print(f"{'=' * 60}")
        try:
            per_state = collect_ghost_for_model(
                label, ckpt_path, user_overrides, critical_states,
                args, args.device, ref_envs,
            )
        except Exception as e:
            print(f"[ERROR] {label}: {e}")
            import traceback; traceback.print_exc()
            continue
        runs_data[label] = {"per_state": per_state}

    ref_envs.close()

    output = {
        "mode": "model_comparison",
        "critical_states": [
            {
                "eef_pos": s["eef_pos"],
                "timestep": s["timestep"],
                "variance": s["variance"],
            }
            for s in critical_states
        ],
        "runs": runs_data,
    }

    with open(args.out, "wb") as f:
        pickle.dump(output, f)
    print(f"\nSaved ghost rollouts to: {args.out}")
    for label, data in runs_data.items():
        n_states = len(data["per_state"])
        k = len(data["per_state"][0]) if n_states > 0 else 0
        print(f"  {label}: {n_states} states × {k} ghosts")


if __name__ == "__main__":
    main()
