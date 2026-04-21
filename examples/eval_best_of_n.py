"""Best-of-N success rate evaluation for LIBERO suite tasks.

Estimates single-rollout success rate p per task from eval.num_rollouts episodes,
then computes  best_of_n(p, n) = 1 - (1 - p)^n  for each n in eval.n_values.

Policy type is auto-detected from the checkpoint keys:
  - "intent_flow_map" present  → FlowIntentAgent  (BC + Intent)
  - "actor" + "critic" present → ResidualPARLAgent (PARL + Intent)
  - otherwise                  → TrainingAgent     (BC baseline)

Usage:
    # BC baseline (state)
    python examples/eval_best_of_n.py \\
        task=libero_spatial_suite_state network=mlp \\
        +eval.checkpoint=/path/to/bc.pt

    # BC baseline (image)
    python examples/eval_best_of_n.py \\
        task=libero_object_suite_image network=mlp \\
        +eval.checkpoint=/path/to/bc_image.pt

    # BC + Intent
    python examples/eval_best_of_n.py \\
        task=libero_spatial_suite_state_flow_intent network=mlp_flow_intent \\
        +eval.checkpoint=/path/to/bc_intent.pt

    # PARL + Intent
    python examples/eval_best_of_n.py \\
        task=libero_spatial_suite_state_flow_intent network=mlp_flow_intent \\
        +eval.checkpoint=/path/to/parl_best.pt \\
        +eval.flow_intent_ckpt=/path/to/bc_intent.pt
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import hydra
import loguru
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict

os.environ.setdefault("MUJOCO_GL", "egl")

from mip.agent import TrainingAgent
from mip.config import Config
from mip.datasets.libero_dataset import make_dataset
from mip.envs.libero.libero_env_wrapper import make_vec_env
from mip.flow_intent_agent import FlowIntentAgent
from mip.residual_parl.parl_agent import PARLConfig, ResidualPARLAgent
from mip.torch_utils import set_seed

torch.set_float32_matmul_precision("high")


# ─────────────────────────────────────────────────────────────────────────────
# Policy wrappers — uniform act() interface
# ─────────────────────────────────────────────────────────────────────────────

class BCPolicy:
    """Wraps TrainingAgent (baseline, no intent)."""

    def __init__(self, agent: TrainingAgent, config: Config, device: str):
        self.agent = agent
        self.config = config
        self.device = device
        self._s = config.task.obs_steps - 1
        self._is_image = getattr(config.task, "obs_type", "state") == "image"

    def act(self, obs_tensor) -> np.ndarray:
        """obs_tensor: TensorDict (image) or Tensor (B,To,obs_dim) normalized → (B,act_steps,act_dim)"""
        B = obs_tensor.shape[0] if self._is_image else obs_tensor.shape[0]
        act_0 = torch.randn(B, self.config.task.horizon, self.config.task.act_dim, device=self.device)
        obs_in = obs_tensor if self._is_image else {"state": obs_tensor}
        with torch.no_grad():
            act_normed = self.agent.sample(act_0=act_0, obs=obs_in, use_ema=True)
        return act_normed[:, self._s: self._s + self.config.task.act_steps].cpu().numpy()


class FlowIntentPolicy:
    """Wraps FlowIntentAgent (BC + Intent, Config A)."""

    def __init__(self, agent: FlowIntentAgent, config: Config):
        self.agent = agent
        self.config = config
        self._s = config.task.obs_steps - 1

    def act(self, obs_tensor: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            act_normed = self.agent.sample(obs=obs_tensor, use_ema=True)
        return act_normed[:, self._s: self._s + self.config.task.act_steps].cpu().numpy()


class PARLPolicy:
    """Wraps FlowIntentAgent + ResidualPARLAgent (PARL + Intent)."""

    def __init__(self, fi_agent: FlowIntentAgent, parl_agent: ResidualPARLAgent, config: Config):
        self.fi_agent = fi_agent
        self.parl_agent = parl_agent
        self.config = config
        self._s = config.task.obs_steps - 1

    def act(self, obs_tensor) -> np.ndarray:
        B = obs_tensor.shape[0]
        act_steps = self.config.task.act_steps
        act_dim = self.config.task.act_dim

        with torch.no_grad():
            base_actions = self.fi_agent.sample(obs=obs_tensor, use_ema=True)
        base_chunk = base_actions[:, self._s: self._s + act_steps].cpu().numpy()
        # PARL actor uses raw state (not images); extract state tensor if TensorDict
        state_np = (
            obs_tensor["state"].cpu().numpy()
            if hasattr(obs_tensor, "__getitem__") and not isinstance(obs_tensor, torch.Tensor)
            else obs_tensor.cpu().numpy()
        )
        obs_flat = state_np.reshape(B, -1)

        result = np.zeros((B, act_steps, act_dim), dtype=np.float32)
        for i in range(B):
            a_exec, _, _ = self.parl_agent.sample_action(
                obs_flat[i], base_chunk[i].reshape(-1), deterministic=True
            )
            result[i] = a_exec.reshape(act_steps, act_dim)
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Policy loading
# ─────────────────────────────────────────────────────────────────────────────

def load_policy(
    checkpoint: str,
    flow_intent_ckpt: str,
    config: Config,
    dataset,
    device: str,
) -> BCPolicy | FlowIntentPolicy | PARLPolicy:
    """Auto-detect checkpoint type and return the appropriate policy wrapper."""
    probe = torch.load(checkpoint, map_location="cpu", weights_only=False)

    if isinstance(probe, dict) and "actor" in probe and "critic" in probe:
        # PARL checkpoint
        assert flow_intent_ckpt, "PARL checkpoint requires +eval.flow_intent_ckpt"
        fi_agent = FlowIntentAgent(config)
        fi_agent.load(flow_intent_ckpt, load_optimizer=False)
        fi_agent.eval()

        obs_dim_flat = config.task.obs_steps * dataset[0]["obs"]["state"].shape[-1]
        parl_cfg = PARLConfig(device=device)
        parl_agent = ResidualPARLAgent(
            config=parl_cfg,
            obs_dim=obs_dim_flat,
            act_dim=config.task.act_dim,
            query_freq=config.task.act_steps,
        )
        parl_agent.set_flow_intent(fi_agent)
        parl_agent.load(checkpoint)
        loguru.logger.info("Loaded PARL + Intent policy")
        return PARLPolicy(fi_agent, parl_agent, config)

    if isinstance(probe, dict) and "intent_flow_map" in probe:
        # FlowIntentAgent checkpoint
        agent = FlowIntentAgent(config)
        agent.load(checkpoint, load_optimizer=False)
        agent.eval()
        loguru.logger.info("Loaded BC + Intent (FlowIntentAgent) policy")
        return FlowIntentPolicy(agent, config)

    # Default: TrainingAgent (BC baseline)
    agent = TrainingAgent(config)
    agent.load(checkpoint, load_optimizer=False)
    agent.eval()
    loguru.logger.info("Loaded BC (TrainingAgent) policy")
    return BCPolicy(agent, config, device)


# ─────────────────────────────────────────────────────────────────────────────
# Best-of-N computation
# ─────────────────────────────────────────────────────────────────────────────

def best_of_n(successes: list[bool], n: int) -> float:
    """Analytical best-of-n from observed success rate: 1 - (1 - p)^n."""
    p = float(np.mean(successes))
    return 1.0 - (1.0 - p) ** n


# ─────────────────────────────────────────────────────────────────────────────
# Per-task rollout
# ─────────────────────────────────────────────────────────────────────────────

def rollout_task(
    policy: BCPolicy | FlowIntentPolicy | PARLPolicy,
    bddl_file: str,
    config: Config,
    dataset,
    device: str,
    num_rollouts: int,
    seed: int = 0,
) -> list[bool]:
    """Run num_rollouts episodes on one task. Returns per-episode success flags."""
    original_bddl = config.task.bddl_file
    config.task.bddl_file = bddl_file
    envs = make_vec_env(config.task, seed=seed)
    config.task.bddl_file = original_bddl

    num_envs = config.task.num_envs
    act_steps = config.task.act_steps
    obs_type = getattr(config.task, "obs_type", "state")
    image_obs_keys = list(getattr(config.task, "image_obs_keys", None) or [])
    successes: list[bool] = []

    num_batches = (num_rollouts + num_envs - 1) // num_envs
    for _ in range(num_batches):
        obs_raw, _ = envs.reset()
        ep_reward = np.zeros(num_envs)
        t = 0

        while t < config.task.max_episode_steps:
            if obs_type == "image":
                obs_dict: dict[str, torch.Tensor] = {
                    "state": torch.tensor(
                        dataset.normalizer["obs"]["state"].normalize(obs_raw["state"].astype(np.float32)),
                        device=device, dtype=torch.float32,
                    )
                }
                for img_key in image_obs_keys:
                    obs_dict[img_key] = torch.tensor(
                        obs_raw[img_key].astype(np.float32), device=device, dtype=torch.float32
                    )
                obs_tensor = TensorDict(obs_dict, batch_size=num_envs)
            else:
                obs_norm = dataset.normalizer["obs"]["state"].normalize(obs_raw.astype(np.float32))
                obs_tensor = torch.tensor(obs_norm, device=device, dtype=torch.float32)

            act_norm = policy.act(obs_tensor)  # (B, act_steps, act_dim)
            act = dataset.normalizer["action"].unnormalize(act_norm)

            obs_raw, reward, terminated, truncated, _ = envs.step(act)
            ep_reward += reward
            t += act_steps

            if np.all(terminated | truncated):
                break

        successes.extend(ep_reward[i] > 0 for i in range(num_envs))
        if len(successes) >= num_rollouts:
            break

    envs.close()
    return successes[:num_rollouts]


# ─────────────────────────────────────────────────────────────────────────────
# Suite evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_suite(
    policy: BCPolicy | FlowIntentPolicy | PARLPolicy,
    bddl_files: list[str],
    config: Config,
    dataset,
    device: str,
    num_rollouts: int,
    n_values: list[int],
    seed: int = 0,
    output_path: str = "",
    checkpoint: str = "",
) -> dict[str, dict]:
    """Evaluate policy on all tasks. Returns task_name → {p_single, best_of_n}.

    Saves results incrementally after each task so progress survives interrupts.
    Re-running with the same output_path resumes from where it left off.
    """
    # Load existing partial results if resuming
    results: dict[str, dict] = {}
    if output_path and Path(output_path).exists():
        with open(output_path) as f:
            saved = json.load(f)
        results = saved.get("results", {})
        if results:
            loguru.logger.info(f"Resuming: {len(results)} tasks already done, skipping them.")

    for bddl_file in bddl_files:
        task_name = Path(bddl_file).stem
        if task_name in results:
            loguru.logger.info(f"[{task_name}] already done, skipping.")
            continue

        loguru.logger.info(f"[{task_name}] running {num_rollouts} rollouts ...")
        successes = rollout_task(policy, bddl_file, config, dataset, device, num_rollouts, seed)
        p = float(np.mean(successes))
        bon = {n: best_of_n(successes, n) for n in n_values}

        results[task_name] = {"p_single": p, "best_of_n": bon}
        loguru.logger.info(
            f"  p={p:.3f}  "
            + "  ".join(f"BoN({n})={bon[n]:.3f}" for n in n_values)
        )

        # Save incrementally after every task
        if output_path:
            out = {"checkpoint": checkpoint, "num_rollouts": num_rollouts,
                   "n_values": n_values, "results": results}
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(out, f, indent=2)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Table printing
# ─────────────────────────────────────────────────────────────────────────────

def print_table(results: dict[str, dict], n_values: list[int], label: str = "") -> None:
    col_w = max((len(t) for t in results), default=20) + 2
    headers = ["p_single"] + [f"BoN({n})" for n in n_values]
    sep = "-" * (col_w + 12 * len(headers))

    if label:
        print(f"\n{label}")
    print(f"{'Task':<{col_w}}" + "".join(f"{h:>12}" for h in headers))
    print(sep)

    mean_p, mean_bon = [], {n: [] for n in n_values}
    for task, data in results.items():
        p, bon = data["p_single"], data["best_of_n"]
        mean_p.append(p)
        print(f"{task:<{col_w}}" + f"{p:>12.3f}" + "".join(f"{bon[n]:>12.3f}" for n in n_values))
        for n in n_values:
            mean_bon[n].append(bon[n])

    print(sep)
    print(
        f"{'MEAN':<{col_w}}"
        + f"{np.mean(mean_p):>12.3f}"
        + "".join(f"{np.mean(mean_bon[n]):>12.3f}" for n in n_values)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="configs/", config_name="eval_best_of_n")
def main(raw_cfg: DictConfig) -> None:
    cfg_dict = OmegaConf.to_container(raw_cfg, resolve=True)
    eval_cfg = cfg_dict.get("eval", {})

    config = Config(
        optimization=OmegaConf.structured(raw_cfg.optimization) if "optimization" in raw_cfg else None,
        network=OmegaConf.structured(raw_cfg.network) if "network" in raw_cfg else None,
        task=OmegaConf.structured(raw_cfg.task) if "task" in raw_cfg else None,
        log=OmegaConf.structured(raw_cfg.log) if "log" in raw_cfg else None,
    )

    checkpoint: str = eval_cfg["checkpoint"]
    flow_intent_ckpt: str = eval_cfg.get("flow_intent_ckpt", "")
    num_rollouts: int = eval_cfg.get("num_rollouts", 50)
    n_values: list[int] = eval_cfg.get("n_values", [1, 10, 50, 100, 500, 1000, 2000])
    num_envs: int = eval_cfg.get("num_envs", 5)
    seed: int = eval_cfg.get("seed", 0)
    output_path: str = eval_cfg.get("output_path", "")

    device = config.optimization.device
    set_seed(seed)
    config.task.num_envs = num_envs

    dataset = make_dataset(config.task, mode="train")
    loguru.logger.info(f"Dataset loaded ({len(dataset)} samples)")

    obs_type = getattr(config.task, "obs_type", "state")
    if obs_type == "image":
        # obs_dim for image policies is the encoder embedding dim (set in network config)
        config.task.obs_dim = config.network.emb_dim
    else:
        # Set obs_dim from dataset (FlowIntentAgent uses base obs_dim; Config B would bump it)
        config.task.obs_dim = dataset[0]["obs"]["state"].shape[-1]

    bddl_files: list[str] = config.task.bddl_files
    assert bddl_files, "config.task.bddl_files must be non-empty; use a suite task config."
    loguru.logger.info(f"Suite: {len(bddl_files)} tasks")

    policy = load_policy(checkpoint, flow_intent_ckpt, config, dataset, device)
    results = evaluate_suite(
        policy, bddl_files, config, dataset, device, num_rollouts, n_values, seed,
        output_path=output_path, checkpoint=checkpoint,
    )

    print_table(results, n_values, label=f"Best-of-N | {Path(checkpoint).name}")

    if output_path:
        loguru.logger.info(f"Results saved → {output_path}")


if __name__ == "__main__":
    main()
