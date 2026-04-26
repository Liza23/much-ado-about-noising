"""PostBC training pipeline.

Two-phase training:
  Phase 1: Train K-member MLP ensemble on bootstrapped dataset subsets, compute
           per-sample posterior variance, and save to disk.
  Phase 2: Train flow BC policy with perturbed action targets (PostBC Algorithm 2).

Reference: "Posterior Behavioral Cloning" (arXiv:2512.16911).
"""

import os
import sys
from pathlib import Path

# Make external/postbc importable without installation.
sys.path.insert(0, str(Path(__file__).parent.parent / "external" / "postbc"))

import hydra
import loguru
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

os.environ.setdefault("MUJOCO_GL", "egl")

from mip.agent import TrainingAgent  # noqa: E402
from mip.config import Config, PostBCConfig  # noqa: E402
from mip.datasets.robomimic_dataset import make_dataset  # noqa: E402
from mip.envs.robomimic.robomimic_env import make_vec_env  # noqa: E402
from mip.logger import Logger  # noqa: E402
from mip.torch_utils import limit_threads, set_seed  # noqa: E402

# Import from examples/ — reuse train() / eval() from train_robomimic.
sys.path.insert(0, str(Path(__file__).parent))
from train_robomimic import eval, train  # noqa: E402

from dataset import PostBCDataset  # noqa: E402
from ensemble import EnsemblePredictor  # noqa: E402

torch.set_float32_matmul_precision("high")


def _run_phase1(postbc_cfg: PostBCConfig, dataset, task_cfg, variance_path: Path, device: str) -> None:
    """Train ensemble and compute per-sample variance; save to variance_path."""
    loguru.logger.info(
        f"Phase 1: Training {postbc_cfg.ensemble_size}-member ensemble "
        f"({postbc_cfg.ensemble_epochs} epochs each) ..."
    )
    predictor = EnsemblePredictor(
        K=postbc_cfg.ensemble_size,
        hidden_dim=postbc_cfg.ensemble_hidden_dim,
        n_layers=postbc_cfg.ensemble_n_layers,
        lr=postbc_cfg.ensemble_lr,
    )
    predictor.train_ensemble(
        dataset,
        obs_steps=task_cfg.obs_steps,
        horizon=task_cfg.horizon,
        n_epochs=postbc_cfg.ensemble_epochs,
        batch_size=postbc_cfg.ensemble_batch_size,
        device=device,
    )
    variance = predictor.compute_variance(
        dataset,
        obs_steps=task_cfg.obs_steps,
        horizon=task_cfg.horizon,
        act_dim=task_cfg.act_dim,
        device=device,
    )
    loguru.logger.info(
        f"Phase 1 done. Variance shape: {variance.shape}, "
        f"mean={variance.mean():.4f}, max={variance.max():.4f}"
    )
    np.save(str(variance_path), variance)
    loguru.logger.info(f"Saved ensemble variance to {variance_path}")

    ensemble_path = variance_path.parent / "ensemble.pt"
    predictor.save(str(ensemble_path))
    loguru.logger.info(f"Saved ensemble checkpoint to {ensemble_path}")


@hydra.main(version_base=None, config_path="configs/", config_name="postbc")
def main(raw_cfg: DictConfig) -> None:
    os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
    if torch.cuda.is_available():
        torch.cuda.set_sync_debug_mode("warn")
        torch.set_float32_matmul_precision("high")

    # Build Config using OmegaConf.structured (same pattern as train_dsrl.py) to
    # avoid strict type-validation failures when fields like compile_mode are null.
    cfg_dict = OmegaConf.to_container(raw_cfg, resolve=True)
    config = Config(
        optimization=OmegaConf.structured(raw_cfg.optimization),
        network=OmegaConf.structured(raw_cfg.network),
        task=OmegaConf.structured(raw_cfg.task),
        log=OmegaConf.structured(raw_cfg.log),
    )
    postbc_cfg = PostBCConfig(**cfg_dict.get("postbc", {}))
    config.postbc = postbc_cfg
    config.mode = cfg_dict.get("mode", "train")
    config.log.log_dir = cfg_dict["log"]["log_dir"]

    set_seed(config.optimization.seed)
    limit_threads(1)
    logger = Logger(config)

    # Env setup (needed to determine obs_dim).
    envs = make_vec_env(config.task, seed=config.optimization.seed)
    obs, _ = envs.reset()
    config.task.obs_dim = obs.shape[-1]
    loguru.logger.info(f"obs_dim={config.task.obs_dim}, act_dim={config.task.act_dim}")

    device = config.optimization.device

    # Phase 1: ensemble variance computation.
    variance_path = (
        Path(config.postbc.variance_path)
        if config.postbc.variance_path
        else Path(logger._log_dir) / "ensemble_variance.npy"
    )

    if config.postbc.skip_ensemble_training:
        if not variance_path.exists():
            raise FileNotFoundError(
                f"skip_ensemble_training=True but variance_path not found: {variance_path}"
            )
        loguru.logger.info(f"Skipping Phase 1; loading variance from {variance_path}")
    else:
        # Use a standard dataset for Phase 1 (no perturbation).
        dataset_plain = make_dataset(config.task)
        _run_phase1(config.postbc, dataset_plain, config.task, variance_path, device)
        del dataset_plain

    # Phase 2: flow BC with PostBC perturbation.
    loguru.logger.info("Phase 2: training flow BC with PostBC perturbation ...")

    # Build PostBCDataset using the same args as make_dataset would, then wrap.
    dataset_plain = make_dataset(config.task)
    # PostBCDataset wraps RobomimicDataset; we copy its internals from the plain dataset
    # and add variance perturbation in __getitem__.  The cleanest approach is to
    # reconstruct using the same factory but swap the class — however RobomimicDataset
    # is a concrete class, so we monkeypatch __getitem__ on the plain instance.
    # Instead, use PostBCDataset directly by forwarding dataset_plain's underlying state.
    from mip.datasets.robomimic_dataset import RobomimicDataset
    assert isinstance(dataset_plain, RobomimicDataset), (
        "PostBC only supports RobomimicDataset (state obs) for now"
    )

    # Attach variance fields directly to the existing instance (duck-type upgrade).
    dataset_plain.__class__ = PostBCDataset
    dataset_plain.variance = np.load(str(variance_path)).astype(np.float32)
    dataset_plain.alpha = config.postbc.alpha
    dataset = dataset_plain
    loguru.logger.info(
        f"PostBC dataset: variance={dataset.variance.shape}, alpha={dataset.alpha}"
    )

    agent = TrainingAgent(config)
    loguru.logger.info(f"Agent created with obs_dim={config.task.obs_dim}")

    resume_state = None
    if config.optimization.model_path and config.optimization.model_path != "None":
        resume_state = agent.load(config.optimization.model_path, load_optimizer=True)
    elif config.optimization.auto_resume:
        checkpoint_base_name = (
            f"{config.task.env_name}_{config.task.env_type}_{config.task.obs_type}_"
            f"{config.optimization.loss_type}_{config.network.network_type}_"
            f"{config.network.emb_dim}_h{config.task.horizon}_seed{config.optimization.seed}_postbc"
        )
        model_latest = Path(logger.model_dir) / "model_latest.pt"
        if model_latest.exists():
            loguru.logger.info(f"Resuming from {model_latest}")
            resume_state = agent.load(str(model_latest), load_optimizer=True)
        else:
            ckpt = logger.find_latest_checkpoint(checkpoint_base_name)
            if ckpt:
                loguru.logger.info(f"Resuming from {ckpt}")
                resume_state = agent.load(str(ckpt), load_optimizer=True)

    if config.mode == "train":
        train(config, envs, dataset, agent, logger, resume_state=resume_state)
    else:
        raise ValueError(f"Unsupported mode for train_postbc: {config.mode}")


if __name__ == "__main__":
    main()
