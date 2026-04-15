from dataclasses import dataclass, field


@dataclass
class LogConfig:
    log_dir: str
    wandb_mode: str
    project: str
    group: str
    exp_name: str
    eval_freq: int = 20000
    log_freq: int = 1000
    save_freq: int = 10000
    eval_episodes: int = 10
    save_video: bool = False


@dataclass
class OptimizationConfig:
    seed: int = 0
    loss_type: str = "flow"
    loss_scale: float = 100.0
    norm_type: str = "l2"
    lr: float = 1e-4
    weight_decay: float = 1e-5
    num_steps: int = 1
    sample_mode: str = "stochastic"  # "zero", "mean"
    t_two_step: float = 0.9
    discrete_dt: float = 0.01
    grad_clip_norm: float = 10.0
    ema_rate: float = 0.995
    batch_size: int = 16
    gradient_steps: int = 300000
    warmup_ratio: float = 0.0
    rampup_ratio: float = 0.5
    min_value: float = 0.0
    max_value: float = 1.0
    model_path: str | None = None
    pretrained_ckpt: str | None = None  # Path to pretrained base checkpoint for finetuning (keys remapped for RenderAugmentedNetwork)
    interp_type: str = "linear"  # "linear" or "trig"
    device: str = "cuda"
    use_compile: bool = True  # Whether to use torch.compile for acceleration
    compile_mode: str = (
        "default"  # Compile mode: "default", "reduce-overhead", "max-autotune"
    )
    use_cudagraphs: bool = False  # Whether to use CUDA graphs (requires static shapes)
    auto_resume: bool = True  # Whether to automatically resume from checkpoint
    draft_loss_weight: float = 1.0  # Weight for draft head loss (VLA two-stage)


@dataclass
class NetworkConfig:
    network_type: str = "mlp"  # "mlp" or "cnn"
    num_layers: int = 4
    emb_dim: int = 512
    dropout: float = 0.1
    encoder_dropout: float = 0.0
    encoder_type: str = "mlp"  # "mlp", "per_step_mlp", "identity"
    expansion_factor: int = 4
    timestep_emb_dim: int = 128
    timestep_emb_type: str = "positional"  # Type of timestep embedding
    # State encoder configs
    num_encoder_layers: int = 2  # Number of layers for MLP encoder
    # Image encoder configs
    rgb_model_name: str = "resnet18"
    use_seq: bool = True
    keep_horizon_dims: bool = True
    # Transformer specific configs
    n_heads: int = 6
    n_cond_layers: int = 0
    attn_dropout: float = 0.1
    # UNet specific configs
    model_dim: int = 256
    kernel_size: int = 5
    cond_predict_scale: bool = True
    obs_as_global_cond: bool = True
    dim_mult: list[int] | None = None
    norm_type: str = "groupnorm"
    attention: bool = False
    # RNN specific configs
    rnn_type: str = "LSTM"  # "LSTM" or "GRU"
    max_freq: float = 100.0
    # VLA specific configs
    use_state_in_decoder: bool = True  # Pass state tokens to decoder cross-attention
    # Render-augmented network configs
    use_render_augmentation: bool = False  # Wrap base network with render-augmented two-stage MIP
    # Architecture variant switch
    # "flow_action" (default) = Config B: FlowMap action + optional IntentPredictor MLP
    # "flow_intent"           = Config A: FlowMap intent + deterministic MLPActionDecoder
    arch_variant: str = "flow_action"


@dataclass
class TaskConfig:
    env_name: str = "lift"
    obs_type: str = "state"
    env_type: str = "ph"
    abs_action: bool = True
    # Dataset configuration - either HuggingFace or local path
    dataset_repo: str | None = (
        None  # HuggingFace repository ID (e.g., "ChaoyiPan/mip-dataset")
    )
    dataset_filename: str | None = (
        None  # Path within the repository (e.g., "robomimic/lift/ph/image.hdf5")
    )
    dataset_path: str | None = (
        None  # Local path (deprecated, use dataset_repo/dataset_filename)
    )
    dataset_paths: list[str] | None = None  # Multiple local paths for multi-task suite training
    bddl_file: str | None = None  # BDDL task file for LIBERO environments
    bddl_files: list[str] | None = None  # Multiple BDDL files for suite-level multi-task eval
    arch_variant: str = "flow_action"  # Architecture variant: "flow_action" or "flow_intent"
    max_episode_steps: int = 400
    obs_keys: list[str] = field(
        default_factory=lambda: [
            "object",
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ]
    )
    obs_dim: int = -1
    act_dim: int = 10
    obs_steps: int = 2
    act_steps: int = 8
    horizon: int = 10  # Prediction horizon (typically obs_steps + act_steps)
    num_envs: int = 1
    save_video: bool = False
    shape_meta: dict = field(default_factory=dict)
    render_obs_key: str = "agentview_image"
    val_dataset_percentage: float = 0.0
    # Image observation settings
    rgb_model: str = "resnet18"
    resize_shape: list[int] | None = None
    crop_shape: list[int] | None = None
    random_crop: bool = True
    use_group_norm: bool = True
    use_seq: bool = True
    # Renderer settings
    use_image_renderer: bool = False  # Enable image renderer (injected into network)
    renderer_type: str = "physics_step"  # "physics_step" (full sim) or "agent_only" (teleport agent, no physics)
    max_renders_per_batch: int | None = None  # None = render all samples; int = cap true renders per batch
    render_size: int = 96  # Size of rendered images
    render_camera_name: str = "sideview"  # Camera name for robomimic rendering
    load_sim_states: bool = False  # Load MuJoCo sim states from HDF5 for rendering
    # Intent conditioning settings
    intent_conditioning: bool = False  # Whether to add intent (future eef pose) as extra conditioning
    intent_dim: int = 7  # Dimension of intent vector (3 pos + 4 quat = 7)
    intent_predictor: bool = False  # Use a learned MLP to predict intent at inference (vs. CV proxy)
    intent_horizon: int = 8  # High-level lookahead N: intent = mean(eef[t+1..t+N]). Independent of act_steps.
    intent_training_mode: str = "independent"  # "independent": flow policy trains on GT intent;
    # "joint": flow policy trains on HL predictor output (detached) — closes train/inference gap.
    intent_type: str = "mean"  # "mean": mean(eef[t+1..t+N]); "final": eef[t+N] (endpoint only);
    # "encoded_mean": per-step MLP encoding then mean-pool → intent_emb_dim-D vector
    intent_emb_dim: int = 64  # output dim of IntentEncoder (only used when intent_type=="encoded_mean")
    intent_encoder_warmup_steps: int = 0  # freeze IntentEncoder after this many steps (0 = never freeze)
    intent_keys: list[str] = field(
        default_factory=lambda: ["robot0_eef_pos", "robot0_eef_quat"]
    )  # obs keys to use for intent extraction; defaults to eef pos+quat for robomimic tasks
    intent_indices: list[int] = field(
        default_factory=lambda: [0, 1]
    )  # obs indices to use for intent (used by PushT and other non-robomimic tasks)


@dataclass
class Config:
    optimization: OptimizationConfig
    network: NetworkConfig
    task: TaskConfig
    log: LogConfig
    mode: str = "train"  # "train" or "eval"
