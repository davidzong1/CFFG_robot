from dataclasses import dataclass, field
from .ac_config import FpoRslRlPpoActorCriticCfg
from .ppo_config import FpoRslRlPpoAlgorithmCfg
from dataclasses import MISSING
from typing import Literal


@dataclass(kw_only=True)
class FpoRslRlOnPolicyRunnerCfg:
    """Configuration of the runner for on-policy algorithms."""

    seed: int = 42
    """The seed for the experiment. Default is 42."""

    device: str = "cuda:0"
    """The device for the rl-agent. Default is cuda:0."""

    num_steps_per_env: int = 24
    """The number of steps per environment per update. Default is 24."""

    max_iterations: int = 1500
    """The maximum number of iterations. Default is 1500.

    Quadrupeds (Go2, A1, Spot, Anymal-B/C/D) use 1500. Humanoids (H1, G1) use 2000.
    """

    empirical_normalization: bool = True
    """Whether to use empirical normalization. Default is True."""

    randomize_reset_episode_progress: float = 0.0
    """Randomize episode progress on reset to prevent synchronization. Default is 0.0 (disabled).

    When > 0, environments that reset will have their episode_length_buf randomized to a value
    between 0 and randomize_reset_episode_progress * max_episode_length. For example, 0.25 means
    episodes will start at a random point between 0-25% completion.
    """

    policy: FpoRslRlPpoActorCriticCfg = MISSING
    """The policy configuration."""

    algorithm: FpoRslRlPpoAlgorithmCfg = MISSING
    """The algorithm configuration."""

    clip_actions: float | None = 2.0
    """The clipping value for actions. If ``None``, then no clipping is done.
    Default is 2.0.

    .. note::
        This clipping is performed inside the :class:`FpoRslRlVecEnvWrapper` wrapper.
    """

    save_interval: int = 50
    """The number of iterations between saves. Default is 50."""

    experiment_name: str = MISSING
    """The experiment name."""

    run_name: str = ""
    """The run name. Default is empty string.

    The name of the run directory is typically the time-stamp at execution. If the run name is not empty,
    then it is appended to the run directory's name, i.e. the logging directory's name will become
    ``{time-stamp}_{run_name}``.
    """

    logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
    """The logger to use. Default is tensorboard."""

    neptune_project: str = "mjlab"
    """The neptune project name. Default is "mjlab"."""

    wandb_project: str = "mjlab"
    """The wandb project name. Default is "mjlab"."""

    # Evaluation configuration
    eval_episodes: int = 10
    """Number of episodes to run per evaluation mode. Default is 10."""

    flow_eval_modes: list[str] = field(default_factory=lambda: ["zero", "random"])
    """Evaluation modes for flow matching deterministic sampling. Default is ["zero", "random"].

    Available modes:
    - "zero": Use zeros for initial noise
    - "fixed_seed": Use fixed random seed for reproducible noise
    - "random": Use random noise (different each time)
    """

    flow_eval_fixed_seed: int = 12345
    """Random seed for fixed_seed evaluation mode. Default is 12345."""

    enable_post_training_eval: bool = True
    """Whether to evaluate all checkpoints after training completes. Default is True.

    When enabled, automatically evaluates all saved checkpoints using the same eval configuration
    (flow_eval_modes, eval_episodes) after training finishes. Results are logged to WandB with
    a custom 'eval_iteration' step metric for comparison with training metrics.
    """

    post_eval_checkpoint_interval: int = 1
    """Evaluate every Nth checkpoint during post-training evaluation. Default is 1 (all checkpoints).

    Set to 2 to evaluate every other checkpoint, 3 for every third, etc. This can significantly
    reduce post-training evaluation time for experiments with many checkpoints.
    """

    resume: bool = False
    """Whether to resume. Default is False."""

    load_run: str = ".*"
    """The run directory to load. Default is ".*" (all).

    If regex expression, the latest (alphabetical order) matching run will be loaded.
    """

    load_checkpoint: str = "model_.*.pt"
    """The checkpoint file to load. Default is ``"model_.*.pt"`` (all).

    If regex expression, the latest (alphabetical order) matching file will be loaded.
    """

    custom_model_param_save: bool = False
    """Whether to save custom model parameters. Default is False."""

    custom_model_param_method: dict | None = None
    """The method to save custom model parameters. Default is None."""
