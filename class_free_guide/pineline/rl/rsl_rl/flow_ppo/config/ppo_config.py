from dataclasses import MISSING
from typing import Literal

from dataclasses import dataclass, field


@dataclass
class FpoRslRlPpoAlgorithmCfg:  # Keeping name for backwards compatibility
    """Configuration for the FPO (Flow Policy Optimization) algorithm."""

    class_name: str = "FPO"
    """The algorithm class name. Default is FPO (Flow Policy Optimization)."""

    num_learning_epochs: int = 16
    """The number of learning epochs per update. Default is 16.

    Quadruped locomotion (Go2, A1, Anymal-B/C/D) uses 16. Humanoid locomotion (H1, G1)
    and Spot use 32. Override in per-robot configs as needed.
    """

    num_mini_batches: int = 4
    """The number of mini-batches per update. Default is 4."""

    critic_learning_rate: float = 1e-4
    """The learning rate for the critic. Default is 1e-4."""

    learning_rate: float = 1e-5
    """The learning rate for the actor. Default is 1e-4."""

    weight_decay: float = 1e-4
    """Weight decay coefficient for AdamW optimizer. Default is 1e-4."""

    adam_betas: tuple[float, float] = (0.9, 0.999)
    """Beta parameters (beta1, beta2) for Adam/AdamW optimizer. Default is (0.9, 0.999).

    - beta1: Exponential decay rate for first moment estimates (momentum)
    - beta2: Exponential decay rate for second moment estimates (RMSprop-like)
    """

    schedule: str = "fixed"
    """The learning rate schedule. Default is 'fixed'.

    - 'fixed': Constant learning rate throughout training
    - 'adaptive': Adjusts learning rate based on KL divergence (requires desired_kl > 0)

    The canonical locomotion baseline (Nov 2025) uses 'fixed'.
    """

    gamma: float = 0.99
    """The discount factor. Default is 0.99."""

    lam: float = 0.95
    """The lambda parameter for Generalized Advantage Estimation (GAE). Default is 0.95."""

    knn_entropy_coef: float = 0.0
    """Coefficient for kNN entropy bonus in the policy loss."""

    knn_entropy_k: int = 1
    """Number of nearest neighbors for kNN entropy bonus in the policy loss. Default is 1.

    Separate from knn_k which is used for the entropy regularization term.
    This parameter controls the k used when computing an additional entropy-based
    exploration bonus. k=1 gives the sharpest entropy estimate and empirically
    outperforms higher k values for locomotion tasks.
    """

    desired_kl: float = 1e-4
    """The desired KL divergence. Default is 1e-4.

    When schedule='adaptive', the learning rate is adjusted to keep KL divergence
    near this target. Ignored when schedule='fixed' (the default).
    """

    max_grad_norm: float = 1.0
    """The maximum gradient norm. Default is 1.0."""

    value_loss_coef: float = 1.0
    """The coefficient for the value loss. Default is 1.0.

    Spot uses 0.5 (override in per-robot config). All other locomotion robots use 1.0.
    """

    use_clipped_value_loss: bool = False
    """Whether to use clipped value loss. Default is False.

    PPO typically clips the value function loss, but with FPO's tighter policy clipping,
    value clipping adds instability.
    """

    clip_param: float = 0.05
    """The clipping parameter for the policy. Default is 0.05.

    FPO needs much tighter clipping than standard PPO (0.2). Locomotion uses 0.05.
    Also used as the SPO epsilon in 'spo' and 'aspo' modes.
    """

    trust_region_mode: Literal["ppo", "spo", "aspo"] = "aspo"
    """Trust region method to use. Default is 'aspo'.

    - 'ppo': Standard PPO with hard clipping constraint
    - 'spo': Structured Policy Optimization with quadratic penalty
    - 'aspo': Asymmetric SPO - uses PPO for positive advantages, SPO for negative advantages

    SPO uses a smoother trust region constraint:
    policy_loss = -mean(ratio * advantage - |advantage| / (2*epsilon) * (ratio - 1)^2)

    This provides more gradual policy updates compared to PPO's hard clipping.
    """

    normalize_advantage: bool = True
    """Whether to normalize advantages at all. Default is True.

    If False, advantages are used as-is without any normalization.
    This can be useful when the agent is near optimal and normalization
    artificially amplifies small differences in returns.
    """

    normalize_advantage_per_mini_batch: bool = False
    """Whether to normalize the advantage per mini-batch. Default is False.

    If True, the advantage is normalized over the mini-batches only.
    Otherwise, the advantage is normalized over the entire collected trajectories.
    Note: This only applies if normalize_advantage is True.
    """

    advantage_clamp: tuple[float, float] = (100.0, 100.0)
    """Symmetric clamp bounds for advantages as (positive_max, negative_max).

    Clamps advantages to [-negative_max, positive_max] before using them in the policy loss.
    Prevents large advantages from causing unstable updates."""

    n_samples_per_action: int = 16
    """Number of samples per action for CFM loss computation. Default is 16.

    The canonical locomotion baseline (Nov 2025, run 03yjn5lj) uses 16.
    Gains plateau beyond 16.
    """

    cfm_diff_clamp_max: float = 10.0
    """Upper bound for CFM loss difference clamping. Default is 10.0.

    The loss difference (old_cfm_loss - new_cfm_loss) is clamped to this upper bound
    using straight-through estimator (STE) before exp().
    """

    cfm_loss_clamp: float = 20.0
    """Maximum value to clamp CFM losses (both old and current). Default is -1.0 (disabled).

    When > 0, clamps both the old (stored) and current (recomputed) CFM loss values
    to this upper bound. This prevents extremely large losses from producing extreme ratios
    that destabilize training. Applied symmetrically to both old and current CFM losses.
    """

    cfm_loss_clamp_negative_advantages: bool = True
    """Clamp current CFM loss when advantage is negative. Default is True.

    When enabled, clamps the current (recomputed) CFM loss to cfm_loss_clamp_negative_advantages_max
    for transitions where the advantage is negative (bad actions). This prevents the policy from
    being destabilized by extreme ratios when aggressively avoiding bad actions.
    Critical for training stability with 32+ learning epochs (H1, G1, Spot).
    """

    cfm_loss_clamp_negative_advantages_max: float = 20.0
    """Maximum CFM loss value for negative advantage clamping.

    Only used when cfm_loss_clamp_negative_advantages is True. The current CFM loss is clamped
    to this value for transitions with negative advantages.
    """

    storage_action_noise_std: float = 0.0
    """Standard deviation of Gaussian noise added to stored actions. Default is 0.0 (no noise).

    This noise is added to actions before storing them in the rollout buffer, affecting
    the CFM loss computation in the PPO ratio. Acts as implicit entropy regularization
    by forcing the policy to be robust to action perturbations. Unlike action_perturb_std
    in the actor, this noise affects the policy gradient computation.

    Typical values: 0.01-0.05 depending on action scale and desired regularization strength.
    """

    ema_decay: float = 0.95
    """EMA decay rate for exponential moving average of flow model weights. Default is 0.95.

    When > 0, maintains a smoothed copy of the actor (flow model) parameters that is updated
    after each PPO update. The EMA weights are saved in checkpoints and typically produce better
    samples than the currently-being-trained weights.

    The canonical locomotion baseline (Nov 2025, run 03yjn5lj) uses 0.95.

    Recommended values:
    - 0.0: Disabled (no EMA)
    - 0.95: Fast adaptation, suitable for short training runs (~1500-2000 steps)
    - 0.99: Slower adaptation, suitable for longer training runs
    - 0.999: Very slow adaptation, only for very long training runs

    The effective averaging window is approximately 1/(1-decay) updates.
    """

    ema_warmup_steps: int = 500
    """Number of PPO updates before starting EMA. Default is 500.

    Prevents early noisy weights from contaminating the EMA by waiting for the policy
    to stabilize before starting exponential averaging. Only applies when ema_decay > 0.
    Set to 0 to start EMA immediately from the first update.
    """
