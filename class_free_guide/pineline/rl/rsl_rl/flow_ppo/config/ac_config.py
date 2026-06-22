# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from dataclasses import dataclass, field
from class_free_guide.pineline.flow_matching.flow_cfg import FlowControlCfg
from typing import Optional

#########################
# Policy configurations #
#########################


@dataclass
class FpoRslRlPpoActorCriticCfg:
    """Configuration for the PPO actor-critic networks."""

    class_name: str = "ActorCritic"
    """The policy class name. Default is ActorCritic."""

    extern_model: bool = False
    """Whether to use an external model for the policy."""

    flow_matching_cfg: Optional[FlowControlCfg] = None
    """Configuration for the flow matching component of the policy."""

    init_noise_std: float = 1.0
    """The initial noise standard deviation for the policy."""

    actor_hidden_dims: Optional[list[int]] = None
    """The hidden dimensions of the actor network."""

    critic_hidden_dims: Optional[list[int]] = None
    """The hidden dimensions of the critic network. Required for non-external models."""

    activation: str = "elu"
    """The activation function for the actor and critic networks. Default is elu."""

    actor_scale: float = 1.0
    """Scaling factor applied to actor network output."""

    actor_mlp_output_scale: float = 1.0
    """Scaling factor applied to actor MLP output."""

    actor_final_layer_weight_scale: Optional[float] = None
    """Scaling factor applied to initial weights of actor's final layer. Default is None (no scaling).

    When set, multiplies the weights of the actor's final linear layer by this value during
    initialization. Can help stabilize training by reducing initial action magnitudes.
    """

    timestep_embed_dim: int = 8
    """Dimension of the timestep embedding for the flow network. Default is 8.

    When > 0, adds a learned embedding of the flow timestep t to the actor network input.
    This allows the network to condition its output on the current denoising step.
    All canonical locomotion baselines (Nov 2025) used timestep_embed_dim=8.
    Set to 0 to disable timestep conditioning.
    """

    training_sampling_steps: Optional[int] = None
    """Override sampling_steps for training CFM loss computation. Default is None (use sampling_steps).

    When set, uses a different number of discretization steps for computing CFM training loss
    than for inference. This allows using fewer steps during training for efficiency while
    using more steps during inference for quality.
    """

    cfm_loss_t_inverse_cdf_beta: float = 1.0
    """Beta parameter for Beta(1, beta) distribution used in timestep sampling.

    Controls the distribution of timesteps t sampled during CFM training:
    - beta = 1.0: Uniform sampling (default)
    - beta > 1.0: Favors sampling timesteps near t=0 (closer to actions)
      - e.g., beta = 2.0 moderately emphasizes action refinement
      - e.g., beta = 3.0 matches DDPM schedule (standard in flow matching)
      - e.g., beta = 4.0 strongly emphasizes action reconstruction
    - beta < 1.0: Favors sampling timesteps near t=1 (closer to noise)
      - e.g., beta = 0.5 moderately emphasizes exploration phase
      - e.g., beta = 0.25 strongly emphasizes early flow matching

    Math: Given uniform u ~ U(0,1), timesteps are sampled as:
    t = 0.005 + 0.99 * (1 - (1-u)^(1/beta))

    This implements the inverse CDF of Beta(1, beta) distribution scaled to [0.005, 0.995].
    """

    sampling_steps: int = 64
    """Number of sampling steps for flow matching inference. Default is 64."""

    cfm_loss_reduction: Literal["mean", "sum", "sqrt"] = "sqrt"
    """Reduction method for CFM loss across action dimensions. Default is "sqrt".

    - "mean": Average loss across action dimensions (divides by action_dim)
    - "sum": Sum loss across action dimensions (no division)
    - "sqrt": Variance-preserving reduction (divides by sqrt(action_dim))

    The "sqrt" option provides variance-preserving scaling that maintains similar
    gradient magnitudes across robots with different action dimensions.
    Empirically outperforms both "mean" and "sum" across all tasks.
    """

    action_perturb_std: float = 0.02
    """Standard deviation of Gaussian noise added to actions during training.
    Perturbs actions with random noise, which can be interpreted as an entropy
    regularizer."""
