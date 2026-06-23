from __future__ import annotations

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from dataclasses import asdict
from typing import TYPE_CHECKING
from muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam
from class_free_guide.pineline.rl.rsl_rl.flow_ppo.module.actor_critic_dit import ActorCritic
from isaaclab_fpo.modules.ema import ExponentialMovingAverage
from class_free_guide.pineline.rl.rsl_rl.flow_ppo.storage.rollout_storage_fpo import RolloutStorage

if TYPE_CHECKING:
    from class_free_guide.pineline.rl.rsl_rl.flow_ppo.config import FpoRslRlPpoAlgorithmCfg
try:
    from transformers.feature_extraction_utils import BatchFeature
except ImportError:
    from collections import UserDict as BatchFeature


def clamp_ste(x, min=None, max=None):
    clamped = x.clamp(min=min, max=max)
    # forward uses clamped; backward uses identity grad wrt x
    return x + (clamped - x).detach()


class FlowPPO:
    """PPO algorithm with support for RND and symmetry-based data augmentation."""

    actor: torch.nn.Module
    """The actor model."""

    critic: torch.nn.Module
    """The critic model."""

    def __init__(
        self,
        policy: ActorCritic,
        cfg: FpoRslRlPpoAlgorithmCfg,
        device="cpu",
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        # Multi-GPU parameters
        if self.is_multi_gpu:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # PPO components
        self.policy: ActorCritic = policy.to(self.device)
        self.cfg = cfg
        self.actor_optimizer: SingleDeviceMuonWithAuxAdam | MuonWithAuxAdam | None = None
        # Opitmizer configuration
        self.critic_optimizer = optim.Adam(
            self.policy.critic.parameters(),
            lr=cfg.critic_learning_rate,
            eps=1e-5,
            weight_decay=0.0,
        )
        self.actor_param_groups = self._build_flow_dit_actor_param_groups()
        if self.is_multi_gpu and self.gpu_world_size > 1:
            self._validate_distributed_for_muon()
            self.actor_optimizer = MuonWithAuxAdam(self.actor_param_groups)
        else:
            self.actor_optimizer = SingleDeviceMuonWithAuxAdam(self.actor_param_groups)

        # Create rollout storage
        self.storage: RolloutStorage = None  # type: ignore
        self.transition = RolloutStorage.Transition()
        print("===================================================")
        print("FpoRslRlPpoAlgorithmCfg is :\n ", asdict(self.cfg))
        print("===================================================")

    def init_storage(
        self,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        actions_shape,
    ):
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            actions_shape,
            self.device,
            self.cfg.n_samples_per_action,
        )

    def _validate_distributed_for_muon(self) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                "MuonWithAuxAdam uses torch.distributed.get_rank()/get_world_size() internally. "
                "Initialize torch.distributed before constructing FlowPPO, or use "
                "SingleDeviceMuonWithAuxAdam for single-process training."
            )
        dist_rank = dist.get_rank()
        dist_world_size = dist.get_world_size()
        if dist_rank != self.gpu_global_rank or dist_world_size != self.gpu_world_size:
            raise RuntimeError(
                "multi_gpu_cfg does not match torch.distributed state: "
                f"cfg rank/world_size=({self.gpu_global_rank}, {self.gpu_world_size}), "
                f"dist rank/world_size=({dist_rank}, {dist_world_size})."
            )

    def _build_flow_dit_actor_param_groups(self):
        muon_params = []
        adam_decay = []
        adam_no_decay = []
        norm_types = (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)
        seen = set()
        for module_name, module in self.policy.actor.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if not param.requires_grad:
                    continue
                if id(param) in seen:
                    continue
                seen.add(id(param))
                full_name = f"{module_name}.{param_name}" if module_name else param_name
                is_bias = param_name.endswith("bias")
                is_norm = isinstance(module, norm_types)
                # FlowControlDIT: actor.model is the DiT backbone.
                # Muon is most suitable for 2D+ matrix weights.
                if full_name.startswith("model.") and param.ndim >= 2 and not is_bias and not is_norm:
                    muon_params.append(param)
                elif param.ndim <= 1 or is_bias or is_norm:
                    adam_no_decay.append(param)
                else:
                    # state_vae/action_vae MLP weights: safer to keep on AdamW.
                    adam_decay.append(param)
        return [
            group
            for group in [
                {
                    "params": muon_params,
                    "use_muon": True,
                    "lr": self.cfg.learning_rate,
                    "weight_decay": self.cfg.weight_decay,
                },
                {
                    "params": adam_decay,
                    "use_muon": False,
                    "lr": self.cfg.learning_rate,
                    "betas": self.cfg.adam_betas,
                    "eps": 1e-5,
                    "weight_decay": self.cfg.weight_decay,
                },
                {
                    "params": adam_no_decay,
                    "use_muon": False,
                    "lr": self.cfg.learning_rate,
                    "betas": self.cfg.adam_betas,
                    "eps": 1e-5,
                    "weight_decay": 0.0,
                },
            ]
            if len(group["params"]) > 0
        ]

    def act(self, obs, critic_obs):
        # Shape assertions
        assert len(obs.shape) == 2, f"Expected obs shape [num_envs, obs_dim], got {obs.shape}"
        assert len(critic_obs.shape) == 2, f"Expected critic_obs shape [num_envs, critic_obs_dim], got {critic_obs.shape}"
        assert obs.shape[0] == self.storage.num_envs, f"Expected {self.storage.num_envs} envs, got {obs.shape[0]}"

        # compute the actions and values
        self.info: BatchFeature = self.policy.act(obs, action=self.transition.actions)
        self.transition.actions = self.policy.act_inference(obs)["actions"].detach()
        self.transition.values = self.policy.evaluate(critic_obs).detach()

        # Shape assertions for outputs
        assert self.transition.actions.shape == (
            self.storage.num_envs,
            self.policy.num_actions,
        ), f"Expected actions shape [{self.storage.num_envs}, {self.policy.num_actions}], got {self.transition.actions.shape}"
        assert self.transition.values.shape == (
            self.storage.num_envs,
            1,
        ), f"Expected values shape [{self.storage.num_envs}, 1], got {self.transition.values.shape}"
