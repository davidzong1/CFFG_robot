# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn

from typing import TYPE_CHECKING
from class_free_guide.pineline.flow_matching.flow_cfg import FlowControlCfg
from class_free_guide.pineline.flow_matching.flow_control import FlowControlDIT
from class_free_guide.network.base.mlp import MLP
from class_free_guide.network.utils.utils import resolve_nn_activation

if TYPE_CHECKING:
    from class_free_guide.pineline.rl.rsl_rl.flow_ppo.config import FpoRslRlPpoActorCriticCfg

try:
    from transformers.feature_extraction_utils import BatchFeature
except ImportError:
    from collections import UserDict as BatchFeature
from typing import Optional


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_critic_obs: int,
        cfg: FpoRslRlPpoActorCriticCfg,
    ):
        super().__init__()
        activation = resolve_nn_activation(cfg.activation)
        self.info: Optional[BatchFeature] = None
        # Policy Network: Actor
        critic_hidden_dims = cfg.critic_hidden_dims
        mlp_input_dim_c = num_critic_obs
        self.use_dit = False
        self.flow_matching_cfg = cfg.flow_matching_cfg
        assert self.flow_matching_cfg is not None, "Flow matching configuration must be provided for actor initialization"
        try:
            self.flow_matching_cfg.num_sample_steps = 1
            self.flow_matching_cfg.num_denoise_steps = 0
            self.flow_matching_cfg.noise_inference = "none"
        except AttributeError:
            pass
        self.actor = FlowControlDIT(cfg=self.flow_matching_cfg)
        self.use_dit = True
        # Policy Network: Critic
        self.critic = MLP(mlp_input_dim_c, 1, critic_hidden_dims, activation)
        print("===================================================")
        print(f"Actor MLP: {self.actor}")
        print("===================================================")
        print(f"Critic MLP: {self.critic}")
        print("===================================================")

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    #########################################################################
    ##                              Interface                              ##
    #########################################################################
    def act(self, observations: torch.Tensor, **kwargs):
        action = kwargs.get("action", None)
        state_hidden = self.actor.state_vae(observations)
        action_hidden = self.actor.action_vae(action)
        self.info: BatchFeature = self.actor.flow_forward(state_hidden, action_hidden)
        return self.info

    def act_inference(self, observations: torch.Tensor, **kwargs):
        action = kwargs.get("action", None)
        state_hidden = self.actor.state_vae(observations)
        action_hidden = self.actor.action_vae(action)
        return self.actor.flow_forward(state_hidden, action_hidden)

    def evaluate(self, critic_observations: torch.Tensor, **kwargs):
        value = self.critic(critic_observations)
        return value

    def load_state_dict(self, state_dict, strict=True, assign=False):
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    #########################################################################
    ##                                 Loss                                ##
    #########################################################################

    def cal_cfm_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        eps: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        n_samples: Optional[int] = None,
    ):
        """Per-sample CFM loss following FPO++ (ref: isaaclab_fpo ActorCritic.get_cfm_loss).

        The FPO ratio is built as ``exp(old_cfm_loss - new_cfm_loss)`` over the SAME
        ``(eps, t)`` pairs. Therefore at rollout time call this with ``eps=t=None`` (fresh
        Monte-Carlo draws) and persist the returned ``(eps, t)`` together with the loss in
        the storage; at update time pass those stored ``eps, t`` back in so the new policy
        is evaluated on identical noise/time samples.

        Args:
            observations: ``(B, state_dim)`` raw observations to be VAE-encoded into the
                flow's hidden token space.
            actions: ``(B, action_dim)`` clean action targets, also VAE-encoded.
            eps: optional MC noise of shape ``(B, N_mc, 2*total_token, hidden_dim)``.
                When ``None`` it is drawn from ``N(0, I)``.
            t: optional flow timesteps of shape ``(B, N_mc, 1, 1)`` in ``(0, 1)``.
                When ``None`` it is sampled via the beta inverse-CDF used by FPO++.
            n_samples: number of MC pairs ``N_mc``. Ignored when ``eps`` / ``t`` are
                supplied; otherwise defaults to ``cfg.num_sample_steps``.

        Returns:
            loss: ``(B, N_mc)`` per-sample squared error (no reduction over ``N_mc``).
            x1_pred: ``(B, N_mc, 2*total_token, hidden_dim)`` predicted noise-end tokens.
            x0_pred: same shape, predicted data-end tokens.
            eps: the ``(B, N_mc, ...)`` noise actually used (for storage / reuse).
            t: the ``(B, N_mc, 1, 1)`` timesteps actually used.
        """
        # 1. Encode clean (state, action) into the joint hidden-token sequence the flow
        #    operates on. Mirrors flow_control._flow_forward_impl's input construction so
        #    train- and rollout-time spaces match.
        state_hidden = self.actor.state_vae(observations)
        action_hidden = self.actor.action_vae(actions)
        clean = torch.cat([state_hidden, action_hidden], dim=1)  # (B, 2*total_token, D_h)
        B, T, D = clean.shape
        device, dtype = clean.device, clean.dtype

        # 2. Resolve N_mc and consistency-check provided samples.
        if eps is not None and t is not None:
            assert eps.shape[0] == B and t.shape[0] == B, "eps/t batch mismatch"
            assert eps.shape[1] == t.shape[1], "eps and t must share N_mc"
            n_mc = eps.shape[1]
        else:
            n_mc = n_samples if n_samples is not None else self.flow_matching_cfg.num_sample_steps

        # 3. Sample (eps, t) when not supplied. Beta inverse-CDF biases mass toward small t
        #    (i.e. near the clean data end), matching FPO++ fpo.py:182.
        if eps is None:
            eps = torch.randn(B, n_mc, T, D, device=device, dtype=dtype)
        else:
            assert eps.shape == (B, n_mc, T, D), f"eps shape mismatch: {tuple(eps.shape)} vs {(B, n_mc, T, D)}"

        if t is None:
            beta = self.flow_matching_cfg.cfm_loss_beta
            u = torch.rand(B, n_mc, 1, 1, device=device, dtype=dtype)
            t = 0.005 + 0.99 * (1.0 - (1.0 - u) ** (1.0 / beta))
        else:
            assert t.shape == (B, n_mc, 1, 1), f"t shape mismatch: {tuple(t.shape)} vs {(B, n_mc, 1, 1)}"

        # 4. Linear interpolation in the FlowControlDIT time convention (t=0 -> data,
        #    t=1 -> noise; see flow_control.py:194). Hence the target velocity below is
        #    eps - clean, NOT clean - eps.
        clean_expanded = clean.unsqueeze(1).expand(B, n_mc, T, D)
        x_t = t * eps + (1.0 - t) * clean_expanded  # (B, N_mc, T, D)

        # 5. Fuse (B, N_mc) into a single batch axis for one DiT forward pass.
        x_t_flat = x_t.reshape(B * n_mc, T, D)
        t_flat = t.reshape(B * n_mc, 1, 1)  # DenoiserTransformer expects (B*, 1, 1)
        velocity_flat = self.actor.model(x_t_flat, t_flat)
        velocity_pred = velocity_flat.reshape(B, n_mc, T, D)

        # 6. u-mode target velocity and per-sample squared error. Reduce over token and
        #    hidden dims only -- DO NOT mean over N_mc, since FPO++ per-sample clipping
        #    (ref FPO++ paper Eq. 10) needs one ratio per (eps_i, t_i) pair.
        target_velocity = eps - clean_expanded
        loss = (velocity_pred - target_velocity).pow(2).mean(dim=(-2, -1))  # (B, N_mc)

        # 7. Endpoint predictions, useful for diagnostics / KL-style auxiliary losses
        #    (mirrors FPO++ get_cfm_loss return).
        x0_pred = x_t - t * velocity_pred
        x1_pred = x0_pred + velocity_pred

        return loss, x1_pred, x0_pred

    def cal_vae_loss(self, state: torch.Tensor, action: torch.Tensor):
        return self.actor.cal_state_action_encoder_loss(state, action)

    def cal_consistency_loss(self, data: dict):
        return self.actor.cal_consistency_loss(data)

    def get_act_state_dict(self):
        return self.actor.state_dict()

    def get_critic_state_dict(self):
        return self.critic.state_dict()

    def load_act_state_dict(self, state_dict, strict=True, assign=False):
        return self.actor.load_state_dict(state_dict, strict=strict, assign=assign), self.flow_matching_cfg

    def load_critic_state_dict(self, state_dict, strict=True, assign=False):
        return self.critic.load_state_dict(state_dict, strict=strict, assign=assign)
