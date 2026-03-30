import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional
from .flow_cfg import FlowInfo, FlowMatchingCfg, FlowNoiseType
from class_free_guide.network.nosise_gen.log_noise_nn import LogNoiseNN
from class_free_guide.algorithm.log_prob import get_logprob_norm

try:
    from transformers.feature_extraction_utils import BatchFeature
except ImportError:
    from collections import UserDict as BatchFeature


class FlowMatcherBase(nn.Module):
    def __init__(self, model, cfg: FlowMatchingCfg):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.model.to(self.cfg.device)
        assert self.cfg.num_sample_steps > 0, "num_sample_steps must be greater than 0"
        assert self.cfg.real_denoise_step > 0, "real_denoise_step must be greater than 0"
        assert self.cfg.num_sample_steps >= self.cfg.real_denoise_step, "num_sample_steps must be greater than or equal to real_denoise_step"
        # time t+1 use in delta
        self.timesteps = torch.linspace(0, 1, self.cfg.num_sample_steps + 1, device=self.cfg.device, dtype=torch.float32)
        self.denoise_flag = torch.zeros(self.cfg.num_sample_steps, dtype=torch.bool, device=self.cfg.device)
        for i in range(self.cfg.real_denoise_step):
            self.denoise_flag[i] = True
        # noise type
        if self.cfg.noise_inference == FlowNoiseType.REINFLOW:
            self.log_noise_nn = LogNoiseNN(self.cfg.hidden_dim, self.cfg.output_dim, self.cfg.noise_hidden_dim, self.cfg.noise_activation)
            self.log_noise_nn.to(self.cfg.device)
        self.to(self.cfg.device)

    def forward(
        self,
        t_idx: torch.Tensor,
        x_t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward function for flow matching sampling, used in evaluation and inference.
        """
        t_input = self.timesteps[t_idx]
        delta = self.timesteps[t_idx + 1] - t_input
        t_input = t_input[:, None, None].expand_as(x_t)
        delta = delta[:, None, None].expand_as(x_t)
        v_t = self.model(x_t, condition)
        x0_pred = x_t - v_t * t_input
        x1_pred = x_t + v_t * (1 - t_input)
        x0_weight = 1 - (t_input + delta)
        x1_weight = t_input + delta
        x_t_next = x0_pred * x0_weight + x1_pred * x1_weight
        return x_t_next

    def sample_noise_action(
        self,
        x_t: torch.Tensor,
        condition: Optional[torch.Tensor],
        time_idx: int,
        inject_noise: bool,
    ):
        """
        Sample once noise std and action mean
        Args:
            condition: [Token_cond,hidden_dim] condition embedding for noise inference, can be None if not used.
            x_t: [B, 1, action_dim] noised action at time step t
            time_idx: int, index of the current time step in the sampling process
            inject_noise: bool, whether to inject noise into the sampled action
        return:
            x_t_mean: [B, 1, action_dim] the mean of the sampled action at time step t
            x_t_std: [B, 1, action_dim] the standard deviation of the sampled action at time step t, if inject_noise is False, this will
        """
        # Time step discretization
        t_input = self.timesteps[time_idx]
        delta = self.timesteps[time_idx + 1] - self.timesteps[time_idx]
        t_input = t_input * torch.ones(x_t.shape[0], 1, dtype=torch.float32, device=x_t.device)  # [B, 1]
        delta = delta * torch.ones(x_t.shape[0], 1, dtype=torch.float32, device=x_t.device)  # [B, 1]
        x_t_cat = torch.cat([x_t, t_input], dim=-1)
        # model forward
        v_t = self.model(x_t_cat, condition)
        x0_pred = x_t - v_t * t_input
        x1_pred = x_t + v_t * (1 - t_input)
        if not inject_noise:
            x0_weight = 1 - (t_input + delta)
            x1_weight = t_input + delta  # notice the plus here, it's different from openpi.
            x_t_std = torch.zeros_like(t_input)
        else:
            noise_dict = self.compute_state_noise(time_idx, v_t, self.cfg.noise_inference, t_input, delta)
            x_t_std = noise_dict["state_noise_std"]
            if self.cfg.noise_inference == FlowNoiseType.SDE:
                x0_weight = torch.ones_like(t_input) - (t_input + delta) - noise_dict["state_noise_std"] ** 2 * delta / (2 * (1 - t_input))
                x1_weight = t_input + delta
            elif self.cfg.noise_inference == FlowNoiseType.REINFLOW:
                x0_weight = 1 - (t_input + delta)
                x1_weight = t_input + delta
            else:
                raise ValueError(f"Unknown noise method: {self.cfg.noise_inference}")
        x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight
        return x_t_mean, x_t_std, t_input, v_t

    def train_floward(self, input: torch.Tensor, condition: Optional[torch.Tensor] = None) -> BatchFeature:
        x_t = input  # x0
        time_stamp = []
        chain = [x_t]
        chain_v = []
        log_probs = []
        for i in range(self.cfg.num_sample_steps):
            x_t_mean, x_t_std, t_input, v_t = self.sample_noise_action(
                x_t,
                condition,
                time_idx=i,
                inject_noise=self.denoise_flag[i],
            )
            # inject noise
            x_t = x_t_mean + torch.normal(mean=0.0, std=1.0, size=x_t_mean.shape, dtype=torch.float32, device=self.cfg.device) * x_t_std
            # log
            time_stamp.append(t_input)
            chain.append(x_t)
            chain_v.append(v_t)
            log_probs.append(get_logprob_norm(sample=x_t, mu=x_t_mean, sigma=x_t_std))
        x1 = x_t
        chain = torch.stack(chain, dim=1)  # [B, num_sample_steps+1, action_dim]
        chain_v = torch.stack(chain_v, dim=1)  # [B, num_sample_steps+1, action_dim]
        log_probs = torch.stack(log_probs, dim=1)
        time_stamp = torch.stack(time_stamp, dim=1)
        return BatchFeature(
            data={
                "x0": input,
                "x1_prev": x1,
                "chain": chain,
                "chain_v": chain_v,
                "log_probs": log_probs,
                "time_stamp": time_stamp,
            }
        )

    def compute_state_noise(
        self, num_step: int, model_output: torch.Tensor, noise_type: FlowNoiseType, t_input: torch.Tensor, delta: torch.Tensor
    ) -> dict:
        """
        Compute state noise standard deviation based on the noise type.
        Args:
            batch_size (int): The batch size.
            input (torch.Tensor): The input tensor for noise computation.
            noise_type (FlowNoiseType): The type of noise to compute ( FlowNoiseType.REINFLOW, FlowNoiseType.SDE).
            time_step (torch.Tensor): The time step tensor.
            delta (torch.Tensor): The time delta tensor.
        Returns:
            torch.Tensor: The computed state noise standard deviation.
        """
        dict_ = {}
        if noise_type == FlowNoiseType.REINFLOW:
            state_noise_std = self.log_noise_nn(model_output)  # (batch_size, action_dim)
            dict_["state_noise_std"] = state_noise_std
        elif noise_type == FlowNoiseType.SDE:
            alpha = self.cfg.alpha
            t = torch.where(t_input == 0, self.timesteps[num_step + 1], self.timesteps[num_step])
            sigma = alpha * torch.sqrt((1 - t) / t)
            # std=sqrt(dt)*sigma
            dict_["sigma"] = sigma
            dict_["state_noise_std"] = torch.sqrt(delta) * sigma
        else:
            raise ValueError(f"Unknown noise_type: {noise_type}")
        return dict_

    def compute_cfm_loss(
        self,
        data: BatchFeature,
        x_ref: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate the conditional flow matching loss.

        Args:
            state_head: [B, N, D_s] State head
            x1: [B, N, D_a] Final denoised action
            eps: [B, N, D_a] Sampled noise
            t: [B, N, 1] Time step

        Returns:
            loss: [B] Loss for each sample in the batch
        """
        t = data["time_stamp"]  # [B, num_sample_steps, 1]
        x_t = data["chain"][:, :-1, :]  # [B, num_sample_steps, D_a]
        v_ref = (x_ref[:, None, :].expand_as(x_t) - x_t) / (1 - t)  # [B, num_sample_steps, D_a]
        v_pred = data["chain_v"]  # [B, num_sample_steps, D_a]
        loss = F.mse_loss(v_pred, v_ref, reduction="none").mean(dim=-1)  # [B, num_sample_steps]
        loss_mean = loss.mean(dim=1).mean(dim=0)
        return loss_mean
