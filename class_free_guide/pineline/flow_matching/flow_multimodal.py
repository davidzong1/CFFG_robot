import time
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal
from class_free_guide.network.nosise_gen.log_noise_nn import LogNoiseNN
from enum import Enum
from typing import Tuple
from .flow_cfg import FlowCfg, FlowNoiseType
from class_free_guide.network.action_head.flow_matching_action_head import CategorySpecificMLP, MultiEmbodimentActionEncoder
from transformers import BatchFeature


class ConditionalFlowMatcher(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        cfg: FlowCfg = FlowCfg(),
    ):
        super().__init__()
        self.cfg = cfg
        self.pos_embedding = self.cfg.pos_embedding
        self.pos_embedding_dim = self.cfg.pos_embedding_dim if self.cfg.pos_embedding else 1
        self.cond_embedding = self.cfg.cond_embedding
        self.cond_embedding_out_dim = self.cfg.cond_embedding_out_dim
        self.init_noise = torch.randn(1, self.cfg.output_dim)
        self.model = base_model.copy()
        # noise inject selection
        if self.cfg.noise_inference == FlowNoiseType.REINFLOW:
            self.log_noise_nn = LogNoiseNN(self.cfg.hidden_dim, self.cfg.output_dim, [256, 128], self.cfg.noise_activation)
        # positional embedding
        if self.cfg.pos_embedding:
            self.pos_emb_net = nn.Embedding(num_embeddings=self.cfg.num_sample_step, embedding_dim=self.cfg.hidden_dim)
        else:
            self.pos_emb_net = nn.Identity()
        # state embedding
        self.state_encoder = CategorySpecificMLP(
            num_categories=self.cfg.num_categories,
            input_dim=self.cfg.max_state_dim,
            hidden_dim=self.cfg.cond_embedding_hidden_dim,
            output_dim=self.cfg.hidden_dim,
        )
        # condition embedding
        if self.cfg.cond_embedding:
            self.cond_emb_net = CategorySpecificMLP(
                num_categories=self.cfg.num_categories,
                input_dim=self.cfg.max_state_dim,
                hidden_dim=self.cfg.cond_embedding_hidden_dim,
                output_dim=self.cfg.hidden_dim,
            )
        else:
            assert (
                self.cfg.cond_embedding_out_dim == self.cfg.condition_input_dim
            ), "cond_embedding_out_dim must equal condition_input_dim when cond_embedding is False"
            self.cond_emb_net = nn.Identity()
        # action encoder
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=self.cfg.output_dim,
            hidden_size=self.cfg.hidden_dim,
            num_embodiments=self.cfg.num_categories,
            nn_type="mlp",
        )
        # action decoder
        self.action_decoder = CategorySpecificMLP(
            num_categories=self.cfg.num_categories,
            input_dim=self.cfg.hidden_dim,
            hidden_dim=self.cfg.hidden_dim,
            output_dim=self.cfg.output_dim,
        )
        # time scale
        self.timesteps = torch.linspace(0, 1, self.cfg.num_sample_step + 1, device=device, dtype=torch.float32)
        self.delta = 1.0 / self.cfg.num_sample_step
        # move to device
        self.device = device
        self.to(device)

    def sample_noise_action(
        self,
        condition: torch.Tensor,
        state_feature: torch.Tensor,
        x_t: torch.Tensor,
        idx: int,
        inject_noise: bool,
        embodiment_id: int,
    ):
        """
        采样动作一次
        Args:
            condition: [Token_cond,hidden_dim] 条件输入(多模态与上层指令等)
            state_feature: [B, 1,hidden_dim] 状态输入（机器人自身状态）
            inject_noise: 是否注入噪声
        """
        batch_size = state_feature.shape[0]
        # Time step discretization
        time_in_step = torch.ones(batch_size, dtype=torch.int32, device=self.device) * (self.timesteps[idx] * self.cfg.num_sample_step)
        action_features = self.action_encoder(x_t, embodiment_id)
        # token position embedding
        if self.cfg.pos_embedding:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=self.device)
            pos_embs = self.pos_emb_net(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs
        condition_emb = condition.unsqueeze(0).expand(batch_size, -1, self.cfg.hidden_dim)
        # Token cat [B,Ts+1+Tc,hidden_dim]
        model_input = torch.cat([state_feature, action_features, condition_emb], dim=1)
        model_output = self.model(hidden_states=model_input, encoder_hidden_states=condition_emb, timestep=time_in_step)
        model_output = model_output[:, -1:]
        v_t = self.action_decoder(model_output, embodiment_id)
        t_input = torch.ones_like(x_t, dtype=torch.float32, device=self.device) * self.timesteps[idx]
        delta = torch.ones_like(x_t, dtype=torch.float32, device=self.device) * self.delta
        x0_pred = x_t - v_t * t_input
        x1_pred = x_t + v_t * (1 - t_input)
        if not inject_noise:
            x0_weight = 1 - (t_input + delta)
            x1_weight = t_input + delta  # notice the plus here, it's different from openpi.
            x_t_std = torch.zeros_like(t_input)
        else:
            noise_dict = self.compute_state_noise(model_output.shape[0], model_output, self.cfg.noise_inference, delta)
            x_t_std = noise_dict["state_noise_std"]
            if self.rl_config.noise_method == "flow_sde":
                x0_weight = torch.ones_like(t_input) - (t_input + delta) - noise_dict["state_noise_std"] ** 2 * delta / (2 * (1 - t_input))
                x1_weight = t_input + delta
            elif self.rl_config.noise_method == "reinflow":
                x0_weight = 1 - (t_input + delta)
                x1_weight = t_input + delta
            else:
                raise ValueError(f"Unknown noise method: {self.rl_config.noise_method}")
        x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight
        return x_t_mean, x_t_std

    def forward(
        self,
        condition: BatchFeature,
        robot_input: BatchFeature,
        chains: torch.Tensor,
        denoise_inds: torch.Tensor,
    ) -> torch.Tensor:
        condition_emb = self.cond_emb_net(condition)
        state_emb = self.state_encoder(robot_input.state)
        embodiment_id = robot_input.embodiment_id
        # Set initial actions as the sampled noise.
        batch_size = condition_emb.shape[0]
        chains_log_probs = []

        if self.rl_config.joint_logprob:
            num_steps = self.config.num_steps
            initial_log_prob = self.get_logprob_norm(
                chains[:, 0],
                torch.zeros_like(chains[:, 0]),
                torch.ones_like(chains[:, 0]),
            )
            chains_log_probs.append(initial_log_prob)
        else:
            num_steps = 1
        for idx in range(num_steps):
            denoise_ind = denoise_inds[:, idx]
            chains_pre = chains[torch.arange(batch_size), denoise_ind]
            chains_next = chains[torch.arange(batch_size), denoise_ind + 1]
            x_t_mean, x_t_std = self.sample_mean_var_val(
                vl_embs=vl_embs,
                idx=denoise_ind,
                x_t=chains_pre,
                embodiment_id=embodiment_id,
                state_features=state_emb,
                mode="train",
                denoise_steps=self.num_inference_timesteps,
            )
            log_probs = self.get_logprob_norm(chains_next, x_t_mean, x_t_std)
            chains_log_probs.append(log_probs)

        chains_log_probs = torch.stack(chains_log_probs, dim=1)
        if compute_values:
            chains_values = self.get_value(vl_embs, state_features)
            chains_values = chains_values[:, None]
        else:
            chains_values = torch.zeros((batch_size, 1), device=chains_log_probs.device, dtype=vl_embs.dtype)  # (B, 1)
        return chains_log_probs, chains_values

    def compute_state_noise(self, batch_size: int, model_output: torch.Tensor, noise_type: FlowNoiseType, delta: torch.Tensor) -> dict:
        """
        Compute state noise standard deviation based on the noise type.
        Args:
            batch_size (int): The batch size.
            input (torch.Tensor): The input tensor for noise computation.
            noise_type (FlowNoiseType): The type of noise to compute ( FlowNoiseType.REINFLOW, FlowNoiseType.SDE).
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
            sigma = alpha * torch.sqrt((1 - self.timesteps) / torch.where(self.timesteps == 0, self.timesteps[1], self.timesteps))[:-1]
            # std=sqrt(dt)*sigma
            sigma = sigma.expand(batch_size, model_output.shape[1], self.cfg.output_dim)
            dict_["sigma"] = sigma
            dict_["state_noise_std"] = torch.sqrt(delta) * sigma
        else:
            raise ValueError(f"Unknown noise_type: {noise_type}")
        return dict_

    def cal_log_prob(self, noise_mean: torch.Tensor, noise_std: torch.Tensor) -> torch.Tensor:
        """计算噪声的对数概率(计算联合概率作为对数似然)。
        参数:
            noise_mean: [B, num_sample_step, D_a] 噪声均值
            noise_std: [B, num_sample_step, D_a] 噪声标准差"""
        dist = Normal(noise_mean, noise_std)
        log_prob = dist.log_prob(self.init_noise.to(noise_mean.device))  # [1, D_a] broadcast to [B, num_sample_step, D_a]
        return log_prob.sum(dim=-1).sum(dim=-1)  # [B] 每个样本的联合对数似然

    def compute_cfm_loss(
        self,
        state_head: torch.Tensor,
        x1: torch.Tensor,
        eps: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算条件流匹配 (CFM) 损失。

        参数:
            state_head: [B, N, D_s] 状态头
            x1: [B, N, D_a] 最终去噪动作
            eps: [B, N, D_a] 采样噪声
            t: [B, N, 1] 时间步

        返回:
            loss: [B] 每个样本的损失
        """
        B, N, D_a = eps.shape
        assert x1.shape == (B, N, D_a), f"x1 must be [B, N, D_a], got {x1.shape}"
        assert state_head.shape[0] == B, f"state_head must have batch size {B}, got {state_head.shape[0]}"
        assert t.shape == (B, N, 1), f"t must be [B, N, 1], got {t.shape}"

        x_t = (1 - t) * eps + t * x1  # [B, N, D_a]

        x1 = x1.reshape(B * N, -1)
        x_t = x_t.reshape(B * N, -1)
        state_head = state_head.reshape(B * N, -1)
        t = t.reshape(B * N, -1)

        inp = torch.cat([state_head, x_t, t], dim=1)  # [B*N, D_s + D_a + 1]
        velocity_pred = self(inp)  # [B*N, D_a]
        eps = eps.reshape(B * N, -1)
        return F.mse_loss(velocity_pred, x1 - eps, reduction="none").mean(dim=1).reshape(B, N)  # [B, N]
