import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional
from torch.distributions import Normal
from class_free_guide.pineline.flow_matching.flow_matching_base import FlowMatcherBase
from class_free_guide.pineline.flow_matching.flow_cfg import FlowControlCfg, FlowNoiseType, FlowInfo
from class_free_guide.network.action_head.state_action_attention import DenoiserTransformer
from class_free_guide.network.action_head.state_action_roller import state_action_denoise_roller
from class_free_guide.network.base.mlp import MLP
from class_free_guide.algorithm.log_prob import get_logprob_norm
from class_free_guide.network.base.vae_block.mlp_vae import VAE
import math

try:
    from transformers.feature_extraction_utils import BatchFeature
except ImportError:
    from collections import UserDict as BatchFeature


class FlowControl(FlowMatcherBase):
    def __init__(self, cfg: FlowControlCfg, model: Optional[nn.Module] = None):
        super().__init__(model=model, cfg=cfg)
        self.last_sa_length = cfg.roll_n_last
        self.future_sa_length = cfg.roll_n_future
        self.roller = state_action_denoise_roller(roll_n_last=cfg.roll_n_last, roll_n_future=cfg.roll_n_future)
        self.total_token = cfg.roll_n_last + cfg.roll_n_future + 1  # (state+action)*roll_n_last + (state+action)*roll_n_future + current_state_action
        state_encoder = MLP(input_dim=cfg.hidden_dim, output_dim=cfg.state_dim, hidden_dims=cfg.coder_hidden_dim, activation="swish")
        action_encoder = MLP(input_dim=cfg.hidden_dim, output_dim=cfg.action_dim, hidden_dims=cfg.coder_hidden_dim, activation="swish")
        # Since it is only used for motion control itself without utilizing multimodal information, cross-attention is not employed
        self.model = DenoiserTransformer(
            hidden_dim=self.cfg.hidden_dim,
            condition_dim=1,  # only use time step as condition
            num_attention_heads=self.cfg.num_attention_heads,
            n_layers=self.cfg.n_layers,
            use_positional_embedding=self.cfg.model_pos_embedding,
            ff_activate=self.cfg.model_ff_activate,
            norm_eps=self.cfg.model_norm_eps,
            ff_bias=self.cfg.model_ff_bias,
            dropout=self.cfg.model_dropout,
            final_droupout=self.cfg.final_droupout,
            attention_bias=self.cfg.attention_bias,
            timer_forzen=self.cfg.timer_forzen,
            model_forzen=self.cfg.model_forzen,
        )
        state_decoder = MLP(input_dim=cfg.hidden_dim, output_dim=cfg.state_dim, hidden_dims=cfg.decoder_hidden_dim, activation="swish")
        action_decoder = MLP(input_dim=cfg.hidden_dim, output_dim=cfg.action_dim, hidden_dims=cfg.decoder_hidden_dim, activation="swish")
        self.state_vae = VAE(encoder_model=state_encoder, decoder_model=state_decoder)
        self.action_vae = VAE(encoder_model=action_encoder, decoder_model=action_decoder)
        self.state_cache = torch.zeros(self.cfg.batch_size, self.cfg.state_dim, dtype=torch.float32, device=self.cfg.device)
        self.action_cache = torch.zeros(self.cfg.batch_size, self.cfg.action_dim, dtype=torch.float32, device=self.cfg.device)
        self._state_action_mask()
        if self.cfg.noise_inference == FlowNoiseType.SDE:
            self.alpha = torch.ones(1, self.total_token * 2, device=self.device)
            index = torch.arange(self.total_token, device=self.device)
            state_k = index - self.cfg.roll_n_last
            state_den = self.total_token - self.cfg.roll_n_last - 1
            state_ramp = self.cfg.state_alpha * torch.sin((state_k / state_den) * math.pi / 2)
            state_index = torch.where(state_k <= 0, torch.zeros_like(state_k, dtype=torch.float32), state_ramp)
            action_k = index + 1 - self.cfg.roll_n_last
            action_den = self.total_token - self.cfg.roll_n_last
            action_ramp = self.cfg.action_alpha * torch.sin((action_k / action_den) * math.pi / 2)
            action_index = torch.where(action_k <= 0, torch.zeros_like(action_k, dtype=torch.float32), action_ramp)
            self.alpha[:, : self.total_token] = state_index
            self.alpha[:, self.total_token :] = action_index
        else:
            raise ValueError(f"Unknown noise method: {self.cfg.noise_inference}, only 'sde' is supported for now.")

        print("mask_2d shape:", self.mask_2d.shape)
        print("mask_2d:map\n", self.mask_2d[0, :, :])

    def _state_action_mask(self):
        """create mask for state and action, where state can attend to both state and action, while action can only attend to state
        Map:\n
        s:1,1,1,1,1,1,...\n
        s:0,1,1,1,1,1,...\n
        s:0,0,1,1,1,1,...\n
        ...\n
        a:1,0,...,1,0,...\n
        a:1,1,...,1,1,...\n
        a:1,1,...,1,1,...\n
        .:s,s,...,a,a,...\n
        """
        # [1,squence_length, squence_length]
        self.mask_2d = torch.zeros(1, 2 * self.total_token, 2 * self.total_token, dtype=torch.bool, device=self.device)
        # state mask
        n = self.total_token
        idx = torch.arange(n, device=self.device)
        mask = idx[None, :] >= idx[:, None]  # [n, n] 上三角含对角
        self.mask_2d[0, :n, :n] = mask
        # action mask
        n = self.total_token
        offset = n  # 你想偏移的行数
        idx = torch.arange(n, device=self.device)
        mask = idx[:, None] >= (idx[None, :])  # [n, n] 下三角不含对角
        self.mask_2d[0, n : n + offset, :n] = mask
        self.mask_2d[0, n : n + offset, n : n + offset] = mask
        pass

    def sample_noise_action(
        self,
        x_t: torch.Tensor,
        time_idx: int,
        inject_noise: bool,
        condition: Optional[torch.Tensor] = None,
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
        t_input = t_input * torch.ones(x_t.shape[0], 1, 1, dtype=torch.float32, device=x_t.device)  # [B, 1，1]
        delta = delta * torch.ones(x_t.shape[0], 1, 1, dtype=torch.float32, device=x_t.device)  # [B,1， 1]
        # model forward
        v_t = self.model(x_t, t_input)
        x0_pred = x_t - v_t * t_input
        x1_pred = x_t + v_t * (1 - t_input)
        if not inject_noise:
            x0_weight = 1 - (t_input + delta)
            x1_weight = t_input + delta  # notice the plus here, it's different from openpi.
            x_t_std: torch.Tensor = torch.zeros_like(t_input)
        else:
            noise_dict = self.compute_state_noise(time_idx, v_t, self.cfg.noise_inference, t_input, delta)
            x_t_std: torch.Tensor = noise_dict["state_noise_std"]  # [B,total_token,1]
            if self.cfg.noise_inference == FlowNoiseType.SDE:
                x0_weight = torch.ones_like(t_input) - (t_input + delta) - x_t_std**2 * delta / (2 * (1 - t_input))  # [B,total_token,1]
                x1_weight = t_input + delta  # [B,1,1]
            elif self.cfg.noise_inference == FlowNoiseType.REINFLOW:
                x0_weight = 1 - (t_input + delta)
                x1_weight = t_input + delta
            else:
                raise ValueError(f"Unknown noise method: {self.cfg.noise_inference}")
        x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight  # [B,total_token, Dh]
        return x_t_mean, x_t_std, t_input, v_t

    def train_forward(self, input: torch.Tensor, condition: Optional[torch.Tensor] = None) -> BatchFeature:
        raise NotImplementedError(
            "train_forward is not implemented for FlowControl, please use the train_forward function in FlowMatcherBase or implement your own train_forward function in FlowControl if you want to use the noise inference and roller mechanism in FlowControl."
        )

    def rolling_schedule(self, state: torch.Tensor, action: torch.Tensor):
        """
        Update the rolling buffer with the new state and action, and get the rolled state and action for model input.
        Args:
            state: [B, state_dim] current state
            action: [B, action_dim] current action
        Returns:
            rolled_state: [B, roll_n_last+roll_n_future+1, state_dim] rolled state for model input
            rolled_action: [B, roll_n_last+roll_n_future+1, action_dim] rolled action for model input
        """
        rolled_state, rolled_action = self.roller(state, action)
        return rolled_state, rolled_action

    def encoder_forward(self, state: torch.Tensor, action: torch.Tensor):
        """
        Encode the state and action into latent space using VAE, and then concatenate the latent state and action for model input.
        Args:
            state: [B, state_dim] current state
            action: [B, action_dim] current action
        Returns:
            state_latent: [B, latent_dim] latent state
            action_latent: [B, latent_dim] latent action
            model_input: [B, roll_n_last+roll_n_future+1, hidden_dim] concatenated latent state and action for model input
        """
        state_latent = self.state_vae(state)
        action_latent = self.action_vae(action)
        model_input = torch.cat([state_latent, action_latent], dim=-1)  # [B, hidden_dim]
        return state_latent, action_latent, model_input

    def flow_forward(self, state_hidden: torch.Tensor, action_hidden: torch.Tensor, guidance_condition: Optional[torch.Tensor] = None):
        x_t = torch.cat([state_hidden, action_hidden], dim=1)  # Token cat [B,t_s+ta,D_h]
        x_t = x_t.contiguous()
        time_stamp = []
        chain = [x_t]
        chain_v = []
        log_probs = []
        for i in range(self.cfg.num_sample_steps):
            x_t_mean, x_t_std, t_input, v_t = self.sample_noise_action(
                x_t,
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

    def decode_forward(self, x1: torch.Tensor):
        """
        Decode the noised action x1 to get the predicted clean action.
        Args:
            x1: [B, roll_n_last+roll_n_future+1, hidden_dim] the noised action at the last time step of the sampling process
        Returns:
            state_pred: [B, state_dim] the predicted clean state
            action_pred: [B, action_dim] the predicted clean action
        """
        state_hidden = x1[:, : self.total_token, :]  # [B, total_token, hidden_dim]
        action_hidden = x1[:, self.total_token :, :]  # [B, total_token, hidden_dim]
        state_pred = self.state_vae.decode(state_hidden)  # [B, total_token, state_dim]
        action_pred = self.action_vae.decode(action_hidden)  # [B, total_token, action_dim]
        return state_pred[:, -1, :], action_pred[:, -1, :]

    ##########################################################################
    # Flow Model Loss
    ##########################################################################
    def compute_cfm_loss(
        self,
        data: BatchFeature,
        x_ref: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate the conditional flow matching loss.

        Args:
            data: BatchFeature containing the training data, which should include:
                - "chain": [B, num_sample_steps+1, action_dim] the chain of noised actions at each time step
                - "chain_v": [B, num_sample_steps+1, action_dim] the predicted velocity at each time step
                - "time_stamp": [B, num_sample_steps+1, 1] the time stamp for each time step
            x_ref: [B, action_dim] the reference action (x0) for calculating the loss

        Returns:
            loss: [B] Loss for each sample in the batch
        """
        rand_index = torch.randint(0, self.cfg.num_sample_steps - 1, (1,), device=x_ref.device).item()
        t = data["time_stamp"][:, rand_index, :].unsqueeze(-1)  # [B, 1,1]
        x_t = data["chain"][:, rand_index, :, :]  # [B,token_length, D_a]
        v_ref = (x_ref - x_t) / (1 - t)  # [B,token_length, D_a]
        v_pred = data["chain_v"][:, rand_index, :, :]  # [B, token_length, D_a]
        loss = F.mse_loss(v_pred, v_ref, reduction="none").mean(dim=-1)  # [B]
        loss_mean = loss.mean(dim=0)
        return loss_mean

    def cal_state_action_encoder_loss(self, state: torch.Tensor, action: torch.Tensor):
        """
        Calculate the VAE reconstruction loss for state and action.
        This can be used to ensure that the latent space of the VAE captures the essential information of the state and action,
        which can help improve the performance of the flow model.
        """
        state_cache = state.detach()  # Detach the state tensor to prevent gradients from flowing back to the original state during VAE training
        action_cache = action.detach()  # Detach the action tensor to prevent gradients from flowing back to the original action during VAE training
        state_latent, state_mu, state_log_var = self.state_vae.encode(state_cache)
        action_latent, action_mu, action_log_var = self.action_vae.encode(action_cache)
        state_recon = self.state_vae.decode(state_latent)
        action_recon = self.action_vae.decode(action_latent)
        state_decoder_loss = self.state_vae.cal_vae_loss(state_recon, state, state_mu, state_log_var)
        action_decoder_loss = self.action_vae.cal_vae_loss(action_recon, action, action_mu, action_log_var)
        return state_decoder_loss, action_decoder_loss

    ##########################################################################
    # VAE Loss
    ##########################################################################
    def compute_last_state_action_consistency_loss(
        self,
        data: BatchFeature,
        state_ref: torch.Tensor,
        action_ref: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the consistency loss between the last state and action in the chain and the reference state and action."""
        last_state_pred = data["x1_prev"][:, : self.last_sa_length, :]  # [B, last_sa_length, D_s]
        last_action_pred = data["x1_prev"][:, self.total_token : self.total_token + self.last_sa_length, :]  # [B, last_sa_length,D_a]
        state_loss = F.mse_loss(last_state_pred, state_ref, reduction="mean")
        action_loss = F.mse_loss(last_action_pred, action_ref, reduction="mean")
        return state_loss + action_loss


if __name__ == "__main__":

    cfg = FlowControlCfg()
    cfg.action_dim = 1
    cfg.state_dim = 1
    flow_control = FlowControl(cfg=cfg)
