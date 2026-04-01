import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils.rolling_input import RollingInputBuffer
from class_free_guide.network.base.dit_block.dit_ln_attention import DitBlock
from class_free_guide.network.base.operator import ScaleShift
from class_free_guide.network.action_head.utils.utils import TimestepEmbedding, Timesteps
from typing import Optional


class denoise_roller(nn.Module):
    def __init__(self, roll_n_last, roll_n_future):
        super().__init__()
        self.roll_n_last = roll_n_last
        self.roll_n_future = roll_n_future

    def first_setup(self, x: torch.Tensor):
        self.last_rolling = RollingInputBuffer(window_size=self.roll_n_last, input_shape=x.shape, device=self.device, type=x.dtype)
        self.future_rolling = RollingInputBuffer(window_size=self.roll_n_future, input_shape=x.shape, device=self.device, type=x.dtype)
        self.last_rolling.reset(x)

    def forward(self, x: torch.Tensor):
        future_noise = torch.rand_like(x)
        last_rolled, _ = self.last_rolling.roll(x)
        future_rolled, estimate_now_data = self.future_rolling.roll(future_noise)
        return last_rolled, future_rolled, estimate_now_data


class DenoiserTransformer(nn.Module):

    class TimestepEncoder(nn.Module):
        def __init__(self, embedding_dim, first_emb_dim=256, flip_sin_to_cos=True, downscale_freq_shift=1):
            super().__init__()
            self.time_proj = Timesteps(num_channels=first_emb_dim, flip_sin_to_cos=flip_sin_to_cos, downscale_freq_shift=downscale_freq_shift)
            self.timestep_embedder = TimestepEmbedding(input_dim=first_emb_dim, time_embed_dim=embedding_dim)

        def forward(self, timesteps):
            dtype = next(self.parameters()).dtype
            timesteps_proj = self.time_proj(timesteps).to(dtype)
            timesteps_emb = self.timestep_embedder(timesteps_proj)  # (N, D)
            return timesteps_emb

    def __init__(
        self,
        hidden_dim: int,
        condition_dim: int,
        output_dim: int,
        num_attention_heads: int,
        n_layers: int,
        cross_dim: Optional[int] = None,
        condition_hidden_dim: Optional[int | list[int]] = None,
        use_self_cross_attention: bool = False,
        use_positional_embedding: bool = True,
        max_token_length: int = 512,
        ff_activate: str = "geglu",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        ff_bias: bool = True,
        ff_inner_dim: Optional[int] = None,
        dropout: float = 0.0,
        final_droupout: bool = True,
        attention_bias: bool = False,
        timer_forzen: bool = False,
        model_forzen: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cross_dim = cross_dim

        self.condition_dim = condition_dim
        self.output_dim = output_dim
        self.num_heads = num_attention_heads
        self.n_layers = n_layers
        # Timestep encoder
        self.timestep_encoder = self.TimestepEncoder(embedding_dim=self.hidden_dim)
        if use_self_cross_attention:
            assert self.n_layers % 2 == 0, "n_layers must be even for self-cross attention"
            assert cross_dim is not None, "cross_dim must be specified when use_self_cross_attention is True"
            self.model = nn.ModuleList(
                [
                    (
                        (
                            DitBlock(
                                hidden_dim=self.hidden_dim,
                                condition_dim=self.condition_dim,
                                condition_hidden_dim=condition_hidden_dim,
                                cross_attention_dim=cross_dim,
                                dropout=dropout if i == n_layers - 1 else 0.0,
                                num_attention_heads=num_attention_heads,
                                max_token_length=max_token_length,
                                norm_elementwise_affine=norm_elementwise_affine,
                                norm_eps=norm_eps,
                                use_attention_out_scale=False,
                                use_feed_scale_shift=False,
                                use_feed_out_scale=True if i == n_layers - 1 else False,
                                final_dropout=final_droupout,
                                activate=ff_activate,
                                use_positional_embedding=use_positional_embedding,
                                ff_bias=ff_bias,
                                ff_inner_dim=ff_inner_dim,
                                attention_bias=attention_bias,
                            )
                        )
                        if i % 2 == 1
                        else (
                            DitBlock(
                                hidden_dim=self.hidden_dim,
                                condition_dim=self.condition_dim,
                                condition_hidden_dim=condition_hidden_dim,
                                dropout=dropout if i == n_layers - 1 else 0.0,
                                num_attention_heads=num_attention_heads,
                                max_token_length=max_token_length,
                                norm_elementwise_affine=norm_elementwise_affine,
                                norm_eps=norm_eps,
                                use_attention_out_scale=False,
                                use_feed_scale_shift=False,
                                use_feed_out_scale=True if i == n_layers - 1 else False,
                                final_dropout=final_droupout,
                                activate=ff_activate,
                                use_positional_embedding=use_positional_embedding,
                                ff_bias=ff_bias,
                                ff_inner_dim=ff_inner_dim,
                                attention_bias=attention_bias,
                            )
                        )
                    )
                    for i in range(n_layers)
                ]
            )
        else:
            self.model = nn.ModuleList(
                [
                    DitBlock(
                        hidden_dim=self.hidden_dim,
                        condition_dim=self.condition_dim,
                        condition_hidden_dim=condition_hidden_dim,
                        dropout=dropout if i == n_layers - 1 else 0.0,
                        num_attention_heads=num_attention_heads,
                        max_token_length=max_token_length,
                        norm_elementwise_affine=norm_elementwise_affine,
                        norm_eps=norm_eps,
                        use_attention_out_scale=False,
                        use_feed_scale_shift=False,
                        use_feed_out_scale=True if i == n_layers - 1 else False,
                        final_dropout=final_droupout,
                        activate=ff_activate,
                        use_positional_embedding=use_positional_embedding,
                        ff_bias=ff_bias,
                        ff_inner_dim=ff_inner_dim,
                        attention_bias=attention_bias,
                    )
                    for i in range(n_layers)
                ]
            )
        self.proj_out = nn.Linear(self.hidden_dim, self.output_dim)
        print(
            "Total number of DiT parameters: ",
            sum(p.numel() for p in self.parameters() if p.requires_grad),
        )
        if timer_forzen:
            for param in self.timestep_encoder.parameters():
                param.requires_grad = False
        print("Dit timestep encoder parameters frozen.")
        if model_forzen:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.proj_out.parameters():
                param.requires_grad = False
            print("Dit model parameters frozen.")

    def get_timestep_embedding(self, t_idx):
        return self.timestep_encoder(t_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        t_idx: torch.Tensor,
        cross_attention_states: Optional[torch.Tensor] = None,
        self_attention_mask: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # timestep embedding
        t_emb = self.timestep_encoder(t_idx)
        # DiT blocks
        for block in self.model:
            if isinstance(block, DitBlock):
                hidden_states = block(
                    hidden_input=hidden_states,
                    cross_attention_states=cross_attention_states,
                    condition_input=t_emb,
                    cross_input=cross_attention_states,
                    self_attention_mask=self_attention_mask,
                    cross_attention_mask=cross_attention_mask,
                )
            else:
                hidden_states = block(hidden_states, t_emb)
        output = self.proj_out(hidden_states)
        return output
