import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from ..transformer_block.self_attention import SelfAttentionBlock
from ..operator import FeedForward, SinusoidalPositionalEmbedding, ScaleShift


class AdaLNDitBlock(nn.Module):
    def __init__(
        self,
        hidden_dim,
        condition_dim,
        condition_norm: bool = True,
        condition_mlp_hidden_dim: list[int] = [128],
        dropout=0.0,
        num_attention_heads=8,
        max_token_length=512,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        activate: str = "geglu",
        num_positional_embeddings: int = 1024,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = False,
    ):
        super().__init__()
        self.self_norm = nn.LayerNorm(hidden_dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        self.cond_norm = nn.LayerNorm(condition_dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps) if condition_norm else nn.Identity()
        self.pos_embed = SinusoidalPositionalEmbedding(hidden_dim, max_seq_length=num_positional_embeddings)
        self.self_scale_shift = ScaleShift(
            hidden_dim=hidden_dim, cond_dim=condition_dim, mlp_hidden_dims=condition_mlp_hidden_dim, activate=activate, use_shift=True
        )
        self.self_attention_block = SelfAttentionBlock(
            hidden_dim=hidden_dim,
            num_attention_heads=num_attention_heads,
            max_token_length=max_token_length,
        )
        self.self_out_scale = ScaleShift(
            hidden_dim=hidden_dim, cond_dim=condition_dim, mlp_hidden_dims=condition_mlp_hidden_dim, activate=activate, use_shift=False
        )
        self.feed_norm = nn.LayerNorm(hidden_dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        self.feed_scale_shift = ScaleShift(
            hidden_dim=hidden_dim, cond_dim=condition_dim, mlp_hidden_dims=condition_mlp_hidden_dim, activate=activate, use_shift=True
        )
        self.ff = FeedForward(
            dim=hidden_dim,
            dim_out=hidden_dim,
            mult=4,
            dropout=0.0,
            activation_fn=activate,
            final_dropout=False,
            bias=ff_bias,
            inner_dim=ff_inner_dim,
        )
        self.feed_out_scale = ScaleShift(
            hidden_dim=hidden_dim, cond_dim=condition_dim, mlp_hidden_dims=condition_mlp_hidden_dim, activate=activate, use_shift=False
        )
        self.attention_out_bias = nn.Linear(hidden_dim, hidden_dim) if attention_out_bias else nn.Identity()
        self.dropout = nn.Dropout(dropout) if final_dropout else nn.Identity()

    def forward(self, hidden_input, condition_input, self_attention_mask=None):
        # input norm
        norm_hidden_state = self.self_norm(hidden_input)
        norm_condition = self.cond_norm(condition_input)
        # pos embedding
        norm_hidden_state_pos = self.pos_embed(norm_hidden_state)
        # scale and shift
        scaled_hidden_state = self.self_scale_shift(norm_hidden_state_pos, norm_condition)
        # self attention
        attention_output = self.self_attention_block(scaled_hidden_state, attention_mask=self_attention_mask)
        # output scale
        attention_output = self.self_out_scale(attention_output, norm_condition) + hidden_input
        # feedforward input norm and scale shift
        norm_feed_input = self.feed_norm(attention_output)
        scaled_feed_input = self.feed_scale_shift(norm_feed_input, norm_condition)
        # feedforward
        ff_output = self.ff(scaled_feed_input)
        # feedforward output scale and residual connection
        ff_output = self.feed_out_scale(ff_output, norm_condition) + attention_output
        ff_output_bias = self.attention_out_bias(ff_output)
        output = self.dropout(ff_output_bias)
        return output
