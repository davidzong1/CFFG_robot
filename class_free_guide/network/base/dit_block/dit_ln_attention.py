import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from ..transformer_block.attention import AttentionBlock
from ..operator import FeedForward, SinusoidalPositionalEmbedding, ScaleShift
from class_free_guide.network.base.norm import ConditionEmbNorm


class DitBlock(nn.Module):
    def __init__(
        self,
        hidden_dim,
        condition_dim,
        cross_attention_dim: Optional[int] = None,
        condition_mlp_hidden_dim: list[int] = [128],
        dropout=0.0,
        num_attention_heads=8,
        max_token_length=512,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        use_positional_embedding: bool = True,
        use_attention_out_scale: bool = False,
        use_feed_scale_shift: bool = False,
        use_feed_out_scale: bool = True,
        final_dropout: bool = False,
        activate: str = "geglu",
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else hidden_dim
        self.self_norm = ConditionEmbNorm(
            emb_dim=hidden_dim,
            condition_dim=condition_dim,
            hidden_dim=condition_mlp_hidden_dim,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
        )
        self.pos_embed = SinusoidalPositionalEmbedding(hidden_dim, max_seq_length=num_attention_heads) if use_positional_embedding else nn.Identity()
        self.self_attention_block = AttentionBlock(
            hidden_dim=hidden_dim,
            cross_hidden_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads,
            max_token_length=max_token_length,
            attention_out_bias=attention_out_bias,
        )
        self.self_out_scale = (
            ScaleShift(hidden_dim=hidden_dim, cond_dim=condition_dim, mlp_hidden_dims=condition_mlp_hidden_dim, use_shift=False)
            if use_attention_out_scale
            else nn.Identity()
        )
        self.feed_norm = nn.LayerNorm(hidden_dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        self.feed_scale_shift = (
            ScaleShift(hidden_dim=hidden_dim, cond_dim=condition_dim, mlp_hidden_dims=condition_mlp_hidden_dim, use_shift=True)
            if use_feed_scale_shift
            else nn.Identity()
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
        self.feed_out_scale = (
            ScaleShift(hidden_dim=hidden_dim, cond_dim=condition_dim, mlp_hidden_dims=condition_mlp_hidden_dim, use_shift=False)
            if use_feed_out_scale
            else nn.Identity()
        )
        self.dropout = nn.Dropout(dropout) if final_dropout else nn.Identity()

    def forward(
        self,
        hidden_input,
        condition_input,
        cross_input: Optional[torch.Tensor] = None,
        mask2d: Optional[torch.Tensor] = None,
    ):
        # input norm
        norm_hidden_state = self.self_norm(hidden_input, condition_input)
        # pos embedding
        norm_hidden_state_pos = self.pos_embed(norm_hidden_state)
        # self attention
        attention_output = self.self_attention_block(
            hidden_input=norm_hidden_state_pos,
            cross_input=cross_input,
            mask2d=mask2d,
        )
        # dropout after attention
        attention_output = self.dropout(attention_output)
        # output scale
        attention_output = self.self_out_scale(attention_output, condition_input) + hidden_input
        if attention_output.ndim == 4:
            attention_output = attention_output.squeeze(1)
        # feedforward input norm and scale shift
        norm_feed_input = self.feed_norm(attention_output)
        scaled_feed_input = self.feed_scale_shift(norm_feed_input, condition_input)
        # feedforward
        ff_output = self.ff(scaled_feed_input)
        # feedforward output scale and residual connection
        ff_output = self.feed_out_scale(ff_output, condition_input) + attention_output
        if ff_output.ndim == 4:
            ff_output = ff_output.squeeze(1)
        return ff_output
