import torch
import torch.nn as nn
from typing import Optional
from ..operator import FeedForward, SinusoidalPositionalEmbedding
from ..norm import ConditionEmbNorm
from ..transformer_block.attention import AttentionBlock
from ..operator import ScaleShift


class CondInjectCrossTransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim,
        cross_dim,
        condition_dim,
        condition_norm: bool = False,
        dropout=0.0,
        num_attention_heads=8,
        max_token_length=512,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        use_feed_scale_shift: bool = False,
        use_feed_out_scale: bool = True,
        final_dropout: bool = False,
        activate: str = "geglu",
        use_positional_embedding: bool = True,
        ff_bias: bool = True,
        ff_inner_dim: Optional[int] = None,
        attention_out_bias: bool = False,
    ):
        super().__init__()
        self.pos_embed = SinusoidalPositionalEmbedding(hidden_dim, max_seq_length=num_attention_heads) if use_positional_embedding else nn.Identity()
        self.norm_hidden = nn.LayerNorm(hidden_dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        self.norm_cross = nn.LayerNorm(cross_dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        self.norm_condition = (
            nn.LayerNorm(condition_dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps) if condition_norm else nn.Identity()
        )
        self.crossattentionblock = AttentionBlock(
            hidden_dim=hidden_dim,
            cross_hidden_dim=condition_dim,
            num_attention_heads=num_attention_heads,
            max_token_length=max_token_length,
            attention_out_bias=attention_out_bias,
        )
        self.norm_feed = (
            ConditionEmbNorm(emb_dim=hidden_dim, condition_dim=condition_dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
            if use_feed_scale_shift
            else nn.Identity()
        )
        self.feedforward = FeedForward(
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
            ScaleShift(hidden_dim=hidden_dim, cond_dim=condition_dim, activate=activate, use_shift=False) if use_feed_out_scale else nn.Identity()
        )
        self.dropout = nn.Dropout(dropout) if final_dropout else nn.Identity()

    def forward(
        self,
        hidden_input: torch.Tensor,
        cross_input: torch.Tensor,
        condition_input: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None,
    ):
        norm_cross_hidden_states = self.norm_cross(hidden_input)
        norm_condition_input = self.norm_condition(condition_input)
        # pos emb
        norm_cross_hidden_states_pos = self.pos_embed(norm_cross_hidden_states)
        # cross attention
        cross_attention_output = self.crossattentionblock(
            hidden_input=norm_cross_hidden_states_pos,
            cross_input=norm_condition_input,
            query_mask=attention_mask,
            key_mask=cross_mask,
        )
        # dropout after attention
        cross_attention_output = self.dropout(cross_attention_output)
        if cross_attention_output.ndim == 4:
            cross_attention_output = cross_attention_output.squeeze(1)
        # cross residual
        cross_attention_output = cross_attention_output + hidden_input
        # condition inject feedforward
        norm_feed_hidden_states = self.norm_feed(cross_attention_output, norm_condition_input)
        # feedforward
        feedforward_output = self.feedforward(norm_feed_hidden_states)
        # feedforward out scale
        feedforward_output = self.feed_out_scale(feedforward_output, norm_condition_input)
        # feedforward residual
        output = feedforward_output + cross_attention_output
        if output.ndim == 4:
            output = output.squeeze(1)
        return output
