import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from class_free_guide.network.base.transformer_block.attention import AttentionBlock
from ..operator import FeedForward, SinusoidalPositionalEmbedding


class CrossTransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim,
        cross_hidden_dim,
        dropout=0.0,
        num_attention_heads=8,
        max_token_length=512,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        activate: str = "geglu",
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cross_hidden_dim = cross_hidden_dim
        # Positional embedding for cross attention
        self.pos_embed = SinusoidalPositionalEmbedding(hidden_dim, max_seq_length=num_attention_heads)
        # attention input norm
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        # cross attention module
        self.cross_attention = AttentionBlock(
            hidden_dim=hidden_dim,
            cross_hidden_dim=cross_hidden_dim,
            num_attention_heads=num_attention_heads,
            max_token_length=max_token_length,
            attention_out_bias=attention_out_bias,
        )
        # feed forward network
        self.ff = FeedForward(
            dim=hidden_dim,
            dim_out=hidden_dim,
            dropout=dropout,
            activation_fn=activate,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )
        # attention output norm
        self.norm3 = nn.LayerNorm(hidden_dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        self.final_dropout = final_dropout
        if final_dropout:
            self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_input: torch.Tensor,
        cross_input: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None,
    ):
        norm_hidden_states = self.norm1(hidden_input)
        norm_hidden_states_pos = self.pos_embed(norm_hidden_states)
        attention_output = self.cross_attention(norm_hidden_states_pos, cross_input, query_mask=attention_mask, key_mask=cross_mask)
        attention_output = self.final_dropout(attention_output)
        if self.final_dropout:
            attention_output = self.dropout(attention_output)
        # feedforward residual connection
        ff_output = self.ff(norm_hidden_states)
        return ff_output + hidden_input
