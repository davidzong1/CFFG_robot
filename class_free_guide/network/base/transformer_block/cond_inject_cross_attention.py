import torch.nn as nn
from typing import Optional
from ..operator import FeedForward, SinusoidalPositionalEmbedding
from .cross_attention import CrossAttentionBlock


class CondInjectCrossTransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim,
        condition_dim,
        condition_norm: bool = True,
        dropout=0.0,
        num_attention_heads=8,
        max_token_length=512,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        activate: str = "geglu",
        num_positional_embeddings: int = 1024,
        use_positional_embedding: bool = True,
        ff_bias: bool = True,
        ff_inner_dim: Optional[int] = None,
    ):
        super().__init__()
        self.pos_embed = (
            SinusoidalPositionalEmbedding(hidden_dim, max_seq_length=num_positional_embeddings) if use_positional_embedding else nn.Identity()
        )
        self.norm_cross = nn.LayerNorm(hidden_dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        self.norm_condition = (
            nn.LayerNorm(condition_dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps) if condition_norm else nn.Identity()
        )
        self.crossattentionblock = CrossAttentionBlock(
            hidden_dim=hidden_dim,
            cross_hidden_dim=condition_dim,
            num_attention_heads=num_attention_heads,
            max_token_length=max_token_length,
        )
        self.norm_feed = nn.LayerNorm(hidden_dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
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
        self.dropout = nn.Dropout(dropout) if final_dropout else nn.Identity()

    def forward(self, hidden_input, condition_input, attention_mask=None):
        norm_cross_hidden_states = self.norm_cross(hidden_input)
        norm_condition_input = self.norm_condition(condition_input)
        norm_cross_hidden_states_pos = self.pos_embed(norm_cross_hidden_states)
        cross_attention_output = self.crossattentionblock(
            norm_cross_hidden_states_pos,
            norm_condition_input,
            causal_mask=attention_mask,
        )
        cross_attention_output = cross_attention_output + hidden_input
        norm_feed_hidden_states = self.norm_feed(cross_attention_output)
        feedforward_output = self.feedforward(norm_feed_hidden_states)
        output = feedforward_output + cross_attention_output
        output = self.dropout(output)
        return output
