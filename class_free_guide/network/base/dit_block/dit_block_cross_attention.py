import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from ..operator import SinusoidalPositionalEmbedding
from ..transformer_block.cond_inject_cross_attention import CondInjectCrossTransformerBlock
from ..transformer_block.self_attention import SelfAttentionBlock


class DitCrossAttentionBlock(nn.Module):
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
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = False,
    ):
        super().__init__()
        self.self_norm = nn.LayerNorm(hidden_dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        self.pos_embed = SinusoidalPositionalEmbedding(hidden_dim, max_seq_length=num_positional_embeddings)
        self.self_attention_block = SelfAttentionBlock(
            hidden_dim=hidden_dim,
            num_attention_heads=num_attention_heads,
            max_token_length=max_token_length,
        )
        self.self_cross_block = CondInjectCrossTransformerBlock(
            hidden_dim=hidden_dim,
            condition_dim=condition_dim,
            condition_norm=condition_norm,
            dropout=dropout,
            num_attention_heads=num_attention_heads,
            max_token_length=max_token_length,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            final_dropout=False,
            activate=activate,
            num_positional_embeddings=num_positional_embeddings,
            use_positional_embedding=False,
            ff_bias=ff_bias,
            ff_inner_dim=ff_inner_dim,
        )
        self.cond_inject = nn.Linear(condition_dim, hidden_dim * 2)
        self.attention_out_bias = nn.Linear(hidden_dim, hidden_dim) if attention_out_bias else nn.Identity()
        self.dropout = nn.Dropout(dropout) if final_dropout else nn.Identity()

    def forward(self, hidden_input, condition_input, self_attention_mask=None, cross_attention_mask=None):
        hidden_state_input = self.self_norm(hidden_input)
        hidden_state_input = self.pos_embed(hidden_state_input)
        cross_attention_output = self.self_attention_block(hidden_state_input, attention_mask=self_attention_mask)
        cross_attention_output = cross_attention_output + hidden_input
        cross_attention_output = self.self_cross_block(cross_attention_output, condition_input, attention_mask=cross_attention_mask)
        scale, shift = self.cond_inject(F.silu(condition_input)).chunk(2, dim=-1)
        cross_attention_output = cross_attention_output * (scale + 1) + shift
        cross_attention_output_bias = self.attention_out_bias(cross_attention_output)
        output = self.dropout(cross_attention_output_bias)
        return output
