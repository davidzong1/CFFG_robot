import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from class_free_guide.network.base.norm import AdaLayerNorm
from ..operator import FeedForward, SinusoidalPositionalEmbedding


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        hidden_dim,  # Query 的输入维度
        cross_hidden_dim,  # Key/Value 的输入维度
        num_attention_heads=8,
        max_token_length=512,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_attention_heads = num_attention_heads
        self.cross_hidden_dim = cross_hidden_dim
        self.head_dim = hidden_dim // num_attention_heads
        self.max_token_length = max_token_length
        self._WQ = nn.Linear(hidden_dim, num_attention_heads * self.head_dim)
        self._WK = nn.Linear(cross_hidden_dim, num_attention_heads * self.head_dim)
        self._WV = nn.Linear(cross_hidden_dim, num_attention_heads * self.head_dim)

        # 输出投影
        self._WO = nn.Linear(num_attention_heads * self.head_dim, hidden_dim)

    def forward(self, hidden_input, cross_input, causal_mask=None):
        """
        Args:
            hidden_input: (batch,token_length, seq_len, hidden_dim)
            cross_input: (batch,token_length, cross_seq_len, cross_hidden_dim)
            causal_mask: (batch, 1, seq_len, cross_seq_len) 可选的因果掩码
        """
        assert hidden_input.shape[1] <= self.max_token_length
        batch_size, seq_len, _ = hidden_input.shape
        cross_seq_len = cross_input.shape[1]

        # 1. Linear projections to get Q, K, V
        # (B, S, D) -> (B, S, H, d_h) -> (B, H, S, d_h)
        Q = self._WQ(hidden_input).view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        K = self._WK(cross_input).view(batch_size, cross_seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        V = self._WV(cross_input).view(batch_size, cross_seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)

        # 2. calculating attention scores
        # (B, H, S, d_h) x (B, H, d_h, S_cross) -> (B, H, S, S_cross)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim**0.5)

        if causal_mask is not None:
            scores = scores.masked_fill(causal_mask == 0, float("-inf"))

        # 3. Softmax 与 加权求和
        attn = F.softmax(scores, dim=-1)
        # (B, H, S, S_cross) x (B, H, S_cross, d_h) -> (B, H, S, d_h)
        context = torch.matmul(attn, V)

        # 4. 合并多头并输出投影
        # (B, H, S, d_h) -> (B, S, H * d_h)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self._WO(context)

        return output


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
        num_positional_embeddings: int = 1024,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cross_hidden_dim = cross_hidden_dim
        # Positional embedding for cross attention
        self.pos_embed = SinusoidalPositionalEmbedding(hidden_dim, max_seq_length=num_positional_embeddings)
        # attention input norm
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        # cross attention module
        self.cross_attention = CrossAttentionBlock(
            hidden_dim=hidden_dim,
            cross_hidden_dim=cross_hidden_dim,
            num_attention_heads=num_attention_heads,
            max_token_length=max_token_length,
        )
        # feed forward network
        self.ff = FeedForward(
            dim=hidden_dim,
            dim_out=hidden_dim,
            mult=4,
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

    def forward(self, hidden_input, attention_mask, cross_input):
        norm_hidden_states = self.norm1(hidden_input)
        norm_hidden_states_pos = self.pos_embed(norm_hidden_states)
        attention_output = self.cross_attention(norm_hidden_states_pos, cross_input, causal_mask=attention_mask)
        attention_output = self.final_dropout(attention_output)
        # feedforward residual connection
        ff_output = self.ff(norm_hidden_states)
        return ff_output + attention_output + hidden_input
