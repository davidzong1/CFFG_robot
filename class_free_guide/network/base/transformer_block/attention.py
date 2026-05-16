import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AttentionBlock(nn.Module):
    def __init__(
        self,
        hidden_dim,  # Query 的输入维度
        cross_hidden_dim: Optional[int] = None,  # Key/Value 的输入维度
        num_attention_heads=8,
        max_token_length=512,
        attention_out_bias: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cross_hidden_dim = cross_hidden_dim if cross_hidden_dim is not None else hidden_dim
        self.cross_attention_flag = cross_hidden_dim is not None
        self.num_attention_heads = num_attention_heads
        assert hidden_dim % num_attention_heads == 0, "hidden_dim must be divisible by num_attention_heads"
        self.head_dim = hidden_dim // num_attention_heads
        self.max_token_length = max_token_length
        self._WQ = nn.Linear(hidden_dim, hidden_dim)
        self._WK = nn.Linear(self.cross_hidden_dim, hidden_dim)
        self._WV = nn.Linear(self.cross_hidden_dim, hidden_dim)
        self._WO = nn.Linear(hidden_dim, hidden_dim) if attention_out_bias else nn.Identity()  # head data to hidden data

    def forward(
        self,
        hidden_input,
        cross_input: Optional[torch.Tensor] = None,
        mask2d: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            hidden_input: (batch,token_length, seq_len, hidden_dim)
            cross_input: (batch,token_length, cross_seq_len, cross_hidden_dim)
            mask2d: (batch, seq_len, cross_seq_len) 2D mask
        """
        batch_size, seq_len, _ = hidden_input.shape
        assert seq_len <= self.max_token_length, f"Sequence length {seq_len} exceeds maximum {self.max_token_length}"
        if self.cross_attention_flag:
            cross_seq_len = cross_input.shape[1]
            cross_input_cache = cross_input
        else:
            cross_seq_len = seq_len
            cross_input_cache = hidden_input

        # (B, S, D) -> (B, S, H, d_h) -> (B, H, S, d_h)
        Q = self._WQ(hidden_input).view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        K = self._WK(cross_input_cache).view(batch_size, cross_seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        V = self._WV(cross_input_cache).view(batch_size, cross_seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)

        # 2. calculating attention scores
        # (B, H, S, d_h) x (B, H, d_h, S_cross) -> (B, H, S, S_cross)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim**0.5)

        # 2.5 applying masks if provided
        if mask2d is not None:
            # (B, S, S_cross) -> (B, 1, S, S_cross) -> broadcast to (B, H, S, S_cross)
            assert mask2d.shape == (
                batch_size,
                seq_len,
                cross_seq_len,
            ), f"Mask shape {mask2d.shape} does not match expected {(batch_size, seq_len, cross_seq_len)}"
            scores = scores.masked_fill(mask2d.unsqueeze(1), float("-inf"))

        # 3. Softmax to get attention weights
        attn = F.softmax(scores, dim=-1)
        # (B, H, S, S_cross) x (B, H, S_cross, d_h) -> (B, H, S, d_h)
        context = torch.matmul(attn, V)

        # 4. Final linear projection to get output
        # (B, H, S, d_h) -> (B, S, H * d_h)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self._WO(context)

        return output
