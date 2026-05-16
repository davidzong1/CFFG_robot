import torch
import torch.nn.functional as F
from torch import nn
import math
from typing import Optional


class LabelLinear(nn.Module):
    def __init__(
        self,
        input_dim: torch.Tensor,
        hidden_dim: torch.Tensor,
        label_len: Optional[int] = None,
        use_label: bool = False,
        use_bias: bool = True,
    ):
        super().__init__()
        if use_label:
            self.W = nn.Parameter(0.02 * torch.randn(label_len, input_dim, hidden_dim))
            self.b = nn.Parameter(torch.zeros(label_len, hidden_dim)) if use_bias else nn.Identity()
        else:
            self.W = nn.Parameter(0.02 * torch.randn(input_dim, hidden_dim))
            self.b = nn.Parameter(torch.zeros(hidden_dim)) if use_bias else nn.Identity()

    def forward(self, x: torch.Tensor, label_idx: int = 0):  # (B,S, D)
        selected_W = self.W[label_idx]
        selected_b = self.b[label_idx]
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)
