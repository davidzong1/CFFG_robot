import torch
import torch.nn as nn
from typing import Optional
from .mlp import MLP
from .operator import ScaleShift


class ConditionEmbNorm(nn.Module):
    """
    One dimension LayerNorm with condition embedding, which is used to inject the condition information into the hidden state.
    """

    def __init__(
        self,
        emb_dim: int,
        condition_dim: int,
        hidden_dim: Optional[int] = None,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.shift_scale_layer = ScaleShift(
            hidden_dim=emb_dim,
            cond_dim=condition_dim,
            mlp_hidden_dims=[hidden_dim] if hidden_dim is not None else None,
            use_shift=True,
        )
        self.norm = nn.LayerNorm(emb_dim, norm_eps, norm_elementwise_affine)

    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.shift_scale_layer(x, condition)
        x = self.norm(x)
        return x
