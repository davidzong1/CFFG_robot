import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU(nn.Module):
    def __init__(self, dim: int, dim_out: int, approximate: str | None = None, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim, dim_out, bias=bias)
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.proj(x), approximate=self.approximate)


class ApproximateGELU(GELU):
    def __init__(self, dim: int, dim_out: int, bias: bool = True):
        super().__init__(dim, dim_out, approximate="tanh", bias=bias)


class GEGLU(nn.Module):
    def __init__(self, dim: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim, dim_out * 2, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim, dim_out * 2, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.silu(gate)


class LinearActivation(nn.Module):
    def __init__(self, dim: int, dim_out: int, bias: bool = True, activation: str | None = None):
        super().__init__()
        self.proj = nn.Linear(dim, dim_out, bias=bias)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        if self.activation == "silu":
            return F.silu(x)
        if self.activation == "gelu":
            return F.gelu(x)
        return x


def resolve_activate_expand(name: str, dim: int, dim_out: int, bias: bool = True):
    if name == "gelu":
        return GELU(dim, dim_out, bias=bias)
    elif name == "geglu":
        return GEGLU(dim, dim_out, bias=bias)
    elif name == "geglu-approximate":
        return ApproximateGELU(dim, dim_out, bias=bias)
    elif name == "swiglu":
        return SwiGLU(dim, dim_out, bias=bias)
    elif name == "linear-silu":
        return LinearActivation(dim, dim_out, bias=bias, activation="silu")
    else:
        return None
