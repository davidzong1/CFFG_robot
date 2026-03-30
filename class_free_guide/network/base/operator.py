import torch
import torch.nn as nn
import torch.nn.functional as F
from class_free_guide.network.utils.deprecate import deprecate
from .activate import GELU, GEGLU, ApproximateGELU, SwiGLU, LinearActivation
import math
from .mlp import MLP


class FeedForward(nn.Module):
    r"""
    A feed-forward layer.

    Parameters:
        dim (`int`): The number of channels in the input.
        dim_out (`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        final_dropout (`bool` *optional*, defaults to False): Apply a final dropout.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
        inner_dim=None,
        bias: bool = True,
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim, bias=bias)
        if activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh", bias=bias)
        elif activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim, bias=bias)
        elif activation_fn == "geglu-approximate":
            act_fn = ApproximateGELU(dim, inner_dim, bias=bias)
        elif activation_fn == "swiglu":
            act_fn = SwiGLU(dim, inner_dim, bias=bias)
        elif activation_fn == "linear-silu":
            act_fn = LinearActivation(dim, inner_dim, bias=bias, activation="silu")

        self.net = nn.ModuleList([])
        # project in
        self.net.append(act_fn)
        # project dropout
        self.net.append(nn.Dropout(dropout))
        # project out
        self.net.append(nn.Linear(inner_dim, dim_out, bias=bias))
        # FF as used in Vision Transformer, MLP-Mixer, etc. have a final dropout
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class SinusoidalPositionalEmbedding(nn.Module):
    """Apply positional information to a sequence of embeddings.

    Takes in a sequence of embeddings with shape (batch_size, seq_length, embed_dim) and adds positional embeddings to
    them

    Args:
        embed_dim: (int): Dimension of the positional embedding.
        max_seq_length: Maximum sequence length to apply positional embeddings

    """

    def __init__(self, embed_dim: int, max_seq_length: int = 32):
        super().__init__()
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(1, max_seq_length, embed_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        _, seq_length, _ = x.shape
        x = x + self.pe[:, :seq_length]
        return x


class ScaleShift(nn.Module):
    """
    A module that applies a scale and shift to the input tensor based on a conditioning vector.
    """

    def __init__(self, hidden_dim: int, cond_dim: int, mlp_hidden_dims: list[int], use_shift: bool = True):
        super().__init__()
        # 条件到(scale, shift)的映射，一次输出2 * hidden_dim后再拆成gamma, beta
        self.use_shift = use_shift
        if use_shift:
            self.proj = MLP(cond_dim, 2 * hidden_dim, mlp_hidden_dims, activation_fn="geglu", bias=True)
        else:
            self.proj = MLP(cond_dim, hidden_dim, mlp_hidden_dims, activation_fn="geglu", bias=True)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # (batch, cond_dim) -> (batch, 2 * hidden_dim)
        gamma_beta = self.proj(cond)
        beta = torch.zeros_like(gamma_beta)
        if self.use_shift:
            gamma, beta = gamma_beta.chunk(2, dim=-1)
        # 将(condition)广播到序列维度
        # y = (1 + gamma) * x + beta
        return x * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
