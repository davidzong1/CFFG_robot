import torch
import torch.nn.functional as F
from torch import nn
import math
from class_free_guide.network.utils.utils import get_param, resolve_nn_activation
from class_free_guide.network.base.activate import resolve_activate_expand


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    Args
        timesteps (torch.Tensor):
            a 1-D Tensor of N indices, one per batch element. These may be fractional.
        embedding_dim (int):
            the dimension of the output.
        flip_sin_to_cos (bool):
            Whether the embedding order should be `cos, sin` (if True) or `sin, cos` (if False)
        downscale_freq_shift (float):
            Controls the delta between frequencies between dimensions
        scale (float):
            Scaling factor applied to the embeddings.
        max_period (int):
            Controls the maximum frequency of the embeddings
    Returns
        torch.Tensor: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        input_dim: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int = None,
        post_act_fn: str | None = None,
        cond_proj_dim=None,
        sample_proj_bias=True,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(input_dim, time_embed_dim, sample_proj_bias)

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, input_dim, bias=False)
        else:
            self.cond_proj = None

        self.act = resolve_nn_activation(act_fn)
        if self.act is None:
            if self.act is None:
                self.act = resolve_activate_expand(act_fn, time_embed_dim, time_embed_dim)
            else:
                self.act = resolve_activate_expand(
                    act_fn,
                    time_embed_dim,
                    time_embed_dim,
                )
            if self.act is None:
                raise ValueError(f"Invalid activation function '{self.act}'. Valid activations are: {list(resolve_nn_activation('').keys())}")

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out, sample_proj_bias)

        if post_act_fn is None:
            self.post_act = None
        else:
            self.post_act = resolve_nn_activation(post_act_fn)
            if self.post_act is None:
                if self.post_act is None:
                    self.post_act = resolve_activate_expand(post_act_fn, time_embed_dim, time_embed_dim)
                else:
                    self.post_act = resolve_activate_expand(
                        post_act_fn,
                        time_embed_dim,
                        time_embed_dim,
                    )
                if self.post_act is None:
                    raise ValueError(
                        f"Invalid activation function '{self.post_act}'. Valid activations are: {list(resolve_nn_activation('').keys())}"
                    )

    def forward(self, sample, condition=None):
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)

        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample


class Timesteps(nn.Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float, scale: int = 1):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        timesteps: shape (N,) 1D tensor of timesteps
        returns: shape (N, num_channels) sinusoidal timestep embeddings
        """
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )
        return t_emb


def swish(x):
    return x * torch.sigmoid(x)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Produces a sinusoidal encoding of shape (B, T, w)
    given timesteps of shape (B, T).
    """

    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps):
        # timesteps: shape (B, T)
        # We'll compute sin/cos frequencies across dim T
        timesteps = timesteps.float()  # ensure float

        B, T = timesteps.shape
        device = timesteps.device

        half_dim = self.embedding_dim // 2
        # typical log space frequencies for sinusoidal encoding
        exponent = -torch.arange(half_dim, dtype=torch.float, device=device) * (torch.log(torch.tensor(10000.0)) / half_dim)
        # Expand timesteps to (B, T, 1) then multiply
        freqs = timesteps.unsqueeze(-1) * exponent.exp()  # (B, T, half_dim)

        sin = torch.sin(freqs)
        cos = torch.cos(freqs)
        enc = torch.cat([sin, cos], dim=-1)  # (B, T, w)

        return enc


class CategorySpecificLinear(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_categories):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids=0):
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    """if num_categories=1, this is just a regular MLP. If num_categories>1, this is a collection of separate MLPs and the forward pass selects which MLP to use based on cat_ids."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_categories=1):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(input_dim, hidden_dim, num_categories)
        self.layer2 = CategorySpecificLinear(hidden_dim, output_dim, num_categories)

    def forward(self, x, cat_ids=0):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError("Expected `timesteps` to have shape (B,) so we can replicate across T.")

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.W1(actions, cat_ids)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # 5) Finally W3 => (B, T, w)
        x = self.W3(x, cat_ids)
        return x


class CondCategorySpecificMLP(nn.Module):
    """A category-specific MLP with a condition injected from input to output. If num_categories=1, this is just a regular MLP with a residual connection."""

    def __init__(self, input_dim, hidden_dim, output_dim, cond_dim, hidden_layer=1, num_categories=1):
        super().__init__()
        if hidden_layer < 1:
            raise ValueError("hidden_layer must be at least 1")
        self.num_categories = num_categories
        self.Linear_list = nn.ModuleList(
            CategorySpecificLinear(input_dim, hidden_dim[0], num_categories),
        )
        self.condiction_layer = CategorySpecificLinear(cond_dim, hidden_dim, num_categories)
        for i in range(hidden_layer - 1):
            self.Linear_list.extend([CategorySpecificLinear(hidden_dim[i], hidden_dim[i + 1], num_categories)])
        self.Linear_list.extend([CategorySpecificLinear(hidden_dim[-1], output_dim, num_categories)])

    def forward(self, x, cond, cat_ids=0):
        hidden = F.relu(self.Linear_list[0](x, cat_ids))
        cond = self.condiction_layer(cond, cat_ids)
        hidden += cond
        for linear in self.Linear_list[1:-1]:
            hidden = F.relu(linear(hidden, cat_ids))
            hidden += cond
        return hidden
