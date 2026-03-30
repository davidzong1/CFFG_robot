import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils.rolling_input import RollingInputBuffer
from class_free_guide.network.base.dit_block.dit_block_cross_attention import DitCrossAttentionBlock
from class_free_guide.network.base.dit_block.dit_block_cross_attention import CondInjectCrossTransformerBlock


class DenoiserTransformer(nn.Module):
    def __init__(
        self,
        state_action_dim,
        condition_dim,
        num_attention_heads,
        n_layers,
        norm_elementwise_affine: bool = True,
        norm_type="ada_norm",
        norm_eps=1e-5,
        compute_dtype=torch.float32,
        roll_n_last=None,
        roll_n_future=None,
        device="cpu",
    ):
        super().__init__()
        self.last_rolling = RollingInputBuffer(window_size=roll_n_last, input_shape=state_action_dim, device=device, type=compute_dtype)
        self.future_rolling = RollingInputBuffer(window_size=roll_n_future, input_shape=state_action_dim, device=device, type=compute_dtype)
        self.attention_dim = state_action_dim * (roll_n_last + roll_n_future + 1)
        self.cross_attention_dim = condition_dim * (roll_n_last + roll_n_future + 1)
        self.n_layers = n_layers
        self.model = nn.ModuleList(
            [
                DitCrossAttentionBlock(
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
                )
                for _ in range(n_layers)
            ]
        )
