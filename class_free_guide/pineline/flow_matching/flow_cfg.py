from dataclasses import dataclass, field
import torch
from enum import Enum


class FlowInfo:
    x_0: torch.Tensor  # (*, action_dim)
    x_1: torch.Tensor  # (*, action_dim)
    x_std: torch.Tensor  # (*, num_sample_step, action_dim)
    x_mean: torch.Tensor  # (*, num_sample_step, action_dim)
    v_t: torch.Tensor  # (*, num_sample_step, action_dim)

    def __init__(self, x_0, x_1, x_mean, x_std, v_t):
        self.x_0 = x_0
        self.x_1 = x_1
        self.x_std = x_std
        self.x_mean = x_mean
        self.v_t = v_t

    def detach(self):
        self.x_0 = self.x_0.detach()
        self.x_1 = self.x_1.detach()
        self.x_std = self.x_std.detach()
        self.x_mean = self.x_mean.detach()
        self.v_t = self.v_t.detach()
        return self


@dataclass
class FlowNoiseType:
    REINFLOW = "reinflow"
    SDE = "sde"


@dataclass
class FlowMatchingCfg:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_sample_steps: int = field(default=10, metadata={"help": "number of sampling steps in flow matching"})
    noise_inference: FlowNoiseType = field(
        default=FlowNoiseType.SDE, metadata={"help": "type of noise inference for flow matching, can be 'reinflow' or 'sde'"}
    )
    real_denoise_step: int = field(default=10, metadata={"help": "number of sampling steps in flow matching"})
    # SDE parameters
    alpha: float = field(default=0.5, metadata={"help": "alpha parameter for SDE noise model, only used when noise_inference is 'sde'"})
    # Reinflow noise model parameters
    noise_activation: str = field(
        default="tanh",
        metadata={"help": "activation function for noise model in reinflow noise inference, only used when noise_inference is 'reinflow'"},
    )
    noise_hidden_dim: list[int] = field(
        default_factory=lambda: [128, 128],
        metadata={"help": "hidden dimension for noise model in reinflow noise inference, only used when noise_inference is 'reinflow'"},
    )
    # model parameters
    hidden_dim: int = field(default=128, metadata={"help": "to base_model input dimension"})
    output_dim: int = field(default=128, metadata={"help": "base_model output dimension"})


@dataclass
class FlowControlCfg(FlowMatchingCfg):
    # model parameters
    state_dim: int = field(default=72, metadata={"help": "dimension of state input to flow model"})
    action_dim: int = field(default=29, metadata={"help": "dimension of action output from flow model"})
    hidden_dim: int = field(default=1024, metadata={"help": "to base_model input dimension"})
    output_dim: int = field(default=29, metadata={"help": "base_model output dimension"})
    n_layers: int = field(default=3, metadata={"help": "number of layers in the flow model"})
    num_attention_heads: int = field(default=8, metadata={"help": "number of attention heads in the flow model"})
    model_pos_embedding: bool = field(default=True, metadata={"help": "whether to use positional embedding in the flow model"})
    model_ff_activate: str = field(default="geglu", metadata={"help": "activation function for feedforward network in the flow model"})
    model_ff_bias: bool = field(default=True, metadata={"help": "whether to use bias in the feedforward network of the flow model"})
    model_norm_eps: float = field(default=1e-5, metadata={"help": "epsilon for layer normalization in the flow model"})
    model_dropout: float = field(default=0.3, metadata={"help": "dropout rate for the flow model"})
    final_droupout: bool = field(default=True, metadata={"help": "whether to apply dropout to the final output of the flow model"})
    attention_bias: bool = field(default=False, metadata={"help": "whether to use attention bias in the flow model"})
    # model fine-tuning
    timer_forzen: bool = field(default=False, metadata={"help": "whether to freeze the time embedding in the flow model"})
    model_forzen: bool = field(default=False, metadata={"help": "whether to freeze the flow model parameters"})
    # roller parameters
    roller_n_last: int = field(default=2, metadata={"help": "number of last steps to roll for denoising"})
    roller_n_future: int = field(default=2, metadata={"help": "number of future steps to roll for denoising"})


class FlowMultimodalCfg(FlowControlCfg):
    num_categories: int = field(default=10, metadata={"help": "number of categories for condition embedding in multimodal flow model"})
    # state embedding parameters
    cond_embedding: bool = field(default=False, metadata={"help": "Data alignment for different dimensions"})
    condition_input_dim: int = field(default=128, metadata={"help": "dimension of state input to noise model."})
    cond_embedding_hidden_dim: int = field(default=128, metadata={"help": "hidden dimensions for condition embedding"})
    cond_embedding_out_dim: int = field(default=128, metadata={"help": "output dimension for condition embedding"})
    cond_activation: str = field(default="elu", metadata={"help": "activation function for condition embedding"})
