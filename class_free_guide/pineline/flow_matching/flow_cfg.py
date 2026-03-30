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
    # control parameters
    state_dim: int = field(default=72, metadata={"help": "dimension of state input to flow model"})
    action_dim: int = field(default=29, metadata={"help": "dimension of action output from flow model"})
    history_length: int = field(default=4, metadata={"help": "number of historical states and actions to use as input to flow model"})


class FlowMultimodalCfg(FlowControlCfg):
    num_categories: int = field(default=10, metadata={"help": "number of categories for condition embedding in multimodal flow model"})
    # state embedding parameters
    cond_embedding: bool = field(default=False, metadata={"help": "Data alignment for different dimensions"})
    condition_input_dim: int = field(default=128, metadata={"help": "dimension of state input to noise model."})
    cond_embedding_hidden_dim: int = field(default=128, metadata={"help": "hidden dimensions for condition embedding"})
    cond_embedding_out_dim: int = field(default=128, metadata={"help": "output dimension for condition embedding"})
    cond_activation: str = field(default="elu", metadata={"help": "activation function for condition embedding"})
