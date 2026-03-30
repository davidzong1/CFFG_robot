import time
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal
from class_free_guide.network.nosise_gen.log_noise_nn import LogNoiseNN
from .flow_matching_base import FlowMatcherBase
from .flow_cfg import FlowControlCfg, FlowNoiseType, FlowInfo


class FlowControl(FlowMatcherBase):
    def __init__(self, model, cfg: FlowControlCfg):
        super().__init__(model, cfg)
