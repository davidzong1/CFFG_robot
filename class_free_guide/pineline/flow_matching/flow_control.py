import time
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal
from .flow_matching_base import FlowMatcherBase
from .flow_cfg import FlowControlCfg, FlowNoiseType, FlowInfo
from class_free_guide.network.action_head.state_action_attention import denoise_roller, DenoiserTransformer


class FlowControl(FlowMatcherBase):
    def __init__(self, model, cfg: FlowControlCfg):
        super().__init__(model, cfg)
        self.roller = denoise_roller(roll_n_last=cfg.roll_n_last, roll_n_future=cfg.roll_n_future)
        # Since it is only used for motion control itself without utilizing multimodal information, cross-attention is not employed
        self.model = DenoiserTransformer(
            hidden_dim=self.cfg.hidden_dim,
            condition_dim=1,  # only use time step as condition
            output_dim=self.cfg.output_dim,
            num_attention_heads=self.cfg.num_attention_heads,
            n_layers=self.cfg.n_layers,
            use_positional_embedding=self.cfg.model_pos_embedding,
            ff_activate=self.cfg.model_ff_activate,
            norm_eps=self.cfg.model_norm_eps,
            ff_bias=self.cfg.model_ff_bias,
            dropout=self.cfg.model_dropout,
            final_droupout=self.cfg.final_droupout,
            attention_bias=self.cfg.attention_bias,
            timer_forzen=self.cfg.timer_forzen,
            model_forzen=self.cfg.model_forzen,
        )

    def _state_action_mask(self):
        # create mask for state and action, where state can attend to both state and action, while action can only attend to state
        mask = torch.zeros(
            (self.cfg.state_dim + self.cfg.action_dim, self.cfg.state_dim + self.cfg.action_dim), dtype=torch.bool, device=self.cfg.device
        )
        mask[: self.cfg.state_dim, :] = True  # state can attend to both state and action
        mask[self.cfg.state_dim :, : self.cfg.state_dim] = True  # action can only attend to state
        return mask
