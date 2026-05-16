import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .utils.rolling_input import RollingInputBuffer


class state_action_denoise_roller(nn.Module):
    def __init__(self, roll_n_last, roll_n_future):
        super().__init__()
        self.roll_n_last = roll_n_last
        self.roll_n_future = roll_n_future

    def first_setup(self, state: torch.Tensor, action: torch.Tensor):
        """
        windows size is roll_n_last + roll_n_future + 1, where the last one is the current state and action to be estimated, roll from tail to head, the last input is the most recent one, and the first input is the oldest one.
         For example, if roll_n_last=2 and roll_n_future=2, the input
        """
        self.state_last_rolling = RollingInputBuffer(window_size=self.roll_n_last, input_shape=state.shape, device=self.device, type=state.dtype)
        self.state_future_rolling = RollingInputBuffer(window_size=self.roll_n_future, input_shape=state.shape, device=self.device, type=state.dtype)
        self.action_last_rolling = RollingInputBuffer(window_size=self.roll_n_last, input_shape=action.shape, device=self.device, type=action.dtype)
        self.action_future_rolling = RollingInputBuffer(
            window_size=self.roll_n_future, input_shape=action.shape, device=self.device, type=action.dtype
        )

    def get_future_data(self):
        return self.state_future_rolling.data, self.action_future_rolling.data

    def get_old_data(self):
        return self.state_last_rolling.data, self.action_last_rolling.data

    def update_all_future_data(self, state_future: torch.Tensor, action_future: torch.Tensor):
        old_future_state_data = self.state_future_rolling.update_all(state_future)
        old_future_action_data = self.action_future_rolling.update_all(action_future)
        return old_future_state_data, old_future_action_data

    def forward(self, now_state: torch.Tensor, now_action: torch.Tensor):
        sf, af = self.get_future_data()
        sl, al = self.get_old_data()
        data_state, data_action = torch.cat([sl, now_state.unsqueeze(1), sf], dim=1), torch.cat([al, now_action.unsqueeze(1), af], dim=1)
        return data_state, data_action

    def update(
        self,
        prior_state: torch.Tensor,
        prior_action: torch.Tensor,
        state_future: Optional[torch.Tensor] = None,
        action_future: Optional[torch.Tensor] = None,
    ):
        assert (state_future is not None and action_future is not None) or (
            state_future is None and action_future is None
        ), "Future state and action data must be coinsistent, either both provided or both None"
        if state_future is not None:
            # update all future data
            state_cache, action_cache = self.update_all_future_data(state_future, action_future)
            state_cache = state_cache[:, 0, ...]  # get the oldest future state data in the buffer
            action_cache = action_cache[:, 0, ...]  # get the oldest future action
            # update current state and action data
            _ = self.state_last_rolling(prior_state)
            _ = self.action_last_rolling(prior_action)
        else:
            # update newest future data with random noise, and get the oldest future data in the buffer as cache
            state_future_noise = torch.empty_like(prior_state).uniform_(-1, 1)
            action_future_noise = torch.empty_like(prior_action).uniform_(-1, 1)
            state_cache = self.state_future_rolling(state_future_noise)  # get the oldest state data in the buffer
            action_cache = self.action_future_rolling(action_future_noise)  # get the oldest action data in the buffer
            # update current state and action data
            _ = self.state_last_rolling(prior_state)
            _ = self.action_last_rolling(prior_action)
        return state_cache, action_cache
