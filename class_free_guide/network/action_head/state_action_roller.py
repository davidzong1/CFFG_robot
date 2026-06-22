import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .utils.rolling_input import RollingInputBuffer


class state_action_denoise_roller(nn.Module):
    """
    structure: [[s_t-N,s_t-N+1,...,s_t,...s_t+N],[a_t-N,a_t-N+1,...,a_t,...a_t+N]]
    """

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
        self.action_last_rolling = RollingInputBuffer(window_size=self.roll_n_last, input_shape=action.shape, device=self.device, type=action.dtype)
        self.state_future_rolling = RollingInputBuffer(window_size=self.roll_n_future, input_shape=state.shape, device=self.device, type=state.dtype)
        self.action_future_rolling = RollingInputBuffer(
            window_size=self.roll_n_future, input_shape=action.shape, device=self.device, type=action.dtype
        )
        self.denoise_now_state = state.clone()
        self.denoise_now_action = action.clone()

    def get_future_data(self):
        return self.state_future_rolling.data, self.action_future_rolling.data

    def get_old_data(self):
        return self.state_last_rolling.data, self.action_last_rolling.data

    def update_all_future_data(
        self,
        denoise_state_future: torch.Tensor,
        denoise_action_future: torch.Tensor,
        state_future_noise: torch.Tensor,
        action_future_noise: torch.Tensor,
    ):
        denoise_state_future = torch.roll(denoise_state_future, shifts=-1, dims=1)  # roll to make the most future data at the end
        denoise_action_future = torch.roll(denoise_action_future, shifts=-1, dims=1)  # roll to make the most future data at the end
        self.state_future_rolling.update_all(denoise_state_future)
        self.action_future_rolling.update_all(denoise_action_future)

    def forward(self, now_state: torch.Tensor, now_action: torch.Tensor, denoise_state_future: torch.Tensor, denoise_action_future: torch.Tensor):
        state_future_noise = torch.empty_like(now_state).uniform_(-1, 1)  # most future noise
        action_future_noise = torch.empty_like(now_action).uniform_(-1, 1)  # most future noise
        self.denoise_now_state = denoise_state_future[:, 0:, ...]
        self.denoise_now_action = denoise_action_future[:, 0:, ...]
        # update all future data
        self.update_all_future_data(denoise_state_future, denoise_action_future, state_future_noise, action_future_noise)
        # update current state and action data
        _ = self.state_last_rolling(now_state)
        _ = self.action_last_rolling(now_action)
        sf, af = self.get_future_data()
        sl, al = self.get_old_data()
        return torch.cat([sl, self.denoise_now_state, sf], dim=1), torch.cat([al, self.denoise_now_action, af], dim=1)
