import torch
import torch.nn as nn
from typing import Optional


class RollingInputBuffer(nn.Module):
    """
    Rolling input buffer ,data from tail to head, the last input is the most recent one, and the first input is the oldest one.
    Args:
        window_size: rolling window size (number of past inputs to keep)
        input_shape: (input: torch.size())
        device: device where the buffer is located
    """

    def __init__(self, window_size, input_shape: tuple, device, type):
        super().__init__()
        self.window_size = window_size
        self.input_shape_one_batch = input_shape[1:]  # 输入的特征维度（不包括 batch_size 和 sequence_length）
        self.batch_size = input_shape[0]
        self.input_cat_shape = self.input_shape_one_batch.numel()
        self._buffer = torch.zeros(self.batch_size, self.window_size, *self.input_shape_one_batch, dtype=type, device=device)

    def reset(self, x: Optional[torch.Tensor] = None):
        if x is not None:
            self.batch_size = x.shape[0]
            for i in range(self.window_size):
                self._buffer[:, i, ...] = x
        else:
            self._buffer = torch.zeros(
                self.batch_size, self.window_size, *self.input_shape_one_batch, dtype=self._buffer.dtype, device=self._buffer.device
            )

    def forward(self, x) -> tuple:
        roll_data = self._buffer[:, 0, ...]  # get the oldest data in the buffer
        self._buffer = torch.roll(self._buffer, shifts=-1, dims=1)
        self._buffer[:, -1, ...] = x
        return self._buffer, roll_data


# ===================== Example =====================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rand_tensor = torch.randn(2, 3, 3, device=device, dtype=torch.float32)  # batch_size=2, feature_dim=3
    buffer = RollingInputBuffer(window_size=5, input_shape=rand_tensor.shape, device=rand_tensor.device, type=rand_tensor.dtype)
    print(f"input shape: {buffer.batch_size}")
    for i in range(10):
        x = torch.randn(2, 3, 3, device=device, dtype=torch.float32)  # batch_size=2, feature_dim=3
        output, _ = buffer(x)
        if i > 0:
            if (cache == output[:, 3, :, :].detach()).all().item():
                print("Rolling buffer is working correctly.")
            else:
                print("Rolling buffer is not working correctly.")
        print(f"Step {i+1}: input shape {x.shape} -> output shape {output.shape}")
        cache = x.detach()
