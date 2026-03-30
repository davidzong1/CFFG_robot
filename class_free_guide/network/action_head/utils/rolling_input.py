import torch
import torch.nn as nn


class RollingInputBuffer(nn.Module):
    """
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
        total_size = window_size * self.input_cat_shape
        self._buffer = torch.zeros(self.batch_size, total_size, dtype=type, device=device)
        self.first_input_received = True

    def forward(self, x) -> tuple:
        shift_data = self._buffer[:, : self.input_cat_shape]
        self._buffer = torch.roll(self._buffer, shifts=-self.input_cat_shape, dims=1)
        self._buffer[:, -self.input_cat_shape :] = x.reshape(self.batch_size, -1)
        return (self._buffer, shift_data)


# ===================== Example =====================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rand_tensor = torch.randn(4096, 3, 3, device=device, dtype=torch.float32)  # batch_size=2, feature_dim=3
    buffer = RollingInputBuffer(window_size=5, input_shape=rand_tensor.shape, device=rand_tensor.device, type=rand_tensor.dtype)
    print(f"input shape: {buffer.batch_size}")
    for i in range(10):
        x = torch.randn(4096, 3, 3, device=device, dtype=torch.float32)  # batch_size=2, feature_dim=3
        output = buffer(x)
        print(f"Step {i+1}: input shape {x.shape} -> output shape {output.shape}")
