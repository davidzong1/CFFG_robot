import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base.mlp import MLP


class LogNoiseNN(nn.Module):
    """
    生成可学习的探索噪声网络，可按时间/状态条件化：\sigma(s, t) 或 \sigma(s)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        logprob_denoising_std_range: list,  # [min_std, max_std]
        hidden_dims=[16],  # [8]  [32],
        activation="Tanh",
    ):
        super().__init__()
        self.mlp_logvar = MLP(input_dim=input_dim, output_dim=output_dim, hidden_dims=hidden_dims, activation=activation)
        self.set_noise_range(logprob_denoising_std_range)

    def set_noise_range(self, logprob_denoising_std_range: list):
        self.logprob_denoising_std_range = logprob_denoising_std_range
        min_logprob_denoising_std = self.logprob_denoising_std_range[0]
        max_logprob_denoising_std = self.logprob_denoising_std_range[1]
        # 存储最小和最大对数方差作为不可训练的参数(期望为0)
        self.logvar_min = torch.nn.Parameter(
            torch.log(torch.tensor(min_logprob_denoising_std**2, dtype=torch.float32)),
            requires_grad=False,
        )
        self.logvar_max = torch.nn.Parameter(
            torch.log(torch.tensor(max_logprob_denoising_std**2, dtype=torch.float32)),
            requires_grad=False,
        )

    def process_noise(self, noise_logvar):
        """
        输入:
            noise_logvar: torch.Tensor([B, Ta, Da])，log \sigma^2
        输出:
            noise_std: torch.Tensor([B, 1, Ta * Da])，\sigma，数值被限制在 [min_logprob_denoising_std, max_logprob_denoising_std]
        """
        noise_logvar = noise_logvar
        noise_logvar = torch.tanh(noise_logvar)
        noise_logvar = self.logvar_min + (self.logvar_max - self.logvar_min) * (noise_logvar + 1) / 2.0
        noise_std = torch.exp(0.5 * noise_logvar)
        return noise_std

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播以生成探索噪声的标准差
        :param state: 状态输入张量，形状为 (batch_size, input_dim)
        :return: 探索噪声的标准差张量，形状为 (batch_size, output_dim)
        """
        logvar = self.mlp_logvar(state)  # 预测对数方差
        return self.process_noise(logvar)  ## 返回处理后的标准差
