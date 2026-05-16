"""简单的二维点 Flow Matching Demo。

运行方式（在项目根目录下）：

        python -m class_free_guide.test.flow_matching

本例子学习一个将二维高斯分布 N(mu0, I) 变换到 N(mu1, I) 的流场。
使用最基础的 flow matching 损失：

  - 采样 x0 ~ p0, x1 ~ p1
  - 采样 t ~ U(0, 1)
  - 构造中间点 x_t = (1 - t) * x0 + t * x1
  - 理论速度 v*(x_t, t) = x1 - x0
  - 训练网络 v_theta(x_t, t) 拟合 v*

训练好以后，从 p0 采样若干点，用欧拉积分沿着 v_theta 推进，即可得到近似的 p1 样本。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from class_free_guide.network.base.mlp import MLP
from class_free_guide.pineline.flow_matching.flow_matching_base import FlowMatcherBase
from class_free_guide.pineline.flow_matching.flow_cfg import FlowMatchingCfg
from torch import nn


class FlowDemoConfig:
    """Flow matching Demo 的超参数配置。"""

    batch_size: int = 256
    train_steps: int = 2000
    learning_rate: float = 1e-3
    model_hidden_dim: list[int] = [64, 32]
    # 用于可视化/评估的采样数
    num_eval_samples: int = 1024
    # 用于绘制轨迹的采样数（太大图会很乱）
    num_plot_samples: int = num_eval_samples

    def __init__(self):
        self.cfg = FlowMatchingCfg()
        self.cfg.hidden_dim = 3
        self.cfg.output_dim = 2
        self.cfg.noise_inference = "sde"
        self.cfg.real_denoise_step = 6
        self.cfg.num_sample_steps = 10
        self.device = self.cfg.device


def sample_gaussians(batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """采样二维高斯点对 (x0, x1)。

    这里简单选择：
        - 源分布 p0: N([-2, 0], I) (左边)
        - 目标分布 p1: 双峰混合，均值在右上 [2, 2] 和右下 [2, -2]
    """

    mean0 = torch.tensor([-5.0, 0.0], device=device)
    mean1_up = torch.tensor([5.0, 10.0], device=device)
    mean1_down = torch.tensor([5.0, 10.0], device=device)
    std = torch.ones(2, device=device)

    x0 = mean0 + std * torch.randn(batch_size, 2, device=device)

    mix = torch.rand(batch_size, 1, device=device)
    mean1 = torch.where(mix < 0.5, mean1_up, mean1_down)
    x1 = mean1 + std * torch.randn(batch_size, 2, device=device)

    return x0, x1


def train_flow_matching(config: FlowDemoConfig):
    device = torch.device(config.device)
    model = MLP(input_dim=2 + 1, hidden_dims=config.model_hidden_dim, output_dim=2, activation="elu")
    flow = FlowMatcherBase(cfg=config.cfg, model=model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    for step in range(1, config.train_steps + 1):
        x0, x1 = sample_gaussians(config.batch_size, device)
        data = flow.train_forward(input=x0)
        loss = flow.compute_cfm_loss(data, x1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 200 == 0 or step == 1:
            print(f"[Train] step={step:04d}, loss={loss.item():.6f}")
    return model


@torch.no_grad()
def generate_samples(
    model: MLP,
    config: FlowDemoConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """使用训练好的 v_theta 从 p0 生成近似 p1 的样本。

    返回:
            x0_samples: 初始分布采样 (N, 2)
            xT_samples: 积分后样本 (最终时刻，近似目标分布) (N, 2)
                target_samples: 直接从目标高斯采样 (用于对比) (N, 2)
                    trajectories: 轨迹点 (N, T+1, 2)
    """
    device = next(model.parameters()).device
    num_steps = config.num_eval_samples
    dt = 1.0 / float(num_steps)

    x0, _ = sample_gaussians(config.num_eval_samples, device)
    # 注意：这里只用 p0 的样本作为起点
    x = x0.clone()

    # 记录轨迹（用于可视化）
    traj_samples = min(config.num_plot_samples, x.shape[0])
    trajectories = torch.zeros(traj_samples, num_steps + 1, 2, device=device)
    trajectories[:, 0, :] = x[:traj_samples]

    for i in range(num_steps):
        t_scalar = (i + 0.5) / float(num_steps)  # 用区间中心点时间
        t = torch.full((x.shape[0], 1), t_scalar, device=device)
        x_cat = torch.cat([x, t], dim=-1)
        v = model(x_cat)
        x = x + v * dt
        trajectories[:, i + 1, :] = x[:traj_samples]

    # 目标分布直接采样，方便画图对比
    _, x1_samples = sample_gaussians(config.num_eval_samples, device)
    return x0.cpu(), x.cpu(), x1_samples.cpu(), trajectories.cpu()


def main() -> None:
    config = FlowDemoConfig()
    print("===== 训练二维点 Flow Matching Demo =====")
    print(config)
    model = train_flow_matching(config)
    print("===== 训练完成，开始采样 =====")
    x0, xT, x1, trajectories = generate_samples(model, config)
    print("源分布样本均值:", x0.mean(dim=0))
    print("流匹配后样本均值:", xT.mean(dim=0))
    print("目标分布样本均值:", x1.mean(dim=0))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 6))
        plt.title("Flow Matching Paths (left -> right-up/right-down)")
        # 轨迹线（从左侧噪声点流向右上/右下）
        for i in range(trajectories.shape[0]):
            plt.plot(
                trajectories[i, :, 0],
                trajectories[i, :, 1],
                color="black",
                alpha=0.18,
                linewidth=0.7,
            )
        plt.scatter(x0[:, 0], x0[:, 1], s=8, alpha=0.6, label="start (p0)")
        plt.scatter(xT[:, 0], xT[:, 1], s=8, alpha=0.6, label="end (flow)")
        plt.scatter(x1[:, 0], x1[:, 1], s=8, alpha=0.4, label="target (p1)")
        plt.legend()
        plt.axis("equal")

        plt.tight_layout()
        import os

        os.makedirs("output", exist_ok=True)
        plt.savefig("output/flow_matching_result.png", dpi=200)
    except Exception as exc:  # matplotlib 不是必须
        print("绘图失败(可能未安装 matplotlib)，错误: ", exc)


if __name__ == "__main__":
    main()
