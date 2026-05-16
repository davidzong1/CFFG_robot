import torch
import torch.nn as nn


def guidance_fn(
    x_t: torch.Tensor,
    t: torch.Tensor,
    guide_model: nn.Module,
    guide_scale: float = 1.0,
) -> torch.Tensor:
    """
    计算引导梯度。
    Args:
        x_t: [B, token, D] 当前含噪样本，需要 requires_grad
        t: 当前时间步
        guide_model: 可微的引导函数/奖励模型，输入 x_t 输出标量 score
        guide_scale: 引导强度
    Returns:
        grad: [B, token, D] 引导梯度，与 x_t 同形状
    """
    x_t_input = x_t.detach().requires_grad_(True)
    # guide_model 输出 [B] 的标量打分
    score = guide_model(x_t_input, t).sum()
    grad = torch.autograd.grad(score, x_t_input)[0]
    return guide_scale * grad
