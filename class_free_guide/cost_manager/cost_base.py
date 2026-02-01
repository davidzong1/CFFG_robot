import torch
import torch.nn as nn
import abc


class CostBase(abc.ABC):
    """
    Base class for cost functions.\n
    Generate cost flow maps and their gradients.
    """

    def __init__(self, env, **kwargs):
        super().__init__()
        self.env = env

    @abc.abstractmethod
    def forward(self, x):
        raise NotImplementedError

    def gradients(self, x):
        x.requires_grad_()
        y = self(x)
        # y.sum() is a surrogate to compute gradients of independent quantities over the batch dimension
        grad = torch.autograd.grad([y.sum()], [x])[0]
        x.detach_()
        return y, grad
