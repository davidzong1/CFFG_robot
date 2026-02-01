import torch
from class_free_guide.algorithm.sdf import cal_sdf_fn_no_min, ObstacleInfo
from class_free_guide.cost_manager.cost_base import CostBase


class CollisionCost(CostBase):
    scale: float
    info: ObstacleInfo

    def __init__(self, env, scale: float, **kwargs):
        super().__init__(env, **kwargs)
        self.scale = scale
        self.info = kwargs.get("collision_info")

    def forward(self, x):
        """
        Compute collision cost for each state in the batch.
        Args:
            x: Tensor of shape (B, D) representing batch of states.
        Returns:
            cost: Tensor of shape (B,) representing collision costs.
        """
        x = self.info.update_object(x)
        d = cal_sdf_fn_no_min(self.info)
        cost = torch.sum(torch.sum(torch.exp(-self.scale * d), dim=-1), dim=-1)
        return cost
