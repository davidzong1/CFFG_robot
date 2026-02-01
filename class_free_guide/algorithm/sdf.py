import torch
from typing import Tuple


class ObstacleInfo:
    _object: torch.Tensor
    _obstacles: dict[str, dict]

    def __init__(self):
        self._obstacles = {}

    def append(self, keys: str, local: torch.Tensor, radius: torch.Tensor, safe_distance: float = 0.0):
        """Append an obstacle to the obstacle list.\n
        keys: str, the key name of the obstacle
        local: Tensor of shape (N, time, 3)
        radius: Tensor of shape (N,)
        """
        assert local.shape[-1] == 3, "local should be of shape (N,3)"
        assert radius.shape[0] == local.shape[0], "radius should be of shape (N,)"
        self._obstacles[keys] = {"local": local, "radius": radius, "safe_distance": safe_distance}

    def update_object(self, object: torch.Tensor):
        """Set the object tensor.\n
        object: Tensor of shape (N, time, 3)
        """
        self._object = object

    def get_obstacles(self, keys: str) -> Tuple[torch.Tensor, torch.Tensor, float]:
        obs = self._obstacles[keys]
        return obs["local"], obs["radius"], obs["safe_distance"]

    @property
    def obstacles(self):
        return self._obstacles

    @property
    def object(self) -> torch.Tensor:
        return self._object


def cal_sdf_fn(obstacles: ObstacleInfo) -> torch.Tensor:
    """Calculate the signed distance field (SDF) for the given obstacles and object.
    obstacles: ObstacleInfo, the obstacle information
    return: Tensor of shape (N,), the signed distance field values for each point in the object
    """
    obj = obstacles.object
    d_list = []
    for key in obstacles.obstacles.keys():
        local, radius, sd = obstacles.get_obstacles(key)
        for i in range(local.shape[0]):
            center = local[..., i]
            r = radius[i]
            d = (obj - center).norm(dim=-1) - r - sd
            d_list.append(d.unsqueeze(-1))

    all_d = torch.cat(d_list, dim=-1)
    dmin, _ = all_d.min(dim=-1)
    return dmin


def cal_sdf_fn_no_min(obstacles: ObstacleInfo) -> torch.Tensor:
    """Calculate the signed distance field (SDF) for the given obstacles and object.
    obstacles: ObstacleInfo, the obstacle information
    return: Tensor of shape (N,), the signed distance field values for each point in the object
    """
    obj = obstacles.object
    d_list = []
    for key in obstacles.obstacles.keys():
        local, radius, sd = obstacles.get_obstacles(key)
        for i in range(local.shape[0]):
            center = local[..., i]
            r = radius[i]
            d = (obj - center).norm(dim=-1) - r - sd
            d_list.append(d.unsqueeze(-1))

    all_d = torch.cat(d_list, dim=-1)
    return all_d
