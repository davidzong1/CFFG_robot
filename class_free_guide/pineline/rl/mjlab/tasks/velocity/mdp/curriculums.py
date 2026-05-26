from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .velocity_command import UniformVelocityCommandCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")

change_level_threshold = 50  # 50轮存活率阈值，超过则增加难度，低于则降低难度


class VelocityStage(TypedDict):
    step: int
    lin_vel_x: tuple[float, float] | None
    lin_vel_y: tuple[float, float] | None
    ang_vel_z: tuple[float, float] | None


class RewardWeightStage(TypedDict):
    step: int
    weight: float


def terrain_levels_vel(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]

    terrain = env.scene.terrain
    assert terrain is not None
    terrain_generator = terrain.cfg.terrain_generator
    assert terrain_generator is not None

    command = env.command_manager.get_command(command_name)
    assert command is not None

    # Compute the distance the robot walked.
    distance = torch.norm(asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)

    # Robots that walked far enough progress to harder terrains.
    move_up = distance > terrain_generator.size[0] / 2

    # Robots that walked less than half of their required distance go to simpler
    # terrains.
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up

    # Update terrain levels.
    terrain.update_env_origins(env_ids, move_up, move_down)

    return torch.mean(terrain.terrain_levels.float())


def commands_vel(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    velocity_stages: list[VelocityStage],
) -> dict[str, torch.Tensor]:
    del env_ids  # Unused.
    global change_level_threshold
    command_term = env.command_manager.get_term(command_name)
    assert command_term is not None
    cfg = cast(UniformVelocityCommandCfg, command_term.cfg)
    if not hasattr(env, "vel_level_cnt"):
        env.vel_level = float(1.0)
        env.vel_level_cnt = 0
    if not hasattr(env, "commands_vel_base_ranges"):
        env.commands_vel_base_ranges = {  # type: ignore[attr-defined]
            "lin_vel_x": cfg.ranges.lin_vel_x,
            "lin_vel_y": cfg.ranges.lin_vel_y,
            "ang_vel_z": cfg.ranges.ang_vel_z,
        }
    base_ranges: dict = env.commands_vel_base_ranges  # type: ignore[attr-defined]

    # 阶段化课程更新：将当前训练步数对应的阶段范围写入 base_ranges（而非 cfg.ranges）
    for stage in velocity_stages:
        if env.common_step_counter > stage["step"]:
            if "lin_vel_x" in stage and stage["lin_vel_x"] is not None:
                base_ranges["lin_vel_x"] = stage["lin_vel_x"]
            if "lin_vel_y" in stage and stage["lin_vel_y"] is not None:
                base_ranges["lin_vel_y"] = stage["lin_vel_y"]
            if "ang_vel_z" in stage and stage["ang_vel_z"] is not None:
                base_ranges["ang_vel_z"] = stage["ang_vel_z"]

    # ============ 基于存活率计算 env.command_scale ============
    # 存活率定义：所有并行环境的平均"已存活步数 / 最大步数"
    #   - 接近 1 → 机器人普遍能撑满整个 episode → 训练效果好，放大速度范围以提升难度
    #   - 接近 0 → 机器人很快失败 → 训练困难，缩小速度范围让任务更可学
    # 下限取 0.1，避免训练初期 survival_rate≈0 时把指令范围压成 0 导致机器人完全无指令
    max_ep_len = max(float(env.max_episode_length), 1.0)
    survival_rate = (env.episode_length_buf.float().mean() / max_ep_len).clamp(min=0.0, max=1.0).item()
    env.vel_level_cnt += 1 if survival_rate > 0.85 else -1 if survival_rate < 0.3 else 0
    if env.vel_level_cnt > change_level_threshold:
        env.vel_level_cnt = 0
        env.vel_level = min(env.vel_level + 0.1, 1.0)
    elif env.vel_level_cnt < -change_level_threshold:
        env.vel_level_cnt = 0
        env.vel_level = max(env.vel_level - 0.1, 0.1)
    s = env.vel_level
    # 把基础范围乘以 command_scale，得到当前 episode 实际使用的最大速度指令范围
    cfg.ranges.lin_vel_x = (base_ranges["lin_vel_x"][0] * s, base_ranges["lin_vel_x"][1] * s)
    cfg.ranges.lin_vel_y = (base_ranges["lin_vel_y"][0] * s, base_ranges["lin_vel_y"][1] * s)
    cfg.ranges.ang_vel_z = (base_ranges["ang_vel_z"][0] * s, base_ranges["ang_vel_z"][1] * s)
    # =========================================================

    return {
        # "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0]),
        # "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1]),
        # "lin_vel_y_min": torch.tensor(cfg.ranges.lin_vel_y[0]),
        # "lin_vel_y_max": torch.tensor(cfg.ranges.lin_vel_y[1]),
        # "ang_vel_z_min": torch.tensor(cfg.ranges.ang_vel_z[0]),
        # "ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1]),
    }


def reward_weight(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    reward_name: str,
    weight_stages: list[RewardWeightStage],
) -> torch.Tensor:
    """Update a reward term's weight based on training step stages."""
    del env_ids  # Unused.
    reward_term_cfg = env.reward_manager.get_term_cfg(reward_name)
    for stage in weight_stages:
        if env.common_step_counter > stage["step"]:
            reward_term_cfg.weight = stage["weight"]
    return torch.tensor([reward_term_cfg.weight])
