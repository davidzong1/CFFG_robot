from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
    unitree_go2_flat_env_cfg,
    robot_go_b_rough_env_cfg,
)
from .rl_cfg import (
    go_b_ppo_diffusion_runner_cfg,
    go_b_ppo_mlp_runner_cfg,
)

register_mjlab_task(
    task_id="Unitree-GoB-Rough",
    env_cfg=robot_go_b_rough_env_cfg(),
    play_env_cfg=robot_go_b_rough_env_cfg(play=True),
    rl_cfg=go_b_ppo_mlp_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Unitree-GoB-Flat",
    env_cfg=unitree_go2_flat_env_cfg(),
    play_env_cfg=unitree_go2_flat_env_cfg(play=True),
    rl_cfg=go_b_ppo_mlp_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
