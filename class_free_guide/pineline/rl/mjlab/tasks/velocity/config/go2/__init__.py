from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import (
    unitree_go2_flat_env_cfg,
    unitree_go2_rough_env_cfg,
)
from .rl_cfg import unitree_go2_ppo_runner_cfg, unitree_go2_fpo_runner_cfg

# Import standard PPO runner (VelocityOnPolicyRunner was moved/deleted; fall back to MjlabOnPolicyRunner).
try:
    from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
except ImportError:
    try:
        from mjlab.rl import MjlabOnPolicyRunner as VelocityOnPolicyRunner
    except ImportError:
        VelocityOnPolicyRunner = None  # type: ignore[assignment]

# Import FPO runner for FPO task configurations.
from class_free_guide.pineline.rl.rsl_rl.flow_ppo.runner.on_policy_fpo_runner import (
    OnPolicyRunner as FpoOnPolicyRunner,
)


# ---------------------------------------------------------------------------
# PPO task registrations
# ---------------------------------------------------------------------------

_ppo_runner_kwargs = {}
if VelocityOnPolicyRunner is not None:
    _ppo_runner_kwargs["runner_cls"] = VelocityOnPolicyRunner

register_mjlab_task(
    task_id="Unitree-Go2-Rough",
    env_cfg=unitree_go2_rough_env_cfg(),
    play_env_cfg=unitree_go2_rough_env_cfg(play=True),
    rl_cfg=unitree_go2_ppo_runner_cfg(),
    **_ppo_runner_kwargs,
)

register_mjlab_task(
    task_id="Unitree-Go2-Flat",
    env_cfg=unitree_go2_flat_env_cfg(),
    play_env_cfg=unitree_go2_flat_env_cfg(play=True),
    rl_cfg=unitree_go2_ppo_runner_cfg(),
    **_ppo_runner_kwargs,
)

# ---------------------------------------------------------------------------
# FPO task registrations (ported from isaaclab_fpo/task_cfgs.py)
# ---------------------------------------------------------------------------

register_mjlab_task(
    task_id="Unitree-Go2-Rough-FPO",
    env_cfg=unitree_go2_rough_env_cfg(),
    play_env_cfg=unitree_go2_rough_env_cfg(play=True),
    rl_cfg=unitree_go2_fpo_runner_cfg(),
    runner_cls=FpoOnPolicyRunner,
)

register_mjlab_task(
    task_id="Unitree-Go2-Flat-FPO",
    env_cfg=unitree_go2_flat_env_cfg(),
    play_env_cfg=unitree_go2_flat_env_cfg(play=True),
    rl_cfg=unitree_go2_fpo_runner_cfg(),
    runner_cls=FpoOnPolicyRunner,
)
