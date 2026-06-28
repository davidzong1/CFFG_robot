"""Script to train RL agent with FPO (Flow Policy Optimization).

Ported from isaaclab_fpo/scripts/train.py and adapted for mjlab environments.

Usage:
    # Train Go2 on flat terrain with FPO
    python train_fpo.py Unitree-Go2-Flat-FPO

    # Train with custom parameters
    python train_fpo.py Unitree-Go2-Flat-FPO --num_envs 4096 --seed 42

    # Resume from checkpoint
    python train_fpo.py Unitree-Go2-Flat-FPO --resume --load_run "2024-01-01_12-00-00"

    # Multi-GPU training
    python train_fpo.py Unitree-Go2-Flat-FPO --gpu_ids 0 1 2 3
"""

from __future__ import annotations

import ast
import logging
import os
import random
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import tyro

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_runner_cls
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder

from class_free_guide.pineline.rl.rsl_rl.flow_ppo.config import (
    FpoRslRlOnPolicyRunnerCfg,
    FpoRslRlPpoActorCriticCfg,
    FpoRslRlPpoAlgorithmCfg,
)
from class_free_guide.pineline.rl.rsl_rl.flow_ppo.runner.on_policy_fpo_runner import (
    OnPolicyRunner as FpoOnPolicyRunner,
)

# ---------------------------------------------------------------------------
# FPO task config registry (go2 only, mirrors isaaclab_fpo/task_cfgs.py)
# ---------------------------------------------------------------------------

FPO_TASK_CONFIGS: dict[str, callable] = {}


def _register_fpo_tasks():
    """Register FPO task configurations.

    Called on module import to populate FPO_TASK_CONFIGS.
    Mirrors isaaclab_fpo/task_cfgs.py but only for Go2 robot.
    """
    from class_free_guide.pineline.rl.mjlab.tasks.velocity.config.go2.rl_cfg import (
        unitree_go2_fpo_runner_cfg,
    )

    # Unitree Go2 flat terrain FPO
    FPO_TASK_CONFIGS["Unitree-Go2-Flat-FPO"] = unitree_go2_fpo_runner_cfg

    # Unitree Go2 rough terrain FPO
    FPO_TASK_CONFIGS["Unitree-Go2-Rough-FPO"] = unitree_go2_fpo_runner_cfg


_register_fpo_tasks()


# ---------------------------------------------------------------------------
# CLI configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FpoTrainConfig:
    """Configuration for FPO training parsed from CLI."""

    env: ManagerBasedRlEnvCfg
    agent: FpoRslRlOnPolicyRunnerCfg
    video: bool = True
    video_length: int = 200
    video_interval: int = 5000
    enable_nan_guard: bool = False
    torchrunx_log_dir: str | None = None
    gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])
    # Override flags (from isaaclab_fpo/scripts/train.py CLI args)
    num_envs: int = 4096
    seed: int = 49
    max_iterations: int = 20000
    headless: bool = False
    # FPO-specific flags (from isaaclab_fpo/cli_args.py)
    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"
    experiment_name: str | None = None
    run_name: str | None = None
    logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
    log_project_name: str | None = None
    # TensorBoard local broadcasting: when set, starts a TensorBoard server
    # on this port binding to 0.0.0.0 so remote devices can monitor training.
    tb_port: int = 114514
    # LLM-driven reward-weight supervisor (rank 0, single-GPU runs only).
    supervisor: bool = False
    supervisor_config: str | None = None
    # Training-objective JSON path. CLI override beats supervisor_config.objective_path.
    objective: str | None = None

    @staticmethod
    def from_task(task_id: str) -> "FpoTrainConfig":
        """Create FpoTrainConfig from a registered task ID.

        Loads the environment config from mjlab registry and the FPO agent
        config from FPO_TASK_CONFIGS.
        """
        # Load environment config from mjlab task registry
        env_cfg = load_env_cfg(task_id)

        # Load FPO agent config from FPO task registry
        if task_id not in FPO_TASK_CONFIGS:
            # Fall back to default go2 FPO config
            from class_free_guide.pineline.rl.mjlab.tasks.velocity.config.go2.rl_cfg import (
                unitree_go2_fpo_runner_cfg,
            )

            agent_cfg = unitree_go2_fpo_runner_cfg()
        else:
            agent_cfg = FPO_TASK_CONFIGS[task_id]()

        return FpoTrainConfig(env=env_cfg, agent=agent_cfg)


# ---------------------------------------------------------------------------
# W&B sweep overrides (from isaaclab_fpo/scripts/train.py)
# ---------------------------------------------------------------------------


def _parse_sweep_overrides() -> tuple[dict, dict]:
    """Parse W&B sweep overrides from sys.argv positional args.

    W&B sweeps pass parameters as positional args like:
        agent.policy.actor_hidden_dims=[256,256,256]  agent.algorithm.clip_param=0.05

    Returns two dicts: one for 'agent.*' overrides and one for 'env.*' overrides.
    """
    agent_overrides: dict = {}
    env_overrides: dict = {}

    for arg in sys.argv[1:]:
        if "=" not in arg or arg.startswith("-"):
            continue

        key, value_str = arg.split("=", 1)

        # Parse the value
        try:
            value = ast.literal_eval(value_str)
        except (ValueError, SyntaxError):
            if value_str.lower() == "true":
                value = True
            elif value_str.lower() == "false":
                value = False
            else:
                value = value_str

        # Route to agent or env overrides
        if key.startswith("agent."):
            parts = key[len("agent.") :].split(".")
        elif key.startswith("env."):
            parts = key[len("env.") :].split(".")
        else:
            continue

        # Build nested dict from dotted path
        d = agent_overrides if key.startswith("agent.") else env_overrides
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = value

    return agent_overrides, env_overrides


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _start_tensorboard_server(logdir: str, port: int) -> None:
    """Start a TensorBoard server in a background daemon thread.

    Binds to 0.0.0.0 so remote devices on the same network can access the
    TensorBoard dashboard at http://<host-ip>:<port>.

    Args:
        logdir: Path to the TensorBoard log directory.
        port: Port number for the TensorBoard web server.
    """
    import threading

    from tensorboard import program

    def _run_tb():
        tb = program.TensorBoard()
        tb.configure(
            argv=[
                None,
                "--logdir",
                logdir,
                "--port",
                str(port),
                "--bind_all",
                "--reload_multifile",
                "true",
            ]
        )
        url = tb.launch()
        print(f"[INFO] TensorBoard server started at {url} (logdir: {logdir})")

    thread = threading.Thread(target=_run_tb, daemon=True)
    thread.start()


def run_fpo_train(task_id: str, cfg: FpoTrainConfig, log_dir: Path) -> None:
    """Run FPO training for a single process.

    Args:
        task_id: The task identifier (e.g. "Unitree-Go2-Flat-FPO").
        cfg: The training configuration.
        log_dir: Directory for logs and checkpoints.
    """
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible == "":
        # device = "cpu"
        # seed = cfg.agent.seed
        # rank = 0
        raise ValueError("Cannot found cuda device. Please check your CUDA_VISIBLE_DEVICES environment variable.")
    else:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        rank = int(os.environ.get("RANK", "0"))
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
        device = f"cuda:{local_rank}"
        seed = cfg.agent.seed + local_rank

    configure_torch_backends()

    # Update seeds for diversity across processes
    cfg.agent.seed = seed
    cfg.env.seed = seed

    print(f"[INFO] FPO Training with: device={device}, seed={seed}, rank={rank}")

    # Enable NaN guard if requested
    if cfg.enable_nan_guard:
        cfg.env.sim.nan_guard.enabled = True
        print(f"[INFO] NaN guard enabled, output dir: {cfg.env.sim.nan_guard.output_dir}")

    if rank == 0:
        print(f"[INFO] Logging experiment in directory: {log_dir}")

    _start_tensorboard_server(str(log_dir.parent), cfg.tb_port)

    # headless overrides video
    if cfg.headless:
        cfg.video = False

    # Create mjlab environment
    render_mode = "rgb_array" if cfg.video else None
    env = ManagerBasedRlEnv(cfg=cfg.env, device=device, render_mode=render_mode)

    log_root_path = log_dir.parent

    # Resolve resume path
    resume_path: Path | None = None
    if cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, cfg.load_run, cfg.load_checkpoint)

    # Video recording (rank 0 only)
    if cfg.video and rank == 0:
        env = VideoRecorder(
            env,
            video_folder=Path(log_dir) / "videos" / "train",
            step_trigger=lambda step: step % cfg.video_interval == 0,
            video_length=cfg.video_length,
            disable_logger=True,
        )
        print("[INFO] Recording videos during training.")

    # Wrap environment for FPO runner (same interface as isaaclab FpoRslRlVecEnvWrapper)
    env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)

    # Create FPO runner with the FPO config dataclass
    runner = FpoOnPolicyRunner(env, cfg.agent, log_dir=str(log_dir), device=device)

    runner.add_git_repo_to_log(__file__)

    if resume_path is not None:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(str(resume_path))

    # Dump configs to log directory (rank 0 only)
    if rank == 0:
        dump_yaml(log_dir / "params" / "env.yaml", asdict(cfg.env))
        dump_yaml(log_dir / "params" / "agent.yaml", asdict(cfg.agent))

    # Optional LLM reward supervisor (rank 0 only).
    supervisor = None
    if cfg.supervisor and rank == 0:
        if cfg.headless:
            raise ValueError("[ ERROR ] When using LLM automatic parameter tuning, the headless mode cannot be activated!")
        try:
            from class_free_guide.pineline.rl.supervisor import (
                Supervisor,
                SupervisorConfig,
            )

            sup_cfg_path = cfg.supervisor_config or str(Path(__file__).resolve().parents[1] / "supervisor" / "config" / "supervisor.yaml")
            sup_cfg = SupervisorConfig.load(sup_cfg_path)
            # CLI --objective overrides whatever the supervisor yaml said.
            if cfg.objective:
                sup_cfg.objective_path = cfg.objective
            supervisor = Supervisor(
                env=env,
                writer=None,
                log_dir=log_dir,
                config=sup_cfg,
                writer_getter=lambda: getattr(runner, "writer", None),
                iter_getter=lambda: int(getattr(runner, "current_learning_iteration", 0)),
            )
            supervisor.start()
            print(
                f"[INFO] Reward supervisor enabled "
                f"(provider={sup_cfg.provider}, interval={sup_cfg.interval_min:.0f}min, "
                f"objective={supervisor.objective.name})."
            )
        except Exception as e:
            print(f"[WARN] Failed to start reward supervisor: {e}")
            supervisor = None

    # Determine number of training iterations
    num_learning_iterations = cfg.agent.max_iterations
    print(f"[INFO] Training for {num_learning_iterations} iterations (max_iterations={cfg.agent.max_iterations})")

    # Run training
    try:
        runner.learn(
            num_learning_iterations=num_learning_iterations,
            init_at_random_ep_len=True,
        )
    finally:
        if supervisor is not None:
            supervisor.stop()

    env.close()


def launch_fpo_training(task_id: str, args: FpoTrainConfig | None = None):
    """Launch FPO training, handling single-GPU and multi-GPU modes.

    Args:
        task_id: The task identifier.
        args: The training configuration. If None, loads from task registry.
    """
    args = args or FpoTrainConfig.from_task(task_id)

    # Update agent config from CLI overrides
    if args.experiment_name:
        args.agent.experiment_name = args.experiment_name
    if args.run_name:
        args.agent.run_name = args.run_name
    if args.resume:
        args.agent.resume = args.resume
        args.agent.load_run = args.load_run
        args.agent.load_checkpoint = args.load_checkpoint
    if args.logger:
        args.agent.logger = args.logger
    if args.log_project_name and args.agent.logger in ("wandb", "neptune"):
        args.agent.wandb_project = args.log_project_name
        args.agent.neptune_project = args.log_project_name

    # Apply CLI overrides (mirrors isaaclab_fpo/scripts/train.py)
    # --num_envs: override the number of parallel environments
    if args.num_envs is not None:
        args.env.scene.num_envs = args.num_envs
    # --seed: override random seed (-1 for random seed)
    if args.seed is not None:
        if args.seed == -1:
            args.agent.seed = random.randint(0, 10000)
        else:
            args.agent.seed = args.seed
    # --max_iterations: override training iterations
    if args.max_iterations is not None:
        args.agent.max_iterations = args.max_iterations

    # Apply W&B sweep overrides
    agent_overrides, env_overrides = _parse_sweep_overrides()
    if agent_overrides:
        _apply_dict_overrides(args.agent, agent_overrides)
    if env_overrides:
        _apply_dict_overrides(args.env, env_overrides)

    # Create log directory
    log_root_path = Path("logs") / "rsl_rl" / args.agent.experiment_name
    log_root_path.resolve()
    log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.agent.run_name:
        log_dir_name += f"_{args.agent.run_name}"
    log_dir = log_root_path / log_dir_name

    # Select GPUs
    selected_gpus, num_gpus = select_gpus(args.gpu_ids)

    # Set environment variables
    if selected_gpus is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))

    # Headless mode: use EGL/OSMesa backend and disable viewer
    if args.headless:
        os.environ["MUJOCO_GL"] = "egl"
        # Disable GPU rendering output (X11/GLX bypass)
        os.environ.pop("DISPLAY", None)
        # Disable the viewer in env config if present
        if hasattr(args.env, "viewer") and args.env.viewer is not None:
            args.env.viewer = None
        print("[INFO] Headless mode enabled: MuJoCo GL backend set to EGL, viewer disabled.")

    if num_gpus <= 1:
        # CPU or single GPU
        run_fpo_train(task_id, args, log_dir)
    else:
        # Multi-GPU via torchrunx
        import torchrunx

        logging.basicConfig(level=logging.INFO)

        if "TORCHRUNX_LOG_DIR" not in os.environ:
            if args.torchrunx_log_dir is not None:
                os.environ["TORCHRUNX_LOG_DIR"] = args.torchrunx_log_dir
            else:
                os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")

        print(f"[INFO] Launching FPO training with {num_gpus} GPUs", flush=True)
        torchrunx.Launcher(
            hostnames=["localhost"],
            workers_per_host=num_gpus,
            backend=None,  # Let rsl_rl handle process group initialization
            copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
        ).run(run_fpo_train, task_id, args, log_dir)


def _apply_dict_overrides(target, overrides: dict) -> None:
    """Apply nested dict overrides to a dataclass instance.

    Args:
        target: The dataclass instance to update.
        overrides: Nested dict of attribute overrides.
    """
    for key, value in overrides.items():
        if isinstance(value, dict) and hasattr(target, key):
            _apply_dict_overrides(getattr(target, key), value)
        elif hasattr(target, key):
            setattr(target, key, value)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main():
    """Main entry point for FPO training script."""
    # Import tasks to populate the mjlab registry
    import mjlab.tasks  # noqa: F401
    from class_free_guide.pineline.rl.mjlab import tasks  # noqa: F401

    # Build list of available tasks: mjlab task IDs + FPO task IDs
    mjlab_tasks = list_tasks()
    all_tasks = sorted(set(mjlab_tasks) | set(FPO_TASK_CONFIGS.keys()))

    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
    )

    args = tyro.cli(
        FpoTrainConfig,
        args=remaining_args,
        default=FpoTrainConfig.from_task(chosen_task),
        prog=sys.argv[0] + f" {chosen_task}",
    )

    launch_fpo_training(task_id=chosen_task, args=args)


if __name__ == "__main__":
    main()
