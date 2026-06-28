"""Script to train RL agent with RSL-RL."""

import ast
import logging
import os
import random
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

import tyro

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder


@dataclass(frozen=True)
class TrainConfig:
    env: ManagerBasedRlEnvCfg
    agent: RslRlBaseRunnerCfg
    motion_file: str | None = None
    video: bool = True
    video_length: int = 200
    video_interval: int = 5000
    enable_nan_guard: bool = False
    torchrunx_log_dir: str | None = None
    gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])
    # Override flags
    num_envs: int = 4096
    seed: int = 49
    max_iterations: int = 20000
    headless: bool = False
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
    def from_task(task_id: str) -> "TrainConfig":
        env_cfg = load_env_cfg(task_id)
        agent_cfg = load_rl_cfg(task_id)
        return TrainConfig(env=env_cfg, agent=agent_cfg)


# ---------------------------------------------------------------------------
# W&B sweep overrides
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


def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible == "":
        raise ValueError("Cannot found cuda device. Please check your CUDA_VISIBLE_DEVICES environment variable.")
    else:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        rank = int(os.environ.get("RANK", "0"))
        # Set EGL device to match the CUDA device.
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
        device = f"cuda:{local_rank}"
        # Set seed to have diversity in different processes.
        seed = cfg.agent.seed + local_rank

    configure_torch_backends()

    cfg.agent.seed = seed
    cfg.env.seed = seed

    print(f"[INFO] Training with: device={device}, seed={seed}, rank={rank}")

    # Check if this is a tracking task by checking for motion command.
    is_tracking_task = "motion" in cfg.env.commands and isinstance(cfg.env.commands["motion"], MotionCommandCfg)

    if is_tracking_task:
        if not cfg.motion_file:
            raise ValueError("For tracking tasks, --motion-file must be set ...")
        motion_path = Path(cfg.motion_file).expanduser().resolve()
        if not motion_path.exists():
            raise FileNotFoundError(f"Motion file not found: {motion_path}")
        motion_cmd = cfg.env.commands["motion"]
        assert isinstance(motion_cmd, MotionCommandCfg)
        motion_cmd.motion_file = str(motion_path)
        print(f"[INFO] Using motion file: {motion_cmd.motion_file}")

        # Check if motion_file is already set (e.g., via CLI --env.commands.motion.motion-file).
        if motion_cmd.motion_file and Path(motion_cmd.motion_file).exists():
            print(f"[INFO] Using local motion file: {motion_cmd.motion_file}")

    # Enable NaN guard if requested.
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

    log_root_path = log_dir.parent  # Go up from specific run dir to experiment dir.

    resume_path: Path | None = None
    if cfg.agent.resume:
        # Load checkpoint from local filesystem.
        resume_path = get_checkpoint_path(log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint)

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

    env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)

    agent_cfg = asdict(cfg.agent)
    env_cfg = asdict(cfg.env)

    runner_cls = load_runner_cls(task_id)
    if runner_cls is None:
        runner_cls = MjlabOnPolicyRunner

    runner_kwargs = {}
    runner = runner_cls(env, agent_cfg, str(log_dir), device, **runner_kwargs)

    runner.add_git_repo_to_log(__file__)
    if resume_path is not None:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(str(resume_path))

    # Only write config files from rank 0 to avoid race conditions.
    if rank == 0:
        dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
        dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)

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

    try:
        runner.learn(num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True)
    finally:
        if supervisor is not None:
            supervisor.stop()

    env.close()


def launch_training(task_id: str, args: TrainConfig | None = None):
    args = args or TrainConfig.from_task(task_id)

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

    # Apply CLI overrides
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

    # Create log directory once before launching workers.
    log_root_path = Path("logs") / "rsl_rl" / args.agent.experiment_name
    log_root_path.resolve()
    log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.agent.run_name:
        log_dir_name += f"_{args.agent.run_name}"
    log_dir = log_root_path / log_dir_name

    # Select GPUs based on CUDA_VISIBLE_DEVICES and user specification.
    selected_gpus, num_gpus = select_gpus(args.gpu_ids)

    # Set environment variables for all modes.
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
        # CPU or single GPU: run directly without torchrunx.
        run_train(task_id, args, log_dir)
    else:
        # Multi-GPU: use torchrunx.
        import torchrunx

        # torchrunx redirects stdout to logging.
        logging.basicConfig(level=logging.INFO)

        # Configure torchrunx logging directory.
        # Priority: 1) existing env var, 2) user flag, 3) default to {log_dir}/torchrunx.
        if "TORCHRUNX_LOG_DIR" not in os.environ:
            if args.torchrunx_log_dir is not None:
                # User specified a value via flag (could be "" to disable).
                os.environ["TORCHRUNX_LOG_DIR"] = args.torchrunx_log_dir
            else:
                # Default: put logs in training directory.
                os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")

        print(f"[INFO] Launching training with {num_gpus} GPUs", flush=True)
        torchrunx.Launcher(
            hostnames=["localhost"],
            workers_per_host=num_gpus,
            backend=None,  # Let rsl_rl handle process group initialization.
            copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
        ).run(run_train, task_id, args, log_dir)


def main():
    # Parse first argument to choose the task.
    # Import tasks to populate the registry.
    import mjlab.tasks  # noqa: F401
    from class_free_guide.pineline.rl.mjlab import tasks

    all_tasks = list_tasks()

    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
    )

    args = tyro.cli(
        TrainConfig,
        args=remaining_args,
        default=TrainConfig.from_task(chosen_task),
        prog=sys.argv[0] + f" {chosen_task}",
    )

    del remaining_args

    launch_training(task_id=chosen_task, args=args)


if __name__ == "__main__":
    main()
