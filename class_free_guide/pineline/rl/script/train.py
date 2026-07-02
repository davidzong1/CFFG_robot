"""Script to train RL agent with RSL-RL."""

from __future__ import annotations

import ast
import inspect
import logging
import os
import random
import signal
import sys
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

import torch

# ---------------------------------------------------------------------------
# Headless rendering: MUST set MUJOCO_GL BEFORE any MuJoCo import.
# MuJoCo loads the GL backend at import time; setting the env var after
# imports have already completed has no effect and GLFW will fail when
# no X11 display is available.
# ---------------------------------------------------------------------------
if "--headless" in sys.argv:
    os.environ["MUJOCO_GL"] = "egl"

import tyro

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder

from class_free_guide.supervisor import (
    Supervisor,
    SupervisorConfig,
)


@dataclass(frozen=True)
class TrainConfig:
    env: ManagerBasedRlEnvCfg
    agent: RslRlBaseRunnerCfg
    motion_file: str | None = None
    video: bool = False
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
    tb_port: int = 11451
    # LLM-driven reward-weight supervisor (rank 0, single-GPU runs only).
    supervisor: bool = False
    supervisor_config: str | None = None
    # Training-objective JSON path. CLI override beats supervisor_config.objective_path.
    objective: str | None = None
    # Debug escape hatch for MuJoCo/EGL teardown segfaults in headless runs.
    skip_env_close: bool = False
    # Avoid MuJoCo/EGL/Python native destructor crashes after successful
    # headless supervisor runs. Only used for single-process runs.
    fast_exit_on_headless_supervisor: bool = True

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

    # Override MASTER_PORT with the port we allocated in the launcher
    # process (same rationale as train_fpo.py).
    if "_NCCL_MASTER_PORT" in os.environ:
        os.environ["MASTER_PORT"] = os.environ["_NCCL_MASTER_PORT"]
        print(f"[INFO] Overriding MASTER_PORT to {os.environ['MASTER_PORT']} (via _NCCL_MASTER_PORT)")

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

    # Create mjlab environment.
    # EGL backend supports offscreen rendering; video recording works
    # in headless mode as long as MUJOCO_GL=egl is set before import.
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

    # If torchrunx created the process group (backend != None), patch
    # rsl_rl's _configure_multi_gpu to avoid calling init_process_group twice.
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        from rsl_rl.runners.on_policy_runner import OnPolicyRunner as _RslPolicyRunner
        if not hasattr(_RslPolicyRunner, "_torchrunx_patched"):

            def _safe_configure_multi_gpu(self):
                self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
                self.is_distributed = self.gpu_world_size > 1
                if not self.is_distributed:
                    self.gpu_local_rank = 0
                    self.gpu_global_rank = 0
                    self.cfg["multi_gpu"] = None
                    return
                self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
                self.gpu_global_rank = int(os.getenv("RANK", "0"))
                self.cfg["multi_gpu"] = {
                    "global_rank": self.gpu_global_rank,
                    "local_rank": self.gpu_local_rank,
                    "world_size": self.gpu_world_size,
                }
                if self.device != f"cuda:{self.gpu_local_rank}":
                    raise ValueError(
                        f"Device '{self.device}' does not match expected device "
                        f"for local rank '{self.gpu_local_rank}'."
                    )
                if self.gpu_local_rank >= self.gpu_world_size:
                    raise ValueError(
                        f"Local rank '{self.gpu_local_rank}' is greater than or "
                        f"equal to world size '{self.gpu_world_size}'."
                    )
                if self.gpu_global_rank >= self.gpu_world_size:
                    raise ValueError(
                        f"Global rank '{self.gpu_global_rank}' is greater than or "
                        f"equal to world size '{self.gpu_world_size}'."
                    )
                if not torch.distributed.is_initialized():
                    torch.distributed.init_process_group(
                        backend="nccl",
                        rank=self.gpu_global_rank,
                        world_size=self.gpu_world_size,
                    )
                torch.cuda.set_device(self.gpu_local_rank)

            _RslPolicyRunner._configure_multi_gpu = _safe_configure_multi_gpu
            _RslPolicyRunner._torchrunx_patched = True

    # Load supervisor configuration unconditionally, like train_fpo.py does,
    # so it is available for both the runner constructor and the supervisor.
    sup_cfg_path = cfg.supervisor_config or str(Path(__file__).resolve().parents[1] / "supervisor" / "config" / "supervisor.yaml")
    sup_cfg = SupervisorConfig.load(sup_cfg_path)

    runner_cls = load_runner_cls(task_id)
    if runner_cls is None:
        runner_cls = MjlabOnPolicyRunner

    # Pass supervisor_cfg to runner constructor when the runner supports it
    # (e.g. VelocityOnPolicyRunner from flow_ppo, FpoOnPolicyRunner).
    runner_kwargs = {}
    sig = inspect.signature(runner_cls.__init__)
    if "supervisor_cfg" in sig.parameters or "sup_cfg" in sig.parameters:
        runner = runner_cls(env, agent_cfg, sup_cfg, str(log_dir), device, **runner_kwargs)
    else:
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
    is_supervisor = cfg.supervisor
    if cfg.supervisor and rank == 0:
        try:
            # CLI --objective overrides whatever the supervisor yaml said.
            if cfg.objective:
                sup_cfg.objective_path = cfg.objective
            stopping_event = threading.Event()
            supervisor = Supervisor.from_rl_env(
                env=env,
                writer=None,
                log_dir=log_dir,
                config=sup_cfg,
                writer_getter=lambda: getattr(runner, "writer", None),
                iter_getter=lambda: int(getattr(runner, "current_learning_iteration", 0)),
                stopping_event=stopping_event,
            )
            supervisor.start()

            # Prefer clean early_stop_event when the runner supports it
            # (e.g. VelocityOnPolicyRunner from flow_ppo, FpoOnPolicyRunner).
            # Fall back to SIGINT for runners without built-in support.
            if hasattr(runner, "early_stop_event"):
                runner.early_stop_event = stopping_event

                # Multi-GPU: write a .stop file in the shared log directory so all
                # ranks detect the stop signal (threading.Event is process-local).
                def _write_stop_file():
                    stopping_event.wait()
                    stop_path = Path(log_dir) / ".stop"
                    stop_path.touch()
                    print("[INFO] Wrote .stop file for multi-GPU stop propagation.")

                threading.Thread(target=_write_stop_file, daemon=True).start()
            else:
                def _early_stop_monitor() -> None:
                    stopping_event.wait()
                    os.kill(os.getpid(), signal.SIGINT)

                threading.Thread(target=_early_stop_monitor, daemon=True).start()

            print(
                f"[INFO] Reward supervisor enabled "
                f"(provider={sup_cfg.provider}, interval={sup_cfg.interval_min:.0f}min, "
                f"objective={supervisor.objective.name})."
            )
        except Exception as e:
            print(f"[WARN] Failed to start reward supervisor: {e}")
            supervisor = None

    # Pre-flight LLM connectivity check (rank 0 only, supervisor enabled).
    # Runs BEFORE the training loop so we don't waste GPU hours on a run
    # whose supervisor can never reach the LLM.  A timeout or connectivity
    # error here causes the script to exit with a non-zero code.
    if supervisor is not None and rank == 0:
        print(
            f"[INFO] Testing LLM connectivity "
            f"(provider={sup_cfg.provider}, model={sup_cfg.model})...",
            flush=True,
        )
        if not supervisor.llm.test_connection(timeout=30.0):
            raise RuntimeError(
                f"Supervisor LLM connectivity check FAILED. "
                f"The {sup_cfg.provider} model '{sup_cfg.model}' is unreachable. "
                f"Check your API key (env: {sup_cfg.api_key_env or 'N/A'}), "
                f"network, and model name. "
                f"To skip the supervisor, remove --supervisor from the CLI."
            )
        if sup_cfg.only_text:
            print("[INFO] LLM connectivity OK (text-only mode, vision disabled).", flush=True)
        else:
            print("[INFO] LLM connectivity OK (vision supported).", flush=True)

    real_max_iterations = cfg.agent.max_iterations if not is_supervisor else 10000000
    num_learning_iterations = real_max_iterations
    print(f"[INFO] Training for {num_learning_iterations} iterations (max_iterations={real_max_iterations})")

    try:
        runner.learn(num_learning_iterations=num_learning_iterations, init_at_random_ep_len=True)
    finally:
        print("[INFO] Cleanup: stopping supervisor...", flush=True)
        if supervisor is not None:
            supervisor.stop()
        print("[INFO] Cleanup: closing runner...", flush=True)
        if hasattr(runner, "close"):
            runner.close()
        print("[INFO] Cleanup: runner closed.", flush=True)

    if (
        cfg.fast_exit_on_headless_supervisor
        and cfg.headless
        and cfg.supervisor
        and int(os.environ.get("WORLD_SIZE", "1")) <= 1
    ):
        print(
            "[WARN] Cleanup: fast process exit after successful headless supervisor run; "
            "skipping env.close() to avoid native MuJoCo/EGL teardown segfault.",
            flush=True,
        )
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    if cfg.skip_env_close:
        print("[WARN] Cleanup: skipping env.close() because --skip_env_close is set.", flush=True)
    else:
        print("[INFO] Cleanup: closing env...", flush=True)
        env.close()
        print("[INFO] Cleanup: env closed.", flush=True)


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

    if args.supervisor and not args.video:
        object.__setattr__(args, "video", True)
        print("[INFO] Supervisor enabled: forcing video recording on for multimodal frame sampling.")

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

    # Headless mode: use EGL offscreen rendering and avoid X11/GLX.
    if args.headless:
        os.environ["MUJOCO_GL"] = "egl"
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        # Disable GPU rendering output (X11/GLX bypass)
        os.environ.pop("DISPLAY", None)
        # Offscreen video rendering needs viewer config for camera/resolution.
        # Only drop it when video is disabled.
        if not args.video and hasattr(args.env, "viewer") and args.env.viewer is not None:
            args.env.viewer = None
        print(
            f"[INFO] Headless mode enabled: MuJoCo GL backend set to EGL "
            f"(video={'enabled' if args.video else 'disabled'})."
        )

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
        # Pick a free port for NCCL TCPStore so it cannot collide with
        # torchrunx's own internal communication ports.
        import socket as _socket

        _s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        _s.bind(("", 0))
        _nccl_port = _s.getsockname()[1]
        _s.close()

        torchrunx.Launcher(
            hostnames=["localhost"],
            workers_per_host=num_gpus,
            backend=None,  # Let our code handle init_process_group with correct device ordering
            copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
            extra_env_vars={"_NCCL_MASTER_PORT": str(_nccl_port)},
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
