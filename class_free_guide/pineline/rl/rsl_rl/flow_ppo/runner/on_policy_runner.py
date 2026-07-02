import os
from pathlib import Path

import wandb

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
    attach_metadata_to_onnx,
    get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner
import time
from ..utils.logger import Logger
from class_free_guide.supervisor import SupervisorConfig
import dzipc
import torch
from rsl_rl.utils import check_nan, resolve_callable


class VelocityOnPolicyRunner(MjlabOnPolicyRunner):
    env: RslRlVecEnvWrapper

    def __init__(
        self,
        env: RslRlVecEnvWrapper,
        train_cfg: dict,
        supervisor_cfg: SupervisorConfig,
        log_dir: str | None = None,
        device: str = "cpu",
    ) -> None:
        super().__init__(env, train_cfg, log_dir, device)
        self.supervisor_cfg = supervisor_cfg
        # Replace parent's rsl_rl Logger with our local Logger so that
        # all logging is channeled through the flow_ppo Logger class.
        self.logger = Logger(
            log_dir=log_dir,
            cfg=train_cfg,
            env_cfg=env.cfg if hasattr(env, "cfg") else None,
            num_envs=env.num_envs,
            is_distributed=self.is_distributed,
            gpu_world_size=self.gpu_world_size,
            gpu_global_rank=self.gpu_global_rank,
            device=self.device,
        )
        self.early_stop_event = None
        self._ipc_msg_template = None
        self._ipc_sub = None
        self._ipc_topic_data = None
        self._init_ipc_subscriber()

    def _init_ipc_subscriber(self) -> None:
        """Create a SHM subscriber so external tooling can watch supervisor state."""
        if not self.supervisor_cfg.ipc_enabled:
            return
        try:
            self._ipc_msg_template = dzipc.Supervisor()
            self._ipc_topic_data = dzipc.make_topic_data(self._ipc_msg_template)
            self._ipc_sub = dzipc.SubscriberIPCPtrMake(
                self._ipc_topic_data,
                self.supervisor_cfg.ipc_topic,
                self.supervisor_cfg.ipc_domain,
                10,
                dzipc.IPC_SHM,
                verbose=False,
            )
            self._ipc_sub.InitChannel()
        except Exception as exc:
            self._ipc_sub = None
            self._ipc_msg_template = None
            self._ipc_topic_data = None

    def close(self) -> None:
        self.logger.stop_logging_writer()
        self._close_ipc()

    def _close_ipc(self) -> None:
        self._ipc_sub = None
        self._ipc_topic_data = None
        self._ipc_msg_template = None
        cleanup = getattr(dzipc, "CleanupIpcInstances", None)
        if cleanup is not None:
            try:
                cleanup()
            except Exception:
                pass

    def _configure_multi_gpu(self):
        """Override parent to skip init_process_group when torchrunx already
        initialized the distributed process group (backend="nccl" in Launcher)."""
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
            raise ValueError(f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'.")
        if self.gpu_local_rank >= self.gpu_world_size:
            raise ValueError(f"Local rank '{self.gpu_local_rank}' is greater than or equal to world size '{self.gpu_world_size}'.")
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(f"Global rank '{self.gpu_global_rank}' is greater than or equal to world size '{self.gpu_world_size}'.")

        # torchrunx already called init_process_group; skip duplicate init.
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size)
        torch.cuda.set_device(self.gpu_local_rank)

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Run the learning loop for the specified number of iterations."""
        # Randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))

        # Start learning
        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()  # switch to train mode (for dropout for example)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Initialize the logging writer
        self.logger.init_logging_writer()

        # Start training
        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        topic_msg = None
        for it in range(start_it, total_it):
            if self.early_stop_event is not None and self.early_stop_event.is_set():
                print(f"[FPO runner] Early stop at iteration {it}. Score threshold met by supervisor.")
                break
            # Multi-GPU: also check for .stop file written by rank 0
            if self.logger.log_dir is not None and (Path(self.logger.log_dir) / ".stop").exists():
                print(f"[FPO runner] Early stop at iteration {it}. .stop file detected.")
                break
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    # Sample actions
                    actions = self.alg.act(obs)
                    # Step the environment
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    # Check for NaN values from the environment
                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)
                    # Move to device
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    # Process the step
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    # Extract intrinsic rewards if RND is used (only for logging)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.cfg["algorithm"]["rnd_cfg"] else None
                    # Book keeping
                    self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)

                stop = time.time()
                collect_time = stop - start
                start = stop

                # Compute returns
                self.alg.compute_returns(obs)

            # Update policy
            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            sup_update_time: str = ""
            if (self._ipc_sub is not None) and (self._ipc_topic_data is not None):
                [topic_success, self._ipc_topic_data] = self._ipc_sub.try_get(self._ipc_topic_data)  # update topic data with latest supervisor state
                topic_msg = self._ipc_topic_data.topic() if topic_success else topic_msg
                sup_update_time = (
                    f"Supervisor update time: {topic_msg.update_time}"
                    if (topic_success) or (topic_msg is not None)
                    else "Supervisor update time: N/A"
                )
            # Log information
            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=self.alg.rnd.weight if self.cfg["algorithm"]["rnd_cfg"] else None,
                additional_info=sup_update_time,
            )

            # Save model
            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore

        # Save the final model after training and stop the logging writer
        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))  # type: ignore
            self.logger.stop_logging_writer()
        self._close_ipc()

    def save(self, path: str, infos=None):
        super().save(path, infos)
        policy_path = path.split("model")[0]
        filename = "policy.onnx"
        self.export_policy_to_onnx(policy_path, filename)
        run_name: str = wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"  # type: ignore[assignment]
        onnx_path = os.path.join(policy_path, filename)
        metadata = get_base_metadata(self.env.unwrapped, run_name)
        attach_metadata_to_onnx(onnx_path, metadata)
        if self.logger.logger_type in ["wandb"]:
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
