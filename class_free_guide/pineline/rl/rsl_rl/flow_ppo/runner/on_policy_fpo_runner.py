# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from pathlib import Path
import time
import torch

from typing import TYPE_CHECKING

from ..alg.flow_ppo_fpo import FPO
from ..utils.logger import Logger
import class_free_guide

if TYPE_CHECKING:
    from class_free_guide.pineline.rl.rsl_rl.flow_ppo.config import FpoRslRlOnPolicyRunnerCfg
from rsl_rl.env import VecEnv
from ..module.actor_critic_fpo import (
    ActorCritic,
)
from rsl_rl.modules import EmpiricalNormalization
from class_free_guide.supervisor import SupervisorConfig
import dzipc


class OnPolicyRunner:
    """On-policy runner for training and evaluation."""

    def __init__(self, env: VecEnv, train_cfg: FpoRslRlOnPolicyRunnerCfg, sup_cfg: SupervisorConfig | None, log_dir: str | None = None, device="cpu"):
        self.cfg = train_cfg
        self.sup_cfg = sup_cfg
        self.device = device
        self.env = env

        # check if multi-gpu is enabled
        self._configure_multi_gpu()

        # resolve dimensions of observations
        obs_ret = self.env.get_observations()
        print(f"[DEBUG] type of get_observations(): {type(obs_ret)}")
        if hasattr(obs_ret, "keys"):
            print(f"[DEBUG] obs_ret keys: {obs_ret.keys()}")

        obs, extras = self._process_obs_ret(obs_ret)

        num_obs = obs.shape[1]

        # resolve type of privileged observations
        if "critic" in extras["observations"]:
            self.privileged_obs_type = "critic"
        else:
            self.privileged_obs_type = None

        # resolve dimensions of privileged observations
        if self.privileged_obs_type is not None:
            num_privileged_obs = extras["observations"][self.privileged_obs_type].shape[1]
        else:
            num_privileged_obs = num_obs

        # initialize policy with config dataclass
        policy: ActorCritic = ActorCritic(num_obs, num_privileged_obs, self.env.num_actions, cfg=train_cfg.policy).to(self.device)

        # initialize algorithm with config dataclass
        self.alg: FPO = FPO(policy, cfg=train_cfg.algorithm, device=self.device, multi_gpu_cfg=self.multi_gpu_cfg)

        # store training configuration
        self.num_steps_per_env = train_cfg.num_steps_per_env
        self.save_interval = train_cfg.save_interval
        self.empirical_normalization = train_cfg.empirical_normalization
        self.randomize_reset_episode_progress = getattr(train_cfg, "randomize_reset_episode_progress", 0.0)
        self.custom_model_param_save = train_cfg.custom_model_param_save
        self.custom_model_param_method = train_cfg.custom_model_param_method
        ### check custom model param save configuration
        if self.custom_model_param_save and self.custom_model_param_method is None:
            raise ValueError("custom_model_param_method must be provided if custom_model_param_save is True")

        # store post-training evaluation configuration
        self.enable_post_training_eval = getattr(self.cfg, "enable_post_training_eval", True)
        self.post_eval_checkpoint_interval = getattr(self.cfg, "post_eval_checkpoint_interval", 1)

        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.privileged_obs_normalizer = EmpiricalNormalization(shape=[num_privileged_obs], until=1.0e8).to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
            self.privileged_obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization

        # init storage and model
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [num_obs],
            [num_privileged_obs],
            [self.env.num_actions],
        )

        # Logger — handles all writer/console/buffer management
        self.logger = Logger(
            log_dir=log_dir,
            cfg=train_cfg,
            env_cfg=None,
            num_envs=self.env.num_envs,
            is_distributed=self.is_distributed,
            gpu_world_size=self.gpu_world_size,
            gpu_global_rank=self.gpu_global_rank,
            device=self.device,
            git_status_repos=[class_free_guide.__file__, __file__],
            num_steps_per_env=train_cfg.num_steps_per_env,
        )
        self.current_learning_iteration = 0
        self.early_stop_event = None  # threading.Event, set by supervisor when training should stop
        self._ipc_msg_template = None
        self._ipc_sub = None
        self._ipc_topic_data = None
        self._init_ipc_subscriber()  # Initialize IPC subscriber for supervisor state monitoring

    def _init_ipc_subscriber(self) -> None:
        """Create a SHM subscriber so external tooling can watch supervisor state."""
        if not self.sup_cfg.ipc_enabled:
            return
        try:
            self._ipc_msg_template = dzipc.Supervisor()
            self._ipc_topic_data = dzipc.make_topic_data(self._ipc_msg_template)
            self._ipc_sub = dzipc.SubscriberIPCPtrMake(
                self._ipc_topic_data,
                self.sup_cfg.ipc_topic,
                self.sup_cfg.ipc_domain,
                10,
                dzipc.IPC_SHM,
                verbose=False,
            )
            self._ipc_sub.InitChannel()
        except Exception as exc:
            self._ipc_sub = None
            self._ipc_msg_template = None
            self._ipc_topic_data = None

    @staticmethod
    def _process_obs_ret(obs_ret):
        """Extract obs tensor and extras dict from get_observations() or step() return.

        Handles both the old API (tuple) and new API (TensorDict/dict) return types,
        ensuring compatibility across IsaacLab and MJLab environments.

        Args:
            obs_ret: Return value from get_observations() or the first element of step().

        Returns:
            tuple: (obs, extras) where obs is a tensor and extras is a dict
                   with an "observations" key containing privileged observation groups.
        """
        if isinstance(obs_ret, tuple):
            obs = obs_ret[0]
            extras = obs_ret[1] if len(obs_ret) > 1 else {}
        else:
            if isinstance(obs_ret, dict) or type(obs_ret).__name__ == "TensorDict":
                if "policy" in obs_ret:
                    obs = obs_ret["policy"]
                elif "actor" in obs_ret:
                    obs = obs_ret["actor"]
                else:
                    obs = obs_ret

                extras = {"observations": {}}
                if "critic" in obs_ret:
                    extras["observations"]["critic"] = obs_ret["critic"]
            else:
                obs = obs_ret
                extras = {"observations": {}}
        return obs, extras

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        # Define wandb metrics callback for post-training eval
        def _define_wandb_metrics():
            if self.enable_post_training_eval:
                import wandb

                wandb.define_metric("eval_iteration")
                eval_modes = getattr(self.cfg, "flow_eval_modes", ["zero", "fixed_seed", "random"])
                for mode in eval_modes:
                    wandb.define_metric(f"PostEval_{mode}/mean_reward", step_metric="eval_iteration")
                    wandb.define_metric(f"PostEval_{mode}/std_reward", step_metric="eval_iteration")
                    wandb.define_metric(f"PostEval_{mode}/mean_episode_length", step_metric="eval_iteration")
                    wandb.define_metric(f"PostEval_{mode}/std_episode_length", step_metric="eval_iteration")

        # Initialize logging writer (Tensorboard / W&B / Neptune)
        self.logger.init_logging_writer(
            wandb_define_metrics_callback=_define_wandb_metrics,
        )

        # randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))

        # start learning
        obs_ret = self.env.get_observations()
        obs, extras = self._process_obs_ret(obs_ret)
        privileged_obs = extras["observations"].get(self.privileged_obs_type, obs)
        obs, privileged_obs = obs.to(self.device), privileged_obs.to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example)

        # Book keeping — Logger manages reward/episode buffers internally
        # via process_env_step().

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()
            # TODO: Do we need to synchronize empirical normalizers?
            #   Right now: No, because they all should converge to the same values "asymptotically".

        # Start training
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        topic_msg = None
        for it in range(start_iter, tot_iter):
            # Check for supervisor-issued early stop signal.
            if self.early_stop_event is not None and self.early_stop_event.is_set():
                print(f"[FPO runner] Early stop at iteration {it}. Score threshold met by supervisor.")
                break
            # Multi-GPU: also check for .stop file written by rank 0
            if self.logger.log_dir is not None and (Path(self.logger.log_dir) / ".stop").exists():
                print(f"[FPO runner] Early stop at iteration {it}. .stop file detected.")
                break

            start = time.perf_counter()
            # Initialize timing accumulators
            env_step_time = 0.0
            action_time = 0.0
            process_time = 0.0
            # Rollout
            with torch.no_grad():
                for _ in range(self.num_steps_per_env):
                    # Sample actions
                    action_start = time.perf_counter()
                    actions = self.alg.act(obs, privileged_obs)
                    action_time += time.perf_counter() - action_start

                    # Step the environment
                    env_start = time.perf_counter()
                    obs_ret, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                    env_step_time += time.perf_counter() - env_start

                    # Extract observation tensor from return value
                    # (handles both TensorDict from MJLab/IsaacLab and legacy tuple API)
                    obs, step_extras = self._process_obs_ret(obs_ret)

                    # Randomize episode length for reset environments to prevent synchronization
                    if self.randomize_reset_episode_progress > 0:
                        reset_env_ids = (dones > 0).nonzero(as_tuple=False).squeeze(-1)
                        if len(reset_env_ids) > 0:
                            # Randomize episode progress for reset environments
                            max_progress = int(self.randomize_reset_episode_progress * self.env.max_episode_length)
                            random_lengths = torch.randint(0, max_progress + 1, (len(reset_env_ids),), device=self.device, dtype=torch.long)
                            self.env.episode_length_buf[reset_env_ids] = random_lengths

                    # Process observations and rewards
                    process_start = time.perf_counter()
                    # Move to device
                    obs, rewards, dones = (
                        obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    # perform normalization
                    obs = self.obs_normalizer(obs)
                    if self.privileged_obs_type is not None:
                        privileged_obs = self.privileged_obs_normalizer(step_extras["observations"][self.privileged_obs_type].to(self.device))
                    else:
                        privileged_obs = obs

                    # process the step
                    self.alg.process_env_step(rewards, dones, infos)
                    process_time += time.perf_counter() - process_start

                    # book keeping — delegate to Logger
                    self.logger.process_env_step(rewards, dones, infos)

                stop = time.perf_counter()
                collection_time = stop - start
                # Store detailed timing
                simulation_time = env_step_time + action_time + process_time
                start = stop

                # compute returns
                self.alg.compute_returns(privileged_obs)

            # update policy
            if self.empirical_normalization:
                loss_dict = self.alg.update(obs_normalizer=self.obs_normalizer, privileged_obs_normalizer=self.privileged_obs_normalizer)
            else:
                loss_dict = self.alg.update()

            # Update EMA after PPO update (Option B: per PPO update)
            if self.alg.ema is not None:
                if self.alg.tot_timesteps == self.alg.ema_warmup_steps:
                    # At warmup threshold, reset EMA to current weights
                    self.alg.ema.reset_to_current()
                elif self.alg.tot_timesteps > self.alg.ema_warmup_steps:
                    # After warmup, do normal EMA updates
                    self.alg.ema.update()

            stop = time.perf_counter()
            learn_time = stop - start
            self.current_learning_iteration = it
            # log info
            if self.logger.writer is not None:
                sup_update_time: str = ""
                if (self._ipc_sub is not None) and (self._ipc_topic_data is not None):
                    [topic_success, self._ipc_topic_data] = self._ipc_sub.try_get(
                        self._ipc_topic_data
                    )  # update topic data with latest supervisor state
                    topic_msg = self._ipc_topic_data.topic() if topic_success else topic_msg
                    sup_update_time = (
                        f"Supervisor update time: {topic_msg.update_time}"
                        if (topic_success) or (topic_msg is not None)
                        else "Supervisor update time: N/A"
                    )
                self.logger.log(
                    it=it,
                    start_it=start_iter,
                    total_it=tot_iter,
                    collect_time=collection_time,
                    learn_time=learn_time,
                    loss_dict=loss_dict,
                    learning_rate=self.alg.learning_rate,
                    timing_details={
                        "env_step_time": env_step_time,
                        "action_time": action_time,
                        "process_time": process_time,
                        "simulation_time": simulation_time,
                    },
                    additional_info=sup_update_time,
                )

                # Save model
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))

        # Save the final model after training
        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))

        # Run post-training checkpoint evaluation
        if self.logger.writer is not None:
            self.run_post_training_checkpoint_eval()

        # Properly close the writer
        self.logger.stop_logging_writer()

    def save(self, path: str, infos=None):
        # -- Prepare model state dict (use EMA if available)
        model_state_dict = self.alg.policy.state_dict()
        if self.alg.ema is not None and self.alg.tot_timesteps > self.alg.ema_warmup_steps:
            # Replace actor weights with EMA shadow params.
            # EMA tracks policy.actor params (keys like "0.weight"), but
            # policy.state_dict() prefixes them with "actor." ("actor.0.weight").
            # Only do this after EMA warmup — before warmup, shadow params are
            # copies of random init weights, not the current trained weights.
            for name, ema_param in self.alg.ema.shadow_params.items():
                full_name = f"actor.{name}"
                if full_name in model_state_dict:
                    model_state_dict[full_name] = ema_param.clone()

        # -- Save model
        saved_dict = {
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        # -- Save EMA state if used
        if self.alg.ema is not None:
            saved_dict["ema_state_dict"] = self.alg.ema.state_dict()
        # -- Save observation normalizer if used
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["privileged_obs_norm_state_dict"] = self.privileged_obs_normalizer.state_dict()

        # save model
        torch.save(saved_dict, path)

        # upload model to external logging service
        self.logger.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True):
        loaded_dict = torch.load(path, weights_only=False)
        # -- Load model
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        # -- Load observation normalizer if used
        if self.empirical_normalization:
            if resumed_training:
                # if a previous training is resumed, the actor/student normalizer is loaded for the actor/student
                # and the critic/teacher normalizer is loaded for the critic/teacher
                self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
                self.privileged_obs_normalizer.load_state_dict(loaded_dict["privileged_obs_norm_state_dict"])
            else:
                # if the training is not resumed but a model is loaded, load the actor normalizer
                # for the privileged obs normalizer (observation space may differ)
                self.privileged_obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
        # -- load optimizer if used
        if load_optimizer and resumed_training:
            # -- algorithm optimizer
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        # -- Load EMA state if used
        if self.alg.ema is not None:
            if "ema_state_dict" in loaded_dict:
                self.alg.ema.load_state_dict(loaded_dict["ema_state_dict"])
                print("[INFO] Loaded EMA state from checkpoint")
            else:
                print("[WARNING] EMA is enabled but no EMA state found in checkpoint")
        # -- load current learning iteration
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
            # -- restore common_step_counter to avoid restarting episode length warmup
            # Calculate total steps from iteration count
            steps_per_iteration = self.num_steps_per_env * self.env.num_envs
            total_steps = self.current_learning_iteration * steps_per_iteration
            # Set the environment's common_step_counter
            if hasattr(self.env.unwrapped, "common_step_counter"):
                self.env.unwrapped.common_step_counter = total_steps
                print(f"[INFO] Restored common_step_counter to {total_steps} based on iteration {self.current_learning_iteration}")
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.policy.to(device)
        policy = self.alg.policy.act_inference
        if self.cfg.empirical_normalization:
            if device is not None:
                self.obs_normalizer.to(device)
            policy = lambda x: self.alg.policy.act_inference(self.obs_normalizer(x))  # noqa: E731
        return policy

    def train_mode(self):
        # -- PPO
        self.alg.policy.train()
        # -- Normalization
        if self.empirical_normalization:
            self.obs_normalizer.train()
            self.privileged_obs_normalizer.train()

    def eval_mode(self):
        # -- PPO
        self.alg.policy.eval()
        # -- Normalization
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.privileged_obs_normalizer.eval()

    def get_checkpoint_paths(self):
        """Scan log directory for checkpoint files and return sorted list.

        Returns:
            List of (iteration, filepath) tuples, filtered by post_eval_checkpoint_interval
        """
        import glob
        import re

        if self.logger.log_dir is None:
            return []

        # Find all checkpoint files
        checkpoint_pattern = os.path.join(self.logger.log_dir, "model_*.pt")
        checkpoint_files = glob.glob(checkpoint_pattern)

        # Parse iteration numbers from filenames
        checkpoints = []
        for filepath in checkpoint_files:
            filename = os.path.basename(filepath)
            match = re.match(r"model_(\d+)\.pt", filename)
            if match:
                iteration = int(match.group(1))
                checkpoints.append((iteration, filepath))

        # Sort by iteration
        checkpoints.sort(key=lambda x: x[0])

        # Filter by checkpoint interval (take every Nth checkpoint)
        if self.post_eval_checkpoint_interval > 1:
            checkpoints = checkpoints[:: self.post_eval_checkpoint_interval]

        return checkpoints

    def evaluate_checkpoint(self, checkpoint_path, iteration):
        """Evaluate a single checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file
            iteration: Iteration number of the checkpoint

        Returns:
            Dictionary containing evaluation metrics for each mode
        """
        import numpy as np

        # Initialize state variables to avoid NameError in exception handler
        current_model_state = None
        current_obs_norm_state = None
        current_priv_obs_norm_state = None

        try:
            # Save current model and normalizer state
            current_model_state = {k: v.clone() for k, v in self.alg.policy.state_dict().items()}
            if self.empirical_normalization:
                current_obs_norm_state = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in self.obs_normalizer.state_dict().items()}
                current_priv_obs_norm_state = {
                    k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in self.privileged_obs_normalizer.state_dict().items()
                }

            # Load checkpoint (this loads both model and normalizer)
            loaded_dict = torch.load(checkpoint_path, weights_only=False)
            self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])

            # Load normalizer state from checkpoint if available
            if self.empirical_normalization and "obs_norm_state_dict" in loaded_dict:
                self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
                self.privileged_obs_normalizer.load_state_dict(loaded_dict["privileged_obs_norm_state_dict"])

            # Note: post-warmup checkpoints have EMA weights baked into model_state_dict
            # (save() replaces actor weights with EMA shadow params after warmup).
            # Pre-warmup checkpoints have regular training weights (no EMA bake-in).
            # No need to do ema.store()/copy_to() here — that would overwrite the
            # checkpoint's weights with end-of-training shadow_params.

            # Switch to eval mode
            self.eval_mode()

            # Determine number of episodes for this eval
            num_episodes = getattr(self.cfg, "eval_episodes", 10)

            # Get eval modes from config
            eval_modes = getattr(self.cfg, "flow_eval_modes", ["zero", "fixed_seed", "random"])
            eval_fixed_seed = getattr(self.cfg, "flow_eval_fixed_seed", 12345)

            eval_results = {}

            for mode in eval_modes:
                # Reset environments for each mode
                obs_ret, _ = self.env.reset()
                obs, _extras = self._process_obs_ret(obs_ret)
                obs = obs.to(self.device)

                # Collect episodes for this mode
                mode_rewards = []
                mode_lengths = []

                # Track episode data
                episode_reward = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
                episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

                # Get max episode length
                max_episode_length = self.env.max_episode_length if hasattr(self.env, "max_episode_length") else 1000

                # Run until we collect enough episodes
                episodes_per_env = torch.zeros(self.env.num_envs, dtype=torch.long, device=self.device)
                target_episodes_per_env = max(1, num_episodes // self.env.num_envs)

                while (episodes_per_env < target_episodes_per_env).any():
                    with torch.no_grad():
                        # Normalize observations if needed
                        if self.cfg.empirical_normalization:
                            norm_obs = self.obs_normalizer(obs)
                        else:
                            norm_obs = obs

                        # Use flow matching with specified evaluation mode
                        actions = self.alg.policy.act_inference(norm_obs, eval_mode=mode, eval_fixed_seed=eval_fixed_seed)

                    # Step environment
                    obs_ret, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                    obs, _step_extras = self._process_obs_ret(obs_ret)
                    obs = obs.to(self.device)

                    # Accumulate rewards and lengths
                    episode_reward += rewards.to(self.device)
                    episode_length += 1

                    # Check for done episodes
                    done_mask = (dones > 0) | (episode_length >= max_episode_length)

                    if done_mask.any():
                        # Record completed episodes
                        for idx in done_mask.nonzero(as_tuple=False).squeeze(-1):
                            if episodes_per_env[idx] < target_episodes_per_env:
                                mode_rewards.append(episode_reward[idx].item())
                                mode_lengths.append(episode_length[idx].item())
                                episodes_per_env[idx] += 1

                        # Reset the environments that are done
                        episode_reward[done_mask] = 0
                        episode_length[done_mask] = 0

                # Store results for this mode
                eval_results[mode] = {
                    "mean_reward": np.mean(mode_rewards) if mode_rewards else 0,
                    "std_reward": np.std(mode_rewards) if mode_rewards else 0,
                    "mean_length": np.mean(mode_lengths) if mode_lengths else 0,
                    "std_length": np.std(mode_lengths) if mode_lengths else 0,
                    "num_episodes": len(mode_rewards),
                }

            # Restore original model and normalizer state
            self.alg.policy.load_state_dict(current_model_state)
            if self.empirical_normalization:
                self.obs_normalizer.load_state_dict(current_obs_norm_state)
                self.privileged_obs_normalizer.load_state_dict(current_priv_obs_norm_state)

            # Switch back to train mode
            self.train_mode()

            # Clear GPU cache
            torch.cuda.empty_cache()

            return eval_results

        except Exception as e:
            print(f"Warning: Failed to evaluate checkpoint at iteration {iteration}: {e}")
            # Attempt to restore state even on failure
            try:
                if current_model_state is not None:
                    self.alg.policy.load_state_dict(current_model_state)
                if self.empirical_normalization and current_obs_norm_state is not None:
                    self.obs_normalizer.load_state_dict(current_obs_norm_state)
                    self.privileged_obs_normalizer.load_state_dict(current_priv_obs_norm_state)
                self.train_mode()
            except:
                pass
            return None

    def run_post_training_checkpoint_eval(self):
        """Run post-training evaluation on all saved checkpoints.

        Evaluates all checkpoints and logs results to WandB with custom eval_iteration metric.
        Only runs on rank 0 in distributed training.
        """
        # Skip if disabled or not on main rank
        if not self.enable_post_training_eval:
            return

        if self.logger.disable_logs:
            print("[INFO] Skipping post-training eval on non-main rank in distributed training")
            return

        if self.logger.writer is None:
            print("[WARNING] No writer available, skipping post-training eval")
            return

        print("\n" + "=" * 80)
        print("Starting post-training checkpoint evaluation...")
        print("=" * 80)

        # Get all checkpoint paths
        checkpoints = self.get_checkpoint_paths()

        if not checkpoints:
            print("[WARNING] No checkpoints found for post-training evaluation")
            return

        print(f"Found {len(checkpoints)} checkpoint(s) to evaluate")
        if self.post_eval_checkpoint_interval > 1:
            print(f"  (evaluating every {self.post_eval_checkpoint_interval} checkpoint(s))")

        num_episodes = getattr(self.cfg, "eval_episodes", 10)
        print(f"  Episodes per mode: {num_episodes}")
        print(f"  Eval modes: {getattr(self.cfg, 'flow_eval_modes', ['zero', 'fixed_seed', 'random'])}")

        # Evaluate each checkpoint
        all_results = {}
        for idx, (iteration, checkpoint_path) in enumerate(checkpoints, 1):
            print(f"\n[{idx}/{len(checkpoints)}] Evaluating checkpoint at iteration {iteration}...")

            eval_results = self.evaluate_checkpoint(checkpoint_path, iteration)

            if eval_results is None:
                print(f"  Skipped due to error")
                continue

            # Store results
            all_results[iteration] = eval_results

            # Log to wandb with custom x-axis
            if self.logger.logger_type == "wandb":
                try:
                    import wandb

                    # Consolidate all metrics into a single log call
                    log_dict = {"eval_iteration": iteration}
                    for mode, results in eval_results.items():
                        log_dict[f"PostEval_{mode}/mean_reward"] = results["mean_reward"]
                        log_dict[f"PostEval_{mode}/std_reward"] = results["std_reward"]
                        log_dict[f"PostEval_{mode}/mean_episode_length"] = results["mean_length"]
                        log_dict[f"PostEval_{mode}/std_episode_length"] = results["std_length"]
                    wandb.log(log_dict)
                except Exception as e:
                    print(f"  Warning: Failed to log to wandb: {e}")
            else:
                # For tensorboard/neptune, use regular writer (they don't have the step ordering issue)
                for mode, results in eval_results.items():
                    self.logger.writer.add_scalar(f"PostEval_{mode}/mean_reward", results["mean_reward"], iteration)
                    self.logger.writer.add_scalar(f"PostEval_{mode}/std_reward", results["std_reward"], iteration)
                    self.logger.writer.add_scalar(f"PostEval_{mode}/mean_episode_length", results["mean_length"], iteration)
                    self.logger.writer.add_scalar(f"PostEval_{mode}/std_episode_length", results["std_length"], iteration)

            # Print results for this checkpoint
            for mode, results in eval_results.items():
                print(
                    f"    {mode:12s}: reward={results['mean_reward']:8.2f} ± {results['std_reward']:6.2f}, "
                    f"length={results['mean_length']:6.1f} ± {results['std_length']:5.1f}"
                )

        # Print summary
        print("\n" + "=" * 80)
        print("Post-training checkpoint evaluation completed!")
        print(f"Successfully evaluated {len(all_results)}/{len(checkpoints)} checkpoints")
        print("=" * 80 + "\n")

    def add_git_repo_to_log(self, repo_file_path):
        self.logger.git_status_repos.append(repo_file_path)

    """
    Helper functions.
    """

    def _configure_multi_gpu(self):
        """Configure multi-gpu training."""
        # check if distributed training is enabled
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        # if not distributed training, set local and global rank to 0 and return
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.multi_gpu_cfg = None
            return

        # get rank and world size
        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        # make a configuration dictionary
        self.multi_gpu_cfg = {
            "global_rank": self.gpu_global_rank,  # rank of the main process
            "local_rank": self.gpu_local_rank,  # rank of the current process
            "world_size": self.gpu_world_size,  # total number of processes
        }

        # check if user has device specified for local rank
        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'.")
        # validate multi-gpu configuration
        if self.gpu_local_rank >= self.gpu_world_size:
            raise ValueError(f"Local rank '{self.gpu_local_rank}' is greater than or equal to world size '{self.gpu_world_size}'.")
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(f"Global rank '{self.gpu_global_rank}' is greater than or equal to world size '{self.gpu_world_size}'.")

        # initialize torch distributed (skip if already initialized by torchrunx)
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size)
        else:
            print(f"[INFO] torch.distributed already initialized (by torchrunx), skipping duplicate init.")
        # set device to the local rank
        torch.cuda.set_device(self.gpu_local_rank)
