from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg, RslRlBaseRunnerCfg
from typing import Optional
import torch.nn as nn
from dataclasses import dataclass, field
from class_free_guide.pineline.rl.rsl_rl.flow_ppo.config import (
    FpoRslRlOnPolicyRunnerCfg,
    FpoRslRlPpoActorCriticCfg,
    FpoRslRlPpoAlgorithmCfg,
)


def unitree_go2_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        logger="tensorboard",
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,  #  model for 5e-5
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="go2_velocity",
        save_interval=500,
        num_steps_per_env=24,
        max_iterations=10001,
    )


def unitree_go2_fpo_runner_cfg() -> FpoRslRlOnPolicyRunnerCfg:
    return FpoRslRlOnPolicyRunnerCfg(
        policy=FpoRslRlPpoActorCriticCfg(
            class_name="ActorCritic",
            extern_model=False,
            init_noise_std=1.0,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
            actor_scale=1.0,
            actor_mlp_output_scale=1.0,
            actor_final_layer_weight_scale=None,
            timestep_embed_dim=8,
            training_sampling_steps=None,
        ),
        algorithm=FpoRslRlPpoAlgorithmCfg(
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            knn_entropy_coef=0.0,
            knn_entropy_k=1,
            desired_kl=0.01,
            max_grad_norm=1.0,
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            trust_region_mode="aspo",
        ),
        seed=42,
        device="cuda:0",
        num_steps_per_env=24,
        max_iterations=10001,
        empirical_normalization=True,
        randomize_reset_episode_progress=0.0,
        clip_actions=2.0,
        save_interval=500,
        experiment_name="go2_velocity_fpo",
        run_name="",
    )
