"""RSL-RL PPO configuration for the tube placement task.

This file is intentionally a scaffold for now.
"""

from __future__ import annotations

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab.utils import configclass

@configclass
class TubePlacePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    device = "cuda"
    num_steps_per_env = 64
    max_iterations = 300
    save_interval = 25
    experiment_name = "tube_place"
    empirical_normalization = True
    
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.3,
        actor_hidden_dims=[128, 128],
        critic_hidden_dims=[128, 128],
        activation="elu",
    )
    
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3e-4,
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
