#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Any

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.sac.configuration_sac import (
    ActorLearnerConfig,
    ActorNetworkConfig,
    ConcurrencyConfig,
    CriticNetworkConfig,
    PolicyConfig,
)
from lerobot.optim.optimizers import MultiAdamConfig


@dataclass
class CriticOnlyNetworkConfig:
    """Configuration for critic-only network architecture"""
    input_dim: int = 256
    hidden_dims: list[int] = field(default_factory=lambda: [1024, 1024])
    activations: str = "SiLU"
    activate_final: bool = False
    dropout_rate: float | None = None
    init_final: float | None = None
    final_activation: str | None = None


@PreTrainedConfig.register_subclass("smolvla_sac")
@dataclass
class SmolVLASACConfig(SmolVLAConfig):
    """
    Configuration for SmolVLA SAC Policy.
    
    This combines SmolVLA as the actor with independent SAC critics.
    The actor uses the pre-trained SmolVLA model, while critics are initialized independently.
    """
    
    # SAC-specific parameters
    # Number of critics in the ensemble
    num_critics: int = 2
    # Number of subsampled critics for training
    num_subsample_critics: int | None = None
    # Learning rate for the critic network
    critic_lr: float = 3e-4
    # Learning rate for the actor network (SmolVLA)
    actor_lr: float = 3e-4
    # Learning rate for the temperature parameter
    temperature_lr: float = 3e-4
    # Weight for the critic target update
    critic_target_update_weight: float = 0.005
    # Update-to-data ratio for the UTD algorithm
    utd_ratio: int = 1
    
    # SAC algorithm parameters
    # Discount factor for the SAC algorithm
    discount: float = 0.99
    # Initial temperature value
    temperature_init: float = 1.0
    # Target entropy for the SAC algorithm
    target_entropy: float | None = None
    # Whether to use backup entropy for the SAC algorithm
    use_backup_entropy: bool = True
    # Gradient clipping norm for the SAC algorithm
    grad_clip_norm: float = 40.0
    
    # Training parameters
    # Number of steps for online training
    online_steps: int = 1000000
    # Seed for the online environment
    online_env_seed: int = 10000
    # Capacity of the online replay buffer
    online_buffer_capacity: int = 100000
    # Capacity of the offline replay buffer
    offline_buffer_capacity: int = 100000
    # Whether to use asynchronous prefetching for the buffers
    async_prefetch: bool = False
    # Number of steps before learning starts
    online_step_before_learning: int = 100
    # Frequency of policy updates
    policy_update_freq: int = 1
    
    # Network architecture configurations
    # Configuration for the critic network architecture
    critic_network_kwargs: CriticOnlyNetworkConfig = field(default_factory=CriticOnlyNetworkConfig)
    # Configuration for the actor network architecture (inherited from SmolVLA)
    actor_network_kwargs: ActorNetworkConfig = field(default_factory=ActorNetworkConfig)
    # Configuration for the policy parameters
    policy_kwargs: PolicyConfig = field(default_factory=PolicyConfig)
    # Configuration for the discrete critic network
    discrete_critic_network_kwargs: CriticNetworkConfig = field(default_factory=CriticNetworkConfig)
    # Configuration for actor-learner architecture
    actor_learner_config: ActorLearnerConfig = field(default_factory=ActorLearnerConfig)
    # Configuration for concurrency settings
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)
    
    # Encoder configurations
    # Whether to use a shared encoder for actor and critic
    shared_encoder: bool = False  # Default to False for independent critics
    # Number of discrete actions, eg for gripper actions
    num_discrete_actions: int | None = None
    # Hidden dimension size for the state encoder (critic side)
    state_encoder_hidden_dim: int = 256
    # Dimension of the latent space (critic side)
    latent_dim: int = 256
    # Vision encoder name for critics
    vision_encoder_name: str | None = None
    # Image encoder hidden dimension for critics
    image_encoder_hidden_dim: int = 256
    # Dimension of the image embedding pooling for critics
    image_embedding_pooling_dim: int = 8
    
    # Model control parameters
    # Whether to freeze SmolVLA during critic training
    freeze_smolvla_for_critic: bool = True
    # Whether to use torch.compile for optimization
    use_torch_compile: bool = False  # Keep False for stability initially
    
    def __post_init__(self):
        super().__post_init__()
        # Validate SAC-specific configurations
        if self.num_critics < 1:
            raise ValueError("num_critics must be at least 1")
        if self.num_subsample_critics is not None and self.num_subsample_critics > self.num_critics:
            raise ValueError("num_subsample_critics cannot be greater than num_critics")
    
    def get_optimizer_preset(self) -> MultiAdamConfig:
        """Get the optimizer configuration for SmolVLA-SAC"""
        optimizer_groups = {
            "actor": {"lr": self.actor_lr},
            "critic": {"lr": self.critic_lr},
            "temperature": {"lr": self.temperature_lr},
        }
        
        if self.num_discrete_actions is not None:
            optimizer_groups["discrete_critic"] = {"lr": self.critic_lr}
            
        return MultiAdamConfig(
            weight_decay=0.0,
            optimizer_groups=optimizer_groups,
        )
    
    def get_scheduler_preset(self) -> None:
        return None
    
    def validate_features(self) -> None:
        """Validate that the features are compatible with SmolVLA-SAC"""
        # Use SmolVLA's validation first
        super().validate_features()
        
        # Additional SAC-specific validation can be added here
        # For now, we rely on SmolVLA's feature validation
        pass