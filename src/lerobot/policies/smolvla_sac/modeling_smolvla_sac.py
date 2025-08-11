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

import json
import math
import os
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.normalize import Normalize, Unnormalize
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla_sac.configuration_smolvla_sac import SmolVLASACConfig
from lerobot.configs.types import PolicyFeature, FeatureType, NormalizationMode
from lerobot.policies.sac.modeling_sac import (
    CriticEnsemble,
    CriticHead,
    DiscreteCritic,
    MLP,
    SACObservationEncoder,
    orthogonal_init,
)
from lerobot.policies.utils import get_device_from_parameters


class IndependentCritic(nn.Module):
    """
    Independent critic network that doesn't use SmolVLA features.
    This critic has its own observation encoder and processes observations independently.
    """
    
    def __init__(
        self,
        config: SmolVLASACConfig,
        normalize_inputs: Normalize,
        normalize_targets: Normalize,
    ):
        super().__init__()
        self.config = config
        
        # Create independent observation encoder for critic
        self.encoder = SACObservationEncoder(
            config=config,
            input_normalizer=normalize_inputs,
        )
        
        # Create critic heads
        critics = []
        critic_input_dim = self.encoder.output_dim + config.output_features["action"].shape[0]
        for _ in range(config.num_critics):
            # Get critic config but exclude input_dim since we provide it explicitly
            critic_kwargs = {k: v for k, v in config.critic_network_kwargs.__dict__.items() if k != 'input_dim'}
            critic = CriticHead(
                input_dim=critic_input_dim,
                **critic_kwargs,
            )
            critics.append(critic)
        
        self.critic_ensemble = CriticEnsemble(
            encoder=self.encoder,
            ensemble=critics,
            output_normalization=normalize_targets,
        )
        
        # Create target critic ensemble
        target_critics = []
        for _ in range(config.num_critics):
            target_critic = CriticHead(
                input_dim=critic_input_dim,
                **critic_kwargs,
            )
            target_critics.append(target_critic)
        
        self.critic_target = CriticEnsemble(
            encoder=self.encoder,
            ensemble=target_critics,
            output_normalization=normalize_targets,
        )
        
        # Initialize target network with same weights
        self.critic_target.load_state_dict(self.critic_ensemble.state_dict())
        
        # Freeze target network initially
        for param in self.critic_target.parameters():
            param.requires_grad = False
    
    def forward(
        self,
        observations: dict[str, Tensor],
        actions: Tensor,
        use_target: bool = False,
        observation_features: Tensor | None = None,
    ) -> Tensor:
        """Forward pass through critic network"""
        if use_target:
            return self.critic_target(observations, actions, observation_features)
        else:
            return self.critic_ensemble(observations, actions, observation_features)
    
    def update_target(self, tau: float = 0.005):
        """Update target network with soft updates"""
        with torch.no_grad():
            for param, target_param in zip(
                self.critic_ensemble.parameters(), self.critic_target.parameters()
            ):
                target_param.data.mul_(1 - tau)
                target_param.data.add_(tau * param.data)


class SmolVLAActorWrapper(nn.Module):
    """
    Wrapper for SmolVLA policy to make it compatible with SAC actor interface.
    """
    
    def __init__(self, smolvla_policy: SmolVLAPolicy):
        super().__init__()
        self.smolvla = smolvla_policy
    
    def forward(
        self,
        observations: dict[str, Tensor],
        observation_features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass through SmolVLA actor
        
        Returns:
            tuple: (actions, log_probs, means)
        """
        # Use SmolVLA's forward method
        actions, log_probs, means = self.smolvla.forward(observations)
        return actions, log_probs, means
    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Get state dict from SmolVLA"""
        return self.smolvla.state_dict(destination, prefix, keep_vars)
    
    def load_state_dict(self, state_dict, strict=True):
        """Load state dict to SmolVLA"""
        return self.smolvla.load_state_dict(state_dict, strict)
    
    def parameters(self):
        """Get parameters from SmolVLA"""
        return self.smolvla.parameters()


class SmolVLASACPolicy(PreTrainedPolicy):
    """
    SmolVLA-SAC Policy that combines SmolVLA as actor with independent SAC critics.
    
    This implementation follows the hybrid approach where:
    - SmolVLA serves as the actor (policy network)
    - Independent critic networks are used for value estimation
    - SAC algorithm is used for training
    """
    
    config_class = SmolVLASACConfig
    name = "smolvla_sac"
    
    def __init__(
        self,
        config: SmolVLASACConfig | None = None,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
        smolvla_policy: SmolVLAPolicy | None = None,
    ):
        """
        Initialize SmolVLA-SAC Policy
        
        Args:
            config: Policy configuration
            dataset_stats: Dataset statistics for normalization
            smolvla_policy: Pre-trained SmolVLA policy to use as actor (optional)
        """
        super().__init__(config)
        config.validate_features()
        self.config = config
        
        # Initialize normalization (inherited from SmolVLA)
        self.normalize_inputs = Normalize(
            config.input_features, config.normalization_mapping, dataset_stats
        )
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        
        # Determine action dimension
        self.continuous_action_dim = config.output_features["action"].shape[0]
        
        # Initialize SmolVLA as actor
        self._init_smolvla_actor(dataset_stats, smolvla_policy)
        
        # Initialize independent critics
        self._init_independent_critics()
        
        # Initialize discrete critics if needed
        if config.num_discrete_actions is not None:
            self._init_discrete_critics()
        
        # Initialize temperature parameter
        self._init_temperature()
    
    def _init_smolvla_actor(self, dataset_stats, smolvla_policy: SmolVLAPolicy | None = None):
        """Initialize SmolVLA as the actor"""
        if smolvla_policy is not None:
            # Use provided pre-trained SmolVLA policy
            self.smolvla = smolvla_policy
        else:
            # Create new SmolVLA policy with same config
            self.smolvla = SmolVLAPolicy(self.config, dataset_stats)
        
        # Wrap SmolVLA for SAC compatibility
        self.actor = SmolVLAActorWrapper(self.smolvla)
    
    def _init_independent_critics(self):
        """Initialize independent critic networks"""
        self.critic = IndependentCritic(
            config=self.config,
            normalize_inputs=self.normalize_inputs,
            normalize_targets=self.normalize_targets,
        )
    
    def _init_discrete_critics(self):
        """Initialize discrete critic if needed"""
        if self.config.num_discrete_actions is None:
            return
        
        # Create encoder for discrete critic (independent from main critic)
        encoder = SACObservationEncoder(
            config=self.config,
            input_normalizer=self.normalize_inputs,
        )
        
        self.discrete_critic = DiscreteCritic(
            encoder=encoder,
            input_dim=encoder.output_dim,
            output_dim=self.config.num_discrete_actions,
            **self.config.discrete_critic_network_kwargs.__dict__,
        )
        
        self.discrete_critic_target = DiscreteCritic(
            encoder=encoder,
            input_dim=encoder.output_dim,
            output_dim=self.config.num_discrete_actions,
            **self.config.discrete_critic_network_kwargs.__dict__,
        )
        
        # Initialize target with same weights
        self.discrete_critic_target.load_state_dict(self.discrete_critic.state_dict())
    
    def _init_temperature(self):
        """Initialize temperature parameter for SAC"""
        temp_init = self.config.temperature_init
        self.log_alpha = nn.Parameter(torch.tensor([math.log(temp_init)]))
        self.temperature = self.log_alpha.exp().item()
        
        # Set target entropy if not specified
        self.target_entropy = self.config.target_entropy
        if self.target_entropy is None:
            dim = self.continuous_action_dim
            if self.config.num_discrete_actions is not None:
                dim += 1
            self.target_entropy = -dim / 2
    
    def get_optim_params(self) -> dict:
        """Get optimizer parameters for different components"""
        optim_params = {
            "actor": self.actor.parameters(),
            "critic": self.critic.critic_ensemble.parameters(),
            "temperature": [self.log_alpha],
        }
        
        if self.config.num_discrete_actions is not None:
            optim_params["discrete_critic"] = self.discrete_critic.parameters()
        
        return optim_params
    
    def reset(self):
        """Reset the policy state"""
        self.smolvla.reset()
    
    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select action for inference/evaluation (deterministic)"""
        # Use SmolVLA's select_action method for consistency
        return self.smolvla.select_action(batch)
    
    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict action chunk using SmolVLA"""
        return self.smolvla.predict_action_chunk(batch)
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        config: SmolVLASACConfig | None = None,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = False,
        **kwargs,
    ):
        """
        Load a pretrained SmolVLASACPolicy with SmolVLA weights automatically loaded.
        
        This method:
        1. Loads the SmolVLA policy from the specified path
        2. Creates a new SmolVLASACPolicy with the loaded SmolVLA as actor
        3. Initializes independent critics with random weights
        
        Args:
            pretrained_name_or_path: Path to the pretrained SmolVLA model
            config: Configuration for the new policy (optional)
            dataset_stats: Dataset statistics for normalization (optional)
            **kwargs: Additional arguments passed to SmolVLA.from_pretrained
            
        Returns:
            SmolVLASACPolicy instance with pretrained SmolVLA actor
        """
        # Load pretrained SmolVLA policy
        smolvla_policy = SmolVLAPolicy.from_pretrained(
            pretrained_name_or_path,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
            strict=strict,
            **kwargs
        )
        
        # Create new config if not provided
        if config is None:
            config = SmolVLASACConfig()
            # Copy relevant settings from SmolVLA config
            config.input_features = smolvla_policy.config.input_features
            config.output_features = smolvla_policy.config.output_features
            config.normalization_mapping = smolvla_policy.config.normalization_mapping
        
        # Create new SmolVLASACPolicy with pretrained SmolVLA
        instance = cls(config=config, dataset_stats=dataset_stats, smolvla_policy=smolvla_policy)
        
        return instance
    
    def critic_forward(
        self,
        observations: dict[str, Tensor],
        actions: Tensor,
        use_target: bool = False,
        observation_features: Tensor | None = None,
    ) -> Tensor:
        """Forward pass through critic network"""
        return self.critic.forward(observations, actions, use_target, observation_features)
    
    def compute_loss_critic(
        self,
        observations: dict[str, Tensor],
        actions: Tensor,
        rewards: Tensor,
        next_observations: dict[str, Tensor],
        done: Tensor,
        observation_features: Tensor | None = None,
        next_observation_features: Tensor | None = None,
    ) -> Tensor:
        """Compute critic loss using SAC algorithm"""
        # Current Q-values
        current_q_values = self.critic_forward(observations, actions, use_target=False, observation_features=observation_features)
        
        with torch.no_grad():
            # Get next actions from actor
            next_actions, next_log_probs, _ = self.actor(next_observations, next_observation_features)
            
            # Get target Q-values
            next_q_values = self.critic_forward(next_observations, next_actions, use_target=True, observation_features=next_observation_features)
            
            # Take minimum over critics (double Q-learning)
            next_q_values = torch.min(next_q_values, dim=0)[0]
            
            # Compute target
            target_q_values = rewards + (1 - done) * self.config.discount * (
                next_q_values - self.temperature * next_log_probs
            )
        
        # Compute loss for each critic
        critic_losses = []
        for i in range(current_q_values.shape[0]):
            loss = nn.functional.mse_loss(current_q_values[i], target_q_values)
            critic_losses.append(loss)
        
        return torch.stack(critic_losses).mean()
    
    def compute_loss_actor(
        self,
        observations: dict[str, Tensor],
        observation_features: Tensor | None = None,
    ) -> Tensor:
        """Compute actor loss using SAC algorithm"""
        # Get actions and log probs from actor
        actions, log_probs, _ = self.actor(observations, observation_features)
        
        # Get Q-values from critic
        q_values = self.critic_forward(observations, actions, use_target=False, observation_features=observation_features)
        
        # Take minimum over critics
        min_q_values = torch.min(q_values, dim=0)[0]
        
        # Actor loss: maximize Q-value while minimizing entropy
        actor_loss = (self.temperature * log_probs - min_q_values).mean()
        
        return actor_loss
    
    def compute_loss_temperature(
        self,
        observations: dict[str, Tensor],
        observation_features: Tensor | None = None,
    ) -> Tensor:
        """Compute temperature loss for adaptive SAC"""
        with torch.no_grad():
            _, log_probs, _ = self.actor(observations, observation_features)
        
        # Temperature loss
        temperature_loss = -self.log_alpha * (log_probs + self.target_entropy).mean()
        
        return temperature_loss
    
    def update_target_networks(self):
        """Update target networks with soft updates"""
        tau = self.config.critic_target_update_weight
        self.critic.update_target(tau)
        
        if self.config.num_discrete_actions is not None:
            with torch.no_grad():
                for param, target_param in zip(
                    self.discrete_critic.parameters(), self.discrete_critic_target.parameters()
                ):
                    target_param.data.mul_(1 - tau)
                    target_param.data.add_(tau * param.data)
    
    def forward(
        self,
        batch: dict[str, Tensor | dict[str, Tensor]],
        model: Literal["actor", "critic", "temperature", "discrete_critic"] = "critic",
    ) -> dict[str, Tensor]:
        """
        Forward pass through the policy for training
        
        Args:
            batch: Batch of data
            model: Which model to forward through
            
        Returns:
            Dictionary containing the appropriate loss
        """
        observations: dict[str, Tensor] = batch["state"]
        actions: Tensor = batch["action"]
        observation_features: Tensor = batch.get("observation_feature")
        
        if model == "critic":
            rewards: Tensor = batch["reward"]
            next_observations: dict[str, Tensor] = batch["next_state"]
            done: Tensor = batch["done"]
            next_observation_features: Tensor = batch.get("next_observation_feature")
            
            loss_critic = self.compute_loss_critic(
                observations=observations,
                actions=actions,
                rewards=rewards,
                next_observations=next_observations,
                done=done,
                observation_features=observation_features,
                next_observation_features=next_observation_features,
            )
            return {"loss_critic": loss_critic}
        
        elif model == "actor":
            loss_actor = self.compute_loss_actor(
                observations=observations,
                observation_features=observation_features,
            )
            return {"loss_actor": loss_actor}
        
        elif model == "temperature":
            loss_temperature = self.compute_loss_temperature(
                observations=observations,
                observation_features=observation_features,
            )
            return {"loss_temperature": loss_temperature}
        
        elif model == "discrete_critic" and self.config.num_discrete_actions is not None:
            # Implement discrete critic loss if needed
            # For now, return a placeholder
            return {"loss_discrete_critic": torch.tensor(0.0, device=actions.device)}
        
        else:
            raise ValueError(f"Unknown model type: {model}")