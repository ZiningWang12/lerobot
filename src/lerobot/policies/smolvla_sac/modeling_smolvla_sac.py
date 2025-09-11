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
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
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

# Import constants from SmolVLA
from lerobot.policies.smolvla.modeling_smolvla import ACTION, OBS_STATE


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
    
    Key insight: SmolVLA is a diffusion policy, so we need to use its inference methods
    (sample_actions) rather than its training methods (forward).
    """
    
    def __init__(self, smolvla_policy: SmolVLAPolicy):
        super().__init__()
        self.smolvla = smolvla_policy
        
        # For SAC compatibility, we need to track action statistics
        # This will be used to compute log_probs for entropy regularization
        self.action_stats = {
            'mean': None,
            'std': None
        }
    
    # TODO: The actor forward is NOT compatible with the SAC framework, and needs a thorough rework.
    # The current implementation is a hack to make it work, and is not usable.
    def forward(
        self,
        observations: dict[str, Tensor],
        observation_features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass through SmolVLA actor for SAC training
        
        Returns:
            tuple: (actions, log_probs, means)
            
        Note: We use the diffusion policy approach where we:
        1. Generate actions using SmolVLA's inference method
        2. Compute log_probs using the diffusion process
        3. Allow gradients to flow back through the denoising process
        """
        # Prepare batch for SmolVLA
        batch = self._prepare_batch_for_smolvla(observations)
        
        # Generate actions using SmolVLA's inference method
        # This is the key: we need to use the inference method but allow gradients
        actions_chunk = self.smolvla._get_action_chunk(batch)
        
        # Unpad actions to get final action dimension
        original_action_dim = self.smolvla.config.output_features["action"].shape[0]
        actions_chunk = actions_chunk[:, :, :original_action_dim]
        
        # Unnormalize actions
        actions_chunk = self.smolvla.unnormalize_outputs({ACTION: actions_chunk})[ACTION]
        
        # For SAC, we need single actions, not action chunks
        actions = actions_chunk[:, 0, :]  # Shape: (batch_size, action_dim)
        
        # Approximate log_probs for SAC entropy regularization
        log_probs = self._approximate_log_probs(actions)
        
        # For means, we use the actions themselves
        means = actions
        
        return actions, log_probs, means
    
    def _prepare_batch_for_smolvla(self, observations: dict[str, Tensor]) -> dict[str, Tensor]:
        """Prepare observations for SmolVLA processing"""
        # Create a batch with the original observation format that SmolVLA expects
        batch = {
            'observation.images.handeye': observations['observation.images.handeye'],
            'observation.images.global': observations['observation.images.global'],
            'observation.state': observations['observation.state'],
            'task': observations.get('task', ['pick up the black ring'] * observations['observation.state'].shape[0])
        }
        
        # Apply SmolVLA preprocessing - this will handle the image processing internally
        batch = self.smolvla._prepare_batch(batch)
        
        return batch
    
    def _approximate_log_probs(self, actions: Tensor) -> Tensor:
        """
        Approximate log probabilities for SAC entropy regularization.
        
        Since SmolVLA is a diffusion policy, we don't have explicit log_probs.
        We approximate them using a simple Gaussian assumption.
        
        IMPORTANT: We need to ensure log_probs are in a reasonable range to avoid
        exploding TD targets in the critic loss computation.
        """
        batch_size, action_dim = actions.shape
        
        # Initialize action statistics if not done
        if self.action_stats['mean'] is None:
            self.action_stats['mean'] = torch.zeros(action_dim, device=actions.device)
            self.action_stats['std'] = torch.ones(action_dim, device=actions.device)
        
        # Update statistics (simple moving average)
        alpha = 0.01  # Learning rate for statistics update
        with torch.no_grad():
            batch_mean = actions.mean(dim=0)
            batch_std = actions.std(dim=0)
            
            self.action_stats['mean'] = (1 - alpha) * self.action_stats['mean'] + alpha * batch_mean
            self.action_stats['std'] = (1 - alpha) * self.action_stats['std'] + alpha * batch_std
        
        # Compute approximate log_probs using current statistics
        # Use a more conservative approach to avoid extreme values
        normalized_actions = (actions - self.action_stats['mean']) / (self.action_stats['std'] + 1e-8)
        
        # Clip normalized actions to prevent extreme values
        normalized_actions = torch.clamp(normalized_actions, -3.0, 3.0)
        
        # Compute log_probs with clipping to reasonable range
        # For 6D actions, typical range should be around -18 to 0
        log_probs = -0.5 * (normalized_actions ** 2).sum(dim=-1)
        
        # Clip log_probs to reasonable range to prevent TD target explosion
        log_probs = torch.clamp(log_probs, -20.0, 0.0)
        
        return log_probs
    
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
        # Optionally initialize critic from a warmup checkpoint if provided
        if getattr(self.config, "critic_init_state_path", None):
            ckpt_path = Path(self.config.critic_init_state_path)
            if ckpt_path.exists():
                try:
                    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                    if isinstance(state, dict) and "critic_state_dict" in state:
                        self.critic.load_state_dict(state["critic_state_dict"], strict=False)
                    else:
                        # Allow loading a raw state_dict as well
                        self.critic.load_state_dict(state, strict=False)
                except Exception as exc:
                    # Do not fail training if loading fails; continue with random init
                    print(f"[SmolVLASAC] Warning: failed to load critic init from {ckpt_path}: {exc}")
    
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
        """Get optimizer parameters for different components (compatible with official SAC)"""
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
        
        # Override config with our settings for RL training
        smolvla_policy.config.train_expert_only = False  # Enable VLM training
        smolvla_policy.config.freeze_vision_encoder = False  # Enable vision encoder training
        
        # Update the model's internal config and re-initialize
        smolvla_policy.model.vlm_with_expert.train_expert_only = False
        smolvla_policy.model.vlm_with_expert.freeze_vision_encoder = False
        smolvla_policy.model.vlm_with_expert.set_requires_grad()
        
        # Debug: Check if parameters are now trainable
        trainable_params = sum(p.numel() for p in smolvla_policy.model.vlm_with_expert.parameters() if p.requires_grad)
        print(f"Debug: VLM with expert trainable parameters: {trainable_params}")
        
        # Also ensure the model is in training mode
        smolvla_policy.model.vlm_with_expert.train()
        
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
        # Filter out non-tensor fields (like 'task') for critic
        critic_observations = {
            k: v for k, v in observations.items() 
            if isinstance(v, torch.Tensor)
        }
        return self.critic.forward(critic_observations, actions, use_target, observation_features)
    
    def discrete_critic_forward(
        self, 
        observations, 
        use_target=False, 
        observation_features=None
    ) -> torch.Tensor:
        """Forward pass through discrete critic network (compatible with official SAC)"""
        if not hasattr(self, 'discrete_critic'):
            raise NotImplementedError("Discrete critic not initialized. Set num_discrete_actions in config.")
        
        discrete_critic = self.discrete_critic_target if use_target else self.discrete_critic
        # Filter out non-tensor fields (like 'task') for critic
        critic_observations = {
            k: v for k, v in observations.items() 
            if isinstance(v, torch.Tensor)
        }
        return discrete_critic(critic_observations, observation_features)
    
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
        # Ensure done is float tensor for arithmetic operations (compatibility with both RL and dataset data)
        done_float = done.float() if done.dtype == torch.bool else done
        
        # Current Q-values
        current_q_values = self.critic_forward(observations, actions, use_target=False, observation_features=observation_features)
        
        with torch.no_grad():
            # Get next actions from actor
            next_actions, next_log_probs, _ = self.actor(next_observations, next_observation_features)
            
            # Get target Q-values
            next_q_values = self.critic_forward(next_observations, next_actions, use_target=True, observation_features=next_observation_features)
            
            # Take minimum over critics (double Q-learning)
            next_q_values = torch.min(next_q_values, dim=0)[0]
            
            # Compute target - use done_float for arithmetic compatibility
            target_q_values = rewards + (1 - done_float) * self.config.discount * (
                next_q_values - self.temperature * next_log_probs
            )
            
            # Debug: Print detailed information for first batch
            if torch.rand(1).item() < 0.01:  # Only print 1% of the time to avoid spam
                print(f"🔍 TD Loss Debug Info:")
                print(f"   Rewards: {rewards[:3].cpu().numpy()}")
                print(f"   Done: {done_float[:3].cpu().numpy()}")
                print(f"   Discount: {self.config.discount}")
                print(f"   Next Q values: {next_q_values[:3].cpu().numpy()}")
                print(f"   Temperature: {self.temperature}")
                print(f"   Next log probs: {next_log_probs[:3].cpu().numpy()}")
                print(f"   Target Q values: {target_q_values[:3].cpu().numpy()}")
                print(f"   Current Q values: {current_q_values[:, :3].cpu().numpy()}")
        
        # Compute loss for each critic
        critic_losses = []
        for i in range(current_q_values.shape[0]):
            loss = nn.functional.mse_loss(current_q_values[i], target_q_values)
            critic_losses.append(loss)
        
        total_loss = torch.stack(critic_losses).mean()
        
        return total_loss
    
    def compute_loss_discrete_critic(
        self,
        observations,
        actions,
        rewards,
        next_observations,
        done,
        observation_features=None,
        next_observation_features=None,
        complementary_info=None,
    ):
        """Compute discrete critic loss using SAC algorithm (compatible with official SAC)"""
        # Ensure done is float tensor for arithmetic operations (compatibility with both RL and dataset data)
        done_float = done.float() if done.dtype == torch.bool else done
        
        # NOTE: We only want to keep the discrete action part
        # In the buffer we have the full action space (continuous + discrete)
        # We need to split them before concatenating them in the critic forward
        actions_discrete: Tensor = actions[:, -1:].clone()  # Assume discrete action is last dimension
        actions_discrete = torch.round(actions_discrete)
        actions_discrete = actions_discrete.long()

        discrete_penalties: Tensor | None = None
        if complementary_info is not None:
            discrete_penalties: Tensor | None = complementary_info.get("discrete_penalty")

        with torch.no_grad():
            # For DQN, select actions using online network, evaluate with target network
            next_discrete_qs = self.discrete_critic_forward(
                next_observations, use_target=False, observation_features=next_observation_features
            )
            best_next_discrete_action = torch.argmax(next_discrete_qs, dim=-1, keepdim=True)

            # Get target Q-values from target network
            target_next_discrete_qs = self.discrete_critic_forward(
                observations=next_observations,
                use_target=True,
                observation_features=next_observation_features,
            )

            # Use gather to select Q-values for best actions
            target_next_discrete_q = torch.gather(
                target_next_discrete_qs, dim=1, index=best_next_discrete_action
            ).squeeze(-1)

            # Compute target Q-value with Bellman equation - use done_float for arithmetic compatibility
            rewards_discrete = rewards
            if discrete_penalties is not None:
                rewards_discrete = rewards + discrete_penalties
            target_discrete_q = rewards_discrete + (1 - done_float) * self.config.discount * target_next_discrete_q

        # Get predicted Q-values for current observations
        predicted_discrete_qs = self.discrete_critic_forward(
            observations=observations, use_target=False, observation_features=observation_features
        )

        # Use gather to select Q-values for taken actions
        predicted_discrete_q = torch.gather(predicted_discrete_qs, dim=1, index=actions_discrete).squeeze(-1)

        # Compute MSE loss between predicted and target Q-values
        discrete_critic_loss = nn.functional.mse_loss(input=predicted_discrete_q, target=target_discrete_q)
        return discrete_critic_loss
    
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
        """Update target networks with exponential moving average (compatible with official SAC)"""
        tau = self.config.critic_target_update_weight
        
        # Update main critic target networks
        for target_param, param in zip(
            self.critic.critic_target.parameters(),
            self.critic.critic_ensemble.parameters(),
            strict=True,
        ):
            target_param.data.mul_(1 - tau)
            target_param.data.add_(tau * param.data)
        
        # Update discrete critic target networks if available
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
            # Extract critic-specific components
            rewards: Tensor = batch["reward"]
            next_observations: dict[str, Tensor] = batch["next_state"]
            done: Tensor = batch["done"]
            next_observation_features: Tensor = batch.get("next_observation_feature")
            complementary_info = batch.get("complementary_info")
            
            loss_discrete_critic = self.compute_loss_discrete_critic(
                observations=observations,
                actions=actions,
                rewards=rewards,
                next_observations=next_observations,
                done=done,
                observation_features=observation_features,
                next_observation_features=next_observation_features,
                complementary_info=complementary_info,
            )
            return {"loss_discrete_critic": loss_discrete_critic}
        
        else:
            raise ValueError(f"Unknown model type: {model}")