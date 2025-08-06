#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig, OptimizerConfig
from lerobot.optim.schedulers import LRSchedulerConfig


@PreTrainedConfig.register_subclass(name="smolvla_reward_classifier")
@dataclass
class SmolVLARewardClassifierConfig(PreTrainedConfig):
    """Configuration for SmolVLA-based Reward Classifier model."""

    name: str = "smolvla_reward_classifier"
    num_classes: int = 2  # Binary classification: success/failure
    use_ignore_label: bool = True  # Support ignore label for uncertain cases
    ignore_label_value: int = -1  # Value used for ignore label
    
    # SmolVLA-specific configs
    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    load_vlm_weights: bool = True
    freeze_vlm: bool = True  # Freeze VLM weights during training
    
    # Task description for language guidance
    task_description: str = "Is the robot successfully completing the task?"
    use_task_description: bool = True
    
    # Classifier head configs
    hidden_dim: int = 512
    dropout_rate: float = 0.1
    
    # Training configs
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    device: str = "cuda"
    
    # Image processing
    image_size: tuple = (224, 224)  # Standard size for VLM
    
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
        }
    )

    @property
    def observation_delta_indices(self) -> list | None:
        return None

    @property
    def action_delta_indices(self) -> list | None:
        return None

    @property
    def reward_delta_indices(self) -> list | None:
        return None

    def get_optimizer_preset(self) -> OptimizerConfig:
        return AdamWConfig(
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            grad_clip_norm=self.grad_clip_norm,
        )

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        return None

    def validate_features(self) -> None:
        """Validate feature configurations."""
        has_image = any(key.startswith("observation.image") for key in self.input_features)
        if not has_image:
            raise ValueError(
                "You must provide an image observation (key starting with 'observation.image') in the input features"
            )