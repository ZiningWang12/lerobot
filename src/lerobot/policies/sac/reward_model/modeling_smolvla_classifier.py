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

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoProcessor, AutoModelForImageTextToText

from lerobot.constants import OBS_IMAGE
from lerobot.policies.normalize import Normalize, Unnormalize
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.sac.reward_model.configuration_smolvla_classifier import SmolVLARewardClassifierConfig


@dataclass
class ClassifierOutput:
    """Output structure for the classifier."""
    logits: Tensor
    probabilities: Tensor
    hidden_states: Tensor


class SmolVLARewardClassifier(PreTrainedPolicy):
    """SmolVLA-based reward classifier for HIL-SERL."""

    name = "smolvla_reward_classifier"
    config_class = SmolVLARewardClassifierConfig

    def __init__(
        self,
        config: SmolVLARewardClassifierConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        super().__init__(config)
        self.config = config

        # Initialize normalization
        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )

        # Load SmolVLM model and processor
        self.processor = AutoProcessor.from_pretrained(config.vlm_model_name)
        
        if config.load_vlm_weights:
            self.vlm = AutoModelForImageTextToText.from_pretrained(
                config.vlm_model_name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
        else:
            # Initialize from config only (for training from scratch)
            from transformers import AutoConfig
            vlm_config = AutoConfig.from_pretrained(config.vlm_model_name)
            self.vlm = AutoModelForImageTextToText.from_config(vlm_config)

        # Freeze VLM weights if specified
        if config.freeze_vlm:
            for param in self.vlm.parameters():
                param.requires_grad = False
            logging.info("Frozen SmolVLM weights for reward classifier training")

        # Get VLM output dimension
        self.vlm_hidden_size = self.vlm.config.text_config.hidden_size

        # Extract image keys from input_features
        self.image_keys = [
            key.replace(".", "_") for key in config.input_features if key.startswith(OBS_IMAGE)
        ]

        # Prepare task description tokens if enabled
        if config.use_task_description:
            self.task_text = config.task_description
        else:
            self.task_text = ""

        # Build classifier head
        self._build_classifier_head()

    def _build_classifier_head(self):
        """Build the classification head on top of VLM features."""
        classifier_layers = []
        
        # Multi-layer perceptron for classification
        classifier_layers.extend([
            nn.Linear(self.vlm_hidden_size, self.config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.config.dropout_rate),
        ])
        
        # Output layer
        if self.config.num_classes == 2:
            # Binary classification - single output
            classifier_layers.append(nn.Linear(self.config.hidden_dim // 2, 1))
        else:
            # Multi-class classification
            classifier_layers.append(nn.Linear(self.config.hidden_dim // 2, self.config.num_classes))
        
        self.classifier_head = nn.Sequential(*classifier_layers)

    def _extract_vlm_features(self, images: list[Tensor]) -> Tensor:
        """Extract features from SmolVLM."""
        # Prepare inputs for the VLM
        # Concatenate multiple camera views if available
        if len(images) == 1:
            image = images[0]
        else:
            # For multiple cameras, we'll process them separately and average features
            # This is a simple approach - could be improved with attention mechanisms
            image = images[0]  # Use first camera for now
            # TODO: Implement proper multi-camera fusion
        
        # Prepare text input
        if self.config.use_task_description:
            text_input = self.task_text
        else:
            text_input = ""
        
        # Process inputs through the processor
        inputs = self.processor(
            images=image,
            text=text_input,
            return_tensors="pt",
            padding=True,
        )
        
        # Move to appropriate device
        inputs = {k: v.to(self.vlm.device) for k, v in inputs.items()}
        
        # Get VLM outputs
        with torch.no_grad() if self.config.freeze_vlm else torch.enable_grad():
            outputs = self.vlm.model(**inputs, output_hidden_states=True)
            
        # Extract last hidden states (text embeddings that have attended to image)
        # Use the embedding of the last token as the representation
        hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Pool the sequence dimension (simple mean pooling)
        pooled_features = hidden_states.mean(dim=1)  # [batch_size, hidden_size]
        
        return pooled_features

    def extract_images_and_labels(self, batch: dict[str, Tensor]) -> tuple[list, Tensor]:
        """Extract images and labels from batch."""
        images = []
        for image_key in self.image_keys:
            if image_key in batch:
                img = batch[image_key]
                if img.dim() == 3:
                    img = img.unsqueeze(0)  # Add batch dimension
                images.append(img)
        
        # Extract labels (assuming they're in the batch)
        if "reward" in batch:
            labels = batch["reward"].float()
        elif "next.reward" in batch:
            labels = batch["next.reward"].float()
        elif "success" in batch:
            labels = batch["success"].float()
        else:
            # Fallback - assume last element in batch is labels
            labels = torch.zeros(images[0].shape[0])  # Dummy labels for inference
            
        return images, labels

    def predict(self, images: list[Tensor]) -> ClassifierOutput:
        """Forward pass for inference."""
        # Extract VLM features
        vlm_features = self._extract_vlm_features(images)
        
        # Pass through classifier head
        logits = self.classifier_head(vlm_features)
        
        # Calculate probabilities
        if self.config.num_classes == 2:
            logits = logits.squeeze(-1)
            probabilities = torch.sigmoid(logits)
        else:
            probabilities = torch.softmax(logits, dim=-1)
        
        return ClassifierOutput(
            logits=logits,
            probabilities=probabilities,
            hidden_states=vlm_features
        )

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Any]]:
        """Forward pass for training."""
        # Normalize inputs if needed
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)
        
        # Extract images and labels
        images, labels = self.extract_images_and_labels(batch)
        
        # Handle ignore labels if enabled
        if self.config.use_ignore_label:
            # Create mask for valid (non-ignore) labels
            valid_mask = labels != self.config.ignore_label_value
            
            if valid_mask.sum() == 0:
                # All labels are ignore labels, return zero loss
                dummy_loss = torch.tensor(0.0, device=labels.device, requires_grad=True)
                output_dict = {
                    "accuracy": 0.0,
                    "correct": 0,
                    "total": 0,
                    "ignored_samples": len(labels),
                }
                return dummy_loss, output_dict
            
            # Filter out ignore labels for training
            images_filtered = [img[valid_mask] for img in images]
            labels_filtered = labels[valid_mask]
        else:
            images_filtered = images
            labels_filtered = labels
            valid_mask = torch.ones_like(labels, dtype=torch.bool)
        
        # Get predictions only for valid samples
        outputs = self.predict(images_filtered)
        
        # Calculate loss
        if self.config.num_classes == 2:
            # Binary classification
            loss = nn.functional.binary_cross_entropy_with_logits(outputs.logits, labels_filtered)
            predictions = (torch.sigmoid(outputs.logits) > 0.5).float()
        else:
            # Multi-class classification
            loss = nn.functional.cross_entropy(outputs.logits, labels_filtered.long())
            predictions = torch.argmax(outputs.logits, dim=1)
        
        # Calculate accuracy only for valid samples
        correct = (predictions == labels_filtered).sum().item()
        total_valid = labels_filtered.size(0)
        accuracy = 100 * correct / total_valid if total_valid > 0 else 0.0
        
        # Return loss and metrics
        output_dict = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total_valid,
            "ignored_samples": len(labels) - total_valid if self.config.use_ignore_label else 0,
        }
        
        return loss, output_dict

    def predict_reward(self, batch: dict[str, Tensor], threshold: float = 0.5) -> float:
        """Predict reward for a single observation."""
        images = []
        for key in batch:
            if "image" in key:
                img = batch[key]
                if img.dim() == 3:
                    img = img.unsqueeze(0)
                images.append(img)
        
        if not images:
            return 0.0
        
        with torch.no_grad():
            outputs = self.predict(images)
            
            if self.config.num_classes == 2:
                # Binary classification
                probability = outputs.probabilities.item()
                return 1.0 if probability > threshold else 0.0
            else:
                # Multi-class - return probability of success class (assume class 1)
                probability = outputs.probabilities[0, 1].item()
                return 1.0 if probability > threshold else 0.0