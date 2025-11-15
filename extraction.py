"""
Activation extraction functionality.
"""

import torch
from typing import List, Dict, Optional
import logging
from models import ModelHandler

logger = logging.getLogger(__name__)


class ActivationExtractor:
    """Extracts activations from model layers."""

    def __init__(self, model_handler: ModelHandler):
        """
        Initialize the activation extractor.

        Args:
            model_handler: ModelHandler instance with loaded model
        """
        self.model_handler = model_handler

    def extract_activations(
        self,
        prompts: List[str],
        layer_idx: int,
        position: str = "last"
    ) -> torch.Tensor:
        """
        Extract activations from a list of prompts at a specific layer.

        Args:
            prompts: List of text prompts
            layer_idx: Layer index to extract from
            position: Which token position to extract ("last", "first", "all")

        Returns:
            Tensor of shape (num_prompts, hidden_size) if position is "last" or "first"
            Tensor of shape (total_tokens, hidden_size) if position is "all"
        """
        self.model_handler.clear_hooks()
        self.model_handler.clear_activations()

        activations = []

        for i, prompt in enumerate(prompts):
            # Register hook for this layer
            hook_name = f"layer_{layer_idx}_prompt_{i}"
            self.model_handler.register_activation_hook(layer_idx, hook_name)

            # Tokenize
            inputs = self.model_handler.tokenizer(
                prompt,
                return_tensors="pt",
                padding=False,
                truncation=True
            ).to(self.model_handler.device)

            # Forward pass
            with torch.no_grad():
                _ = self.model_handler.model(**inputs)

            # Get activation
            activation = self.model_handler.get_activation(hook_name)

            if activation is None:
                logger.error(f"Failed to extract activation for prompt {i}")
                continue

            # Extract based on position
            # activation shape: (1, seq_len, hidden_size)
            if position == "last":
                # Get the last token's activation
                act = activation[0, -1, :].cpu()
            elif position == "first":
                act = activation[0, 0, :].cpu()
            elif position == "all":
                # Get all tokens
                act = activation[0, :, :].cpu()
            else:
                raise ValueError(f"Unknown position: {position}")

            if position == "all":
                # For "all", we want to collect all tokens across all prompts
                activations.append(act)
            else:
                # For specific positions, one vector per prompt
                activations.append(act)

            # Clear this hook
            self.model_handler.clear_hooks()

        # Stack activations
        if position == "all":
            # Concatenate all tokens from all prompts
            return torch.cat(activations, dim=0)
        else:
            # Stack one vector per prompt
            return torch.stack(activations, dim=0)

    def extract_mean_activation(
        self,
        prompts: List[str],
        layer_idx: int,
        position: str = "last"
    ) -> torch.Tensor:
        """
        Extract and average activations across prompts.

        Args:
            prompts: List of text prompts
            layer_idx: Layer index to extract from
            position: Which token position to extract

        Returns:
            Mean activation vector of shape (hidden_size,)
        """
        activations = self.extract_activations(prompts, layer_idx, position)
        return activations.mean(dim=0)

    def compute_null_vector(self, layer_idx: int) -> torch.Tensor:
        """
        Compute null vector by averaging all token embeddings and passing through model.

        Args:
            layer_idx: Layer index to extract null vector from

        Returns:
            Null vector of shape (hidden_size,)
        """
        logger.info(f"Computing null vector at layer {layer_idx}")

        # Get all token embeddings
        embedding_layer = self.model_handler.model.model.embed_tokens
        vocab_size = self.model_handler.tokenizer.vocab_size

        # Average all token embeddings
        # embedding_layer.weight shape: (vocab_size, hidden_size)
        avg_embedding = embedding_layer.weight[:vocab_size].mean(dim=0)  # (hidden_size,)

        # Create a dummy input with this average embedding
        # We need to pass it through the model up to layer_idx
        # We'll do this by creating a single-token input and replacing its embedding

        # Create a dummy token (just use token id 0)
        dummy_input_ids = torch.tensor([[0]], device=self.model_handler.device)

        # Register hook to capture activation at target layer
        hook_name = "null_vector"
        self.model_handler.clear_hooks()
        self.model_handler.clear_activations()
        self.model_handler.register_activation_hook(layer_idx, hook_name)

        # We need to forward pass with the average embedding
        # We'll use a hook to replace the embedding
        def replace_embedding_hook(module, input, output):
            # Replace the embedding with our average
            return avg_embedding.unsqueeze(0).unsqueeze(0)  # (1, 1, hidden_size)

        embedding_hook = embedding_layer.register_forward_hook(replace_embedding_hook)

        try:
            with torch.no_grad():
                _ = self.model_handler.model(dummy_input_ids)

            # Get the activation at the target layer
            activation = self.model_handler.get_activation(hook_name)

            if activation is None:
                raise RuntimeError("Failed to extract null vector activation")

            # activation shape: (1, 1, hidden_size)
            null_vector = activation[0, 0, :].cpu()

        finally:
            embedding_hook.remove()
            self.model_handler.clear_hooks()

        logger.info("Null vector computed successfully")
        return null_vector
