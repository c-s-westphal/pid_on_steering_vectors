"""
Model loading and hook management for activation extraction.
"""

import os
# Disable hf_transfer to avoid module not found errors
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelHandler:
    """Handles model loading and provides hook management for activation extraction."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B",
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.float16
    ):
        """
        Initialize the model handler.

        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on. If None, automatically selects cuda if available
            torch_dtype: Data type for model weights
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype

        logger.info(f"Loading model: {model_name}")
        logger.info(f"Device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=self.device
        )
        self.model.eval()

        # Store model architecture info
        self.num_layers = len(self.model.model.layers)
        self.hidden_size = self.model.config.hidden_size

        logger.info(f"Model loaded: {self.num_layers} layers, hidden size {self.hidden_size}")

        # Hook storage
        self.hooks = []
        self.activations = {}

    def get_layer_range(self, range_spec: str = "middle-late") -> List[int]:
        """
        Get layer indices based on specification.

        Args:
            range_spec: One of "all", "early", "middle", "late", "middle-late"

        Returns:
            List of layer indices
        """
        if range_spec == "all":
            return list(range(self.num_layers))
        elif range_spec == "early":
            return list(range(self.num_layers // 3))
        elif range_spec == "middle":
            return list(range(self.num_layers // 3, 2 * self.num_layers // 3))
        elif range_spec == "late":
            return list(range(2 * self.num_layers // 3, self.num_layers))
        elif range_spec == "middle-late":
            return list(range(self.num_layers // 3, self.num_layers))
        else:
            raise ValueError(f"Unknown range_spec: {range_spec}")

    def register_activation_hook(self, layer_idx: int, name: str):
        """
        Register a forward hook to capture activations at a specific layer.

        Args:
            layer_idx: Index of the layer to hook
            name: Name to store the activation under
        """
        def hook_fn(module, input, output):
            # For transformer layers, output is typically a tuple (hidden_states, ...)
            # We want the hidden states (residual stream)
            if isinstance(output, tuple):
                activation = output[0]
            else:
                activation = output

            # Detach and store
            self.activations[name] = activation.detach()

        layer = self.model.model.layers[layer_idx]
        hook = layer.register_forward_hook(hook_fn)
        self.hooks.append(hook)

        logger.debug(f"Registered hook at layer {layer_idx} with name '{name}'")

    def register_steering_hook(self, layer_idx: int, steering_vector: torch.Tensor, scale: float = 1.0):
        """
        Register a forward hook to add a steering vector during generation.

        Args:
            layer_idx: Index of the layer to apply steering
            steering_vector: The vector to add to activations
            scale: Scaling factor for the steering vector
        """
        # Convert to model's dtype and device
        steering_vector = steering_vector.to(device=self.device, dtype=self.torch_dtype)

        def steering_hook_fn(module, input, output):
            # Output is typically (hidden_states, ...)
            if isinstance(output, tuple):
                hidden_states = output[0]
                # Add steering vector to all positions
                # steering_vector shape: (hidden_size,)
                # hidden_states shape: (batch_size, seq_len, hidden_size)
                steered = hidden_states + scale * steering_vector.view(1, 1, -1)
                return (steered,) + output[1:]
            else:
                steered = output + scale * steering_vector.view(1, 1, -1)
                return steered

        layer = self.model.model.layers[layer_idx]
        hook = layer.register_forward_hook(steering_hook_fn)
        self.hooks.append(hook)

        logger.debug(f"Registered steering hook at layer {layer_idx} with scale {scale}")

    def register_dynamic_mlp_steering_hook(
        self,
        layer_idx: int,
        mlp_model: torch.nn.Module,
        dog_direction: torch.Tensor,
        bridge_direction: torch.Tensor,
        both_direction: torch.Tensor,
        scale: float = 1.0,
        use_float32: bool = True
    ):
        """
        Register a dynamic steering hook that uses MLP predictions to weight directions.

        At each forward pass:
        1. Extract current hidden state
        2. Run MLP to predict p(dog), p(bridge), p(both)
        3. Compute steering vector = p(dog)*dog_dir + p(bridge)*bridge_dir + p(both)*both_dir
        4. Add weighted steering vector to activations

        Args:
            layer_idx: Index of the layer to apply steering
            mlp_model: Trained MLP for 4-class prediction
            dog_direction: Steering direction for dog concept
            bridge_direction: Steering direction for bridge concept
            both_direction: Steering direction for both concepts
            scale: Global scaling factor
            use_float32: Whether MLP expects float32 inputs
        """
        # Convert directions to model's device/dtype
        dog_direction = dog_direction.to(device=self.device, dtype=self.torch_dtype)
        bridge_direction = bridge_direction.to(device=self.device, dtype=self.torch_dtype)
        both_direction = both_direction.to(device=self.device, dtype=self.torch_dtype)

        mlp_model = mlp_model.to(self.device)
        mlp_model.eval()

        def dynamic_steering_hook_fn(module, input, output):
            # Output is typically (hidden_states, ...)
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # hidden_states shape: (batch_size, seq_len, hidden_size)
            batch_size, seq_len, hidden_size = hidden_states.shape

            # Reshape for MLP: (batch_size * seq_len, hidden_size)
            flat_states = hidden_states.reshape(-1, hidden_size)

            # Convert to float32 if needed for MLP
            mlp_input = flat_states.float() if use_float32 else flat_states

            # Get MLP predictions
            with torch.no_grad():
                logits = mlp_model(mlp_input)
                probs = torch.softmax(logits, dim=1)

            # Extract probabilities for each concept
            # probs shape: (batch_size * seq_len, 4) where 4 = [neither, dog, bridge, both]
            p_dog = probs[:, 1:2]      # Shape: (batch_size * seq_len, 1)
            p_bridge = probs[:, 2:3]   # Shape: (batch_size * seq_len, 1)
            p_both = probs[:, 3:4]     # Shape: (batch_size * seq_len, 1)

            # Compute weighted steering vector for each position
            # Broadcasting: (batch*seq, 1) * (1, hidden_size) = (batch*seq, hidden_size)
            steering_vector = (
                p_dog * dog_direction.unsqueeze(0) +
                p_bridge * bridge_direction.unsqueeze(0) +
                p_both * both_direction.unsqueeze(0)
            )

            # Reshape back to (batch_size, seq_len, hidden_size)
            steering_vector = steering_vector.reshape(batch_size, seq_len, hidden_size)

            # Convert back to original dtype if needed
            steering_vector = steering_vector.to(dtype=hidden_states.dtype)

            # Apply steering
            steered = hidden_states + scale * steering_vector

            if isinstance(output, tuple):
                return (steered,) + output[1:]
            else:
                return steered

        layer = self.model.model.layers[layer_idx]
        hook = layer.register_forward_hook(dynamic_steering_hook_fn)
        self.hooks.append(hook)

        logger.debug(f"Registered DYNAMIC MLP steering hook at layer {layer_idx} with scale {scale}")

    def clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}
        logger.debug("Cleared all hooks")

    def get_activation(self, name: str) -> Optional[torch.Tensor]:
        """Retrieve stored activation by name."""
        return self.activations.get(name)

    def clear_activations(self):
        """Clear stored activations."""
        self.activations = {}

    def __del__(self):
        """Cleanup hooks on deletion."""
        if hasattr(self, 'hooks'):
            self.clear_hooks()
