"""
Model configurations for different models to test.
"""

from typing import Dict, Any
import torch

MODEL_CONFIGS = {
    "qwen-3b": {
        "model_name": "Qwen/Qwen2.5-3B",
        "torch_dtype": torch.float16,
        "description": "Qwen 2.5 3B - Small, fast, efficient",
    },
    "qwen-7b": {
        "model_name": "Qwen/Qwen2.5-7B",
        "torch_dtype": torch.float16,
        "description": "Qwen 2.5 7B - Larger, high quality",
    },
    "llama-7b": {
        "model_name": "meta-llama/Llama-2-7b-hf",
        "torch_dtype": torch.float16,
        "description": "Llama 2 7B - Gated model (requires HF access)",
        "gated": True,
    },
    "llama-3-8b": {
        "model_name": "meta-llama/Llama-3.2-3B",
        "torch_dtype": torch.float16,
        "description": "Llama 3.2 3B - Newer ungated model",
    },
    "mistral-7b": {
        "model_name": "mistralai/Mistral-7B-v0.1",
        "torch_dtype": torch.float16,
        "description": "Mistral 7B - High quality 7B model",
    },
    "gemma-7b": {
        "model_name": "google/gemma-7b",
        "torch_dtype": torch.bfloat16,
        "description": "Gemma 7B - Gated model (requires HF access)",
        "gated": True,
    },
    "phi-3": {
        "model_name": "microsoft/Phi-3-mini-4k-instruct",
        "torch_dtype": torch.float16,
        "description": "Phi-3 Mini 3.8B - Microsoft's small model",
    },
}


def get_model_config(model_key: str) -> Dict[str, Any]:
    """
    Get configuration for a specific model.

    Args:
        model_key: Key for the model (e.g., "qwen-3b", "llama-7b")

    Returns:
        Dictionary with model configuration

    Raises:
        ValueError: If model_key is not recognized
    """
    if model_key not in MODEL_CONFIGS:
        available = ", ".join(MODEL_CONFIGS.keys())
        raise ValueError(f"Unknown model key: {model_key}. Available: {available}")

    return MODEL_CONFIGS[model_key]


def list_available_models() -> None:
    """Print all available model configurations."""
    print("Available models:")
    for key, config in MODEL_CONFIGS.items():
        gated_tag = " [GATED - requires HF access]" if config.get('gated', False) else ""
        print(f"  {key:15s} - {config['description']}{gated_tag}")
        print(f"  {'':15s}   {config['model_name']}")
