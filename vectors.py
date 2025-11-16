"""
Steering vector computation and combination methods.
"""

import torch
from typing import List, Optional, Dict, Literal
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


CombinationMethod = Literal["diff", "mean", "max", "min", "rms_signed", "abs_diff"]


def compute_rms_signed(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute f(a,b) = sign(a+b) · sqrt((a² + b²) / 2)

    This is applied element-wise across the vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Combined vector using RMS with sign preservation
    """
    sign = torch.sign(a + b)
    rms = torch.sqrt((a ** 2 + b ** 2) / 2)
    return sign * rms


class SteeringVector:
    """Represents a steering vector with metadata."""

    def __init__(
        self,
        vector: torch.Tensor,
        layer_idx: int,
        concept: Optional[str] = None,
        method: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Initialize a steering vector.

        Args:
            vector: The steering vector tensor
            layer_idx: Layer index this vector was extracted from
            concept: Description of the concept (e.g., "dogs")
            method: Method used to create this vector (e.g., "diff_from_null")
            metadata: Additional metadata dictionary
        """
        # Store vector on CPU in float32 for consistency
        # It will be converted to model's dtype when used for steering
        if vector.is_cuda:
            self.vector = vector.cpu().float()
        else:
            self.vector = vector.float()
        self.layer_idx = layer_idx
        self.concept = concept
        self.method = method
        self.metadata = metadata or {}

    def save(self, path: str):
        """Save steering vector to disk."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save vector and metadata
        torch.save({
            'vector': self.vector,
            'layer_idx': self.layer_idx,
            'concept': self.concept,
            'method': self.method,
            'metadata': self.metadata
        }, save_path)

        logger.info(f"Saved steering vector to {save_path}")

    @classmethod
    def load(cls, path: str) -> 'SteeringVector':
        """Load steering vector from disk."""
        data = torch.load(path)
        logger.info(f"Loaded steering vector from {path}")
        return cls(
            vector=data['vector'],
            layer_idx=data['layer_idx'],
            concept=data.get('concept'),
            method=data.get('method'),
            metadata=data.get('metadata', {})
        )

    def __repr__(self):
        return (f"SteeringVector(concept='{self.concept}', layer={self.layer_idx}, "
                f"method='{self.method}', shape={tuple(self.vector.shape)})")


class VectorComputer:
    """Computes steering vectors using various methods."""

    @staticmethod
    def from_diff_with_null(
        concept_activation: torch.Tensor,
        null_vector: torch.Tensor,
        layer_idx: int,
        concept: str
    ) -> SteeringVector:
        """
        Create steering vector by diffing concept activation from null vector.

        Args:
            concept_activation: Mean activation for the concept
            null_vector: Null vector (average embedding space)
            layer_idx: Layer index
            concept: Concept name

        Returns:
            SteeringVector instance
        """
        vector = concept_activation - null_vector
        return SteeringVector(
            vector=vector,
            layer_idx=layer_idx,
            concept=concept,
            method="diff_from_null"
        )

    @staticmethod
    def from_traditional_diff(
        activation_a: torch.Tensor,
        activation_b: torch.Tensor,
        layer_idx: int,
        concept: str
    ) -> SteeringVector:
        """
        Create steering vector using traditional contrastive method (a - b).

        Args:
            activation_a: Mean activation for concept A
            activation_b: Mean activation for concept B
            layer_idx: Layer index
            concept: Description (e.g., "dogs vs cats")

        Returns:
            SteeringVector instance
        """
        vector = activation_a - activation_b
        return SteeringVector(
            vector=vector,
            layer_idx=layer_idx,
            concept=concept,
            method="traditional_diff"
        )

    @staticmethod
    def combine_vectors(
        vector_a: SteeringVector,
        vector_b: SteeringVector,
        method: CombinationMethod = "mean",
        concept: Optional[str] = None
    ) -> SteeringVector:
        """
        Combine two steering vectors using specified method.

        Args:
            vector_a: First steering vector
            vector_b: Second steering vector
            method: Combination method ("diff", "mean", "max", "min", "rms_signed", "abs_diff")
            concept: Optional concept description for combined vector

        Returns:
            Combined SteeringVector

        Raises:
            ValueError: If vectors are from different layers
        """
        if vector_a.layer_idx != vector_b.layer_idx:
            raise ValueError(
                f"Cannot combine vectors from different layers: "
                f"{vector_a.layer_idx} vs {vector_b.layer_idx}"
            )

        a = vector_a.vector
        b = vector_b.vector

        if method == "diff":
            combined = a - b
            method_name = "diff"
        elif method == "mean":
            combined = (a + b) / 2
            method_name = "mean"
        elif method == "max":
            # Max by absolute value, preserving the sign of the larger magnitude element
            abs_a = torch.abs(a)
            abs_b = torch.abs(b)
            # Where |a| > |b|, take a, otherwise take b
            combined = torch.where(abs_a > abs_b, a, b)
            method_name = "elementwise_max_abs"
        elif method == "min":
            # Min by absolute value, preserving the sign of the smaller magnitude element
            abs_a = torch.abs(a)
            abs_b = torch.abs(b)
            # Where |a| < |b|, take a, otherwise take b
            combined = torch.where(abs_a < abs_b, a, b)
            method_name = "elementwise_min_abs"
        elif method == "rms_signed":
            combined = compute_rms_signed(a, b)
            method_name = "rms_signed"
        elif method == "abs_diff":
            # Absolute difference: |a| - |b|, preserving sign from larger component
            abs_a = torch.abs(a)
            abs_b = torch.abs(b)
            diff_magnitude = abs_a - abs_b
            # Preserve sign from whichever had larger absolute value
            combined = torch.where(abs_a > abs_b,
                                  torch.sign(a) * diff_magnitude,
                                  torch.sign(b) * diff_magnitude)
            method_name = "abs_diff"
        else:
            raise ValueError(f"Unknown combination method: {method}")

        if concept is None:
            concept = f"{vector_a.concept} + {vector_b.concept} ({method_name})"

        return SteeringVector(
            vector=combined,
            layer_idx=vector_a.layer_idx,
            concept=concept,
            method=method_name,
            metadata={
                'source_vectors': [
                    {'concept': vector_a.concept, 'method': vector_a.method},
                    {'concept': vector_b.concept, 'method': vector_b.method}
                ],
                'combination_method': method
            }
        )

    @staticmethod
    def combine_multiple_vectors(
        vectors: List[SteeringVector],
        method: CombinationMethod = "mean",
        concept: Optional[str] = None
    ) -> SteeringVector:
        """
        Combine multiple steering vectors.

        Args:
            vectors: List of steering vectors to combine
            method: Combination method
            concept: Optional concept description

        Returns:
            Combined SteeringVector
        """
        if len(vectors) < 2:
            raise ValueError("Need at least 2 vectors to combine")

        # Check all from same layer
        layer_idx = vectors[0].layer_idx
        if not all(v.layer_idx == layer_idx for v in vectors):
            raise ValueError("All vectors must be from the same layer")

        # Combine pairwise for binary operations, or all at once for mean
        if method == "mean":
            combined = torch.stack([v.vector for v in vectors]).mean(dim=0)
            method_name = "mean"
        elif method in ["diff", "max", "min", "rms_signed", "abs_diff"]:
            # For binary operations, combine sequentially
            result = vectors[0]
            for v in vectors[1:]:
                result = VectorComputer.combine_vectors(result, v, method=method)
            return result
        else:
            raise ValueError(f"Unknown combination method: {method}")

        if concept is None:
            concept = f"combined_{len(vectors)}_vectors ({method_name})"

        return SteeringVector(
            vector=combined,
            layer_idx=layer_idx,
            concept=concept,
            method=method_name,
            metadata={
                'source_vectors': [
                    {'concept': v.concept, 'method': v.method} for v in vectors
                ],
                'combination_method': method
            }
        )
