"""
Steering Vectors Research Library

A PyTorch-based library for research on activation-based steering vectors
for language models.
"""

__version__ = "0.1.0"

from .models import ModelHandler
from .extraction import ActivationExtractor
from .vectors import VectorComputer, SteeringVector
from .steering import SteeredGenerator
from .evaluation import SteeringEvaluator
from .data import (
    ConceptDataset,
    DatasetBuilder,
    PromptTemplate,
    EvaluationPrompts
)
from .utils import (
    cosine_similarity,
    vector_norm,
    compare_vectors,
    analyze_vector_statistics,
    ExperimentLogger
)

__all__ = [
    # Core classes
    "ModelHandler",
    "ActivationExtractor",
    "VectorComputer",
    "SteeringVector",
    "SteeredGenerator",
    "SteeringEvaluator",

    # Data utilities
    "ConceptDataset",
    "DatasetBuilder",
    "PromptTemplate",
    "EvaluationPrompts",

    # Utilities
    "cosine_similarity",
    "vector_norm",
    "compare_vectors",
    "analyze_vector_statistics",
    "ExperimentLogger",
]
