"""
Utility functions and helpers.
"""

import torch
from typing import List, Dict, Any, Optional
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Cosine similarity score
    """
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def vector_norm(v: torch.Tensor) -> float:
    """
    Compute L2 norm of a vector.

    Args:
        v: Vector

    Returns:
        L2 norm
    """
    return torch.norm(v).item()


def compare_vectors(
    vectors: Dict[str, torch.Tensor],
    metric: str = "cosine"
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple vectors pairwise.

    Args:
        vectors: Dictionary mapping names to vectors
        metric: Comparison metric ("cosine" or "euclidean")

    Returns:
        Dictionary of pairwise comparisons
    """
    comparisons = {}
    names = list(vectors.keys())

    for i, name_a in enumerate(names):
        comparisons[name_a] = {}
        for name_b in names[i:]:
            if metric == "cosine":
                score = cosine_similarity(vectors[name_a], vectors[name_b])
            elif metric == "euclidean":
                score = torch.dist(vectors[name_a], vectors[name_b]).item()
            else:
                raise ValueError(f"Unknown metric: {metric}")

            comparisons[name_a][name_b] = score
            if name_a != name_b:
                comparisons.setdefault(name_b, {})[name_a] = score

    return comparisons


def print_comparison_matrix(comparisons: Dict[str, Dict[str, float]]):
    """
    Pretty print a comparison matrix.

    Args:
        comparisons: Output from compare_vectors
    """
    names = list(comparisons.keys())

    # Print header
    print(f"{'':20s}", end="")
    for name in names:
        print(f"{name[:18]:20s}", end="")
    print()

    # Print rows
    for name_a in names:
        print(f"{name_a[:18]:20s}", end="")
        for name_b in names:
            score = comparisons[name_a].get(name_b, 0.0)
            print(f"{score:20.4f}", end="")
        print()


def analyze_vector_statistics(vector: torch.Tensor) -> Dict[str, float]:
    """
    Compute statistics about a vector.

    Args:
        vector: Vector to analyze

    Returns:
        Dictionary of statistics
    """
    return {
        'mean': vector.mean().item(),
        'std': vector.std().item(),
        'min': vector.min().item(),
        'max': vector.max().item(),
        'norm': torch.norm(vector).item(),
        'sparsity': (vector.abs() < 1e-6).float().mean().item(),  # Fraction near zero
        'positive_fraction': (vector > 0).float().mean().item(),
    }


def load_json(filepath: str) -> Dict[str, Any]:
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], filepath: str):
    """Save data to JSON file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved JSON to {filepath}")


def format_generation_comparison(comparison: Dict[str, Any]) -> str:
    """
    Format a generation comparison for pretty printing.

    Args:
        comparison: Output from SteeredGenerator.compare_generations

    Returns:
        Formatted string
    """
    lines = []
    lines.append("=" * 80)
    lines.append(f"Prompt: {comparison['prompt']}")
    lines.append("=" * 80)

    lines.append("\nBaseline:")
    lines.append("-" * 80)
    lines.append(comparison['baseline']['text'])

    if 'steered' in comparison:
        for steered in comparison['steered']:
            lines.append(f"\nSteered ({steered['steering_info']['concept']}, "
                        f"scale={steered['steering_info']['scale']}):")
            lines.append("-" * 80)
            lines.append(steered['text'])

    lines.append("=" * 80)

    return "\n".join(lines)


def batch_process_with_progress(
    items: List[Any],
    process_fn: callable,
    batch_size: int = 1,
    desc: str = "Processing"
) -> List[Any]:
    """
    Process items in batches with progress logging.

    Args:
        items: Items to process
        process_fn: Function to apply to each item
        batch_size: Number of items per batch
        desc: Description for logging

    Returns:
        List of processed results
    """
    results = []
    total = len(items)

    for i in range(0, total, batch_size):
        batch = items[i:i+batch_size]
        logger.info(f"{desc}: {i+1}-{min(i+batch_size, total)}/{total}")

        batch_results = [process_fn(item) for item in batch]
        results.extend(batch_results)

    return results


class ExperimentLogger:
    """Logger for experiment results."""

    def __init__(self, experiment_name: str, output_dir: str = "outputs/experiments"):
        """
        Initialize experiment logger.

        Args:
            experiment_name: Name of the experiment
            output_dir: Directory to save results
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir) / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results = {
            'experiment_name': experiment_name,
            'runs': []
        }

        logger.info(f"Initialized experiment: {experiment_name}")
        logger.info(f"Output directory: {self.output_dir}")

    def log_run(self, run_name: str, data: Dict[str, Any]):
        """
        Log a single run.

        Args:
            run_name: Name of this run
            data: Data to log
        """
        run_data = {
            'run_name': run_name,
            'data': data
        }
        self.results['runs'].append(run_data)

        # Save individual run
        run_file = self.output_dir / f"{run_name}.json"
        save_json(data, str(run_file))

        logger.info(f"Logged run: {run_name}")

    def save_summary(self):
        """Save summary of all runs."""
        summary_file = self.output_dir / "summary.json"
        save_json(self.results, str(summary_file))
        logger.info(f"Saved experiment summary to {summary_file}")

    def get_output_path(self, filename: str) -> Path:
        """Get path for saving output file."""
        return self.output_dir / filename
