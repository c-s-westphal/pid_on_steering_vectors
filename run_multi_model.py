#!/usr/bin/env python3
"""
Run steering vector experiments across multiple models.

This script runs the same steering vector experiments on different models
to compare how steering works across model architectures and sizes.
"""

import argparse
import logging
import sys
from pathlib import Path
import torch
from datetime import datetime

from model_configs import MODEL_CONFIGS, get_model_config, list_available_models
from models import ModelHandler
from extraction import ActivationExtractor
from vectors import VectorComputer, SteeringVector
from steering import SteeredGenerator
from data import DatasetBuilder
from evaluation import SteeringEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_for_model(model_key: str, output_base_dir: Path):
    """
    Run steering vector experiments for a specific model.

    Args:
        model_key: Key for the model configuration
        output_base_dir: Base directory for outputs
    """
    logger.info("=" * 80)
    logger.info(f"RUNNING EXPERIMENTS FOR: {model_key.upper()}")
    logger.info("=" * 80)

    # Get model config
    config = get_model_config(model_key)
    logger.info(f"Model: {config['model_name']}")
    logger.info(f"Description: {config['description']}")

    # Create model-specific output directory
    model_output_dir = output_base_dir / model_key / datetime.now().strftime("%Y%m%d_%H%M%S")
    model_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {model_output_dir}")

    # Initialize model
    logger.info("\n" + "-" * 60)
    logger.info("Loading model...")
    logger.info("-" * 60)
    model_handler = ModelHandler(
        model_name=config['model_name'],
        torch_dtype=config['torch_dtype']
    )

    # Initialize components
    extractor = ActivationExtractor(model_handler)
    generator = SteeredGenerator(model_handler)
    evaluator = SteeringEvaluator(model_handler, generator)
    vector_computer = VectorComputer()

    # Determine target layer (middle-late range)
    target_layer = int(model_handler.num_layers * 2 / 3)
    logger.info(f"Target layer: {target_layer} (out of {model_handler.num_layers})")

    # Build datasets
    logger.info("\n" + "-" * 60)
    logger.info("Building datasets...")
    logger.info("-" * 60)
    dogs_dataset = DatasetBuilder.create_dog_dataset()
    bridge_dataset = DatasetBuilder.create_bridge_dataset()
    dogs_prompts = dogs_dataset.get_prompts(template="topic")
    bridge_prompts = bridge_dataset.get_prompts(template="topic")

    # Compute null vector (zeros method)
    logger.info("\n" + "-" * 60)
    logger.info("Computing null vector...")
    logger.info("-" * 60)
    null_vector = extractor.compute_null_vector(target_layer, method="zeros")
    logger.info(f"Null vector norm: {torch.norm(null_vector).item():.4f}")

    # Extract activations
    logger.info("\n" + "-" * 60)
    logger.info("Extracting activations...")
    logger.info("-" * 60)
    dogs_activation = extractor.extract_mean_activation(dogs_prompts, target_layer)
    bridge_activation = extractor.extract_mean_activation(bridge_prompts, target_layer)
    logger.info(f"Dogs activation norm: {torch.norm(dogs_activation).item():.4f}")
    logger.info(f"Bridge activation norm: {torch.norm(bridge_activation).item():.4f}")

    # Create steering vectors
    logger.info("\n" + "-" * 60)
    logger.info("Creating steering vectors...")
    logger.info("-" * 60)

    dogs_vector = vector_computer.from_diff_with_null(
        dogs_activation, null_vector, target_layer, "dogs"
    )
    bridge_vector = vector_computer.from_diff_with_null(
        bridge_activation, null_vector, target_layer, "bridge"
    )

    # Test all combination methods
    combinations = {
        "mean": vector_computer.combine_vectors(dogs_vector, bridge_vector, method="mean"),
        "max": vector_computer.combine_vectors(dogs_vector, bridge_vector, method="max"),
        "min": vector_computer.combine_vectors(dogs_vector, bridge_vector, method="min"),
        "rms_signed": vector_computer.combine_vectors(dogs_vector, bridge_vector, method="rms_signed"),
        "diff": vector_computer.combine_vectors(dogs_vector, bridge_vector, method="diff"),
        "abs_diff": vector_computer.combine_vectors(dogs_vector, bridge_vector, method="abs_diff"),
    }

    # Traditional contrastive
    traditional_diff = vector_computer.from_traditional_diff(
        dogs_activation, bridge_activation, target_layer, "dogs vs bridge"
    )

    # Save all vectors
    logger.info("\n" + "-" * 60)
    logger.info("Saving steering vectors...")
    logger.info("-" * 60)
    vectors_dir = model_output_dir / "steering_vectors"
    vectors_dir.mkdir(parents=True, exist_ok=True)

    dogs_vector.save(vectors_dir / "dogs_vector.pt")
    bridge_vector.save(vectors_dir / "bridge_vector.pt")
    for name, vec in combinations.items():
        vec.save(vectors_dir / f"dogs_bridge_{name}.pt")
    traditional_diff.save(vectors_dir / "traditional_diff.pt")
    logger.info(f"Saved {2 + len(combinations) + 1} vectors to {vectors_dir}")

    # Test multiple scales for ALL combination methods
    logger.info("\n" + "-" * 60)
    logger.info("Testing multiple scales for all combination methods...")
    logger.info("-" * 60)

    test_scales = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    test_prompt = "Write a short paragraph about"

    all_vectors = [
        ("Dogs", dogs_vector),
        ("Bridge", bridge_vector),
        ("Mean", combinations["mean"]),
        ("Max", combinations["max"]),
        ("Min", combinations["min"]),
        ("RMS", combinations["rms_signed"]),
        ("Diff", combinations["diff"]),
        ("AbsDiff", combinations["abs_diff"]),
        ("Traditional", traditional_diff),
    ]

    logger.info(f"Testing {len(all_vectors)} vectors at {len(test_scales)} scales each")

    # Generate baseline once
    logger.info("\nBASELINE (no steering):")
    baseline_result = generator.generate(
        prompt=test_prompt,
        steering_vector=None,
        max_new_tokens=50,
        temperature=0.7
    )
    logger.info(f"  {baseline_result['text'][:100]}...")

    # Test all vectors at all scales
    scale_results = {}
    for vec_name, vec in all_vectors:
        logger.info(f"\n{vec_name} vector (norm: {torch.norm(vec.vector).item():.2f}):")
        scale_results[vec_name] = {}

        for scale in test_scales:
            result = generator.generate(
                prompt=test_prompt,
                steering_vector=vec,
                scale=scale,
                max_new_tokens=50,
                temperature=0.7
            )
            text = result['text']
            scale_results[vec_name][scale] = text

            # Check for garbled output (multiple heuristics)
            has_control_chars = any(ord(c) < 32 and c not in '\n\t' for c in text[:50])

            # Check for repetitive punctuation patterns (sign of collapse)
            has_repetitive_punct = bool(re.search(r'[.,;:]{3,}|(\b\w+\b.*?\1.*?\1)', text[:100]))

            # Check for very short repetitive words
            words = text[:100].split()
            if len(words) > 5:
                word_set_ratio = len(set(words)) / len(words)
                is_repetitive = word_set_ratio < 0.3  # Less than 30% unique words
            else:
                is_repetitive = False

            is_garbled = has_control_chars or has_repetitive_punct or is_repetitive

            concept_mentioned = (
                'dog' in text.lower() or
                'bridge' in text.lower() or
                'golden gate' in text.lower()
            )

            # Garbled overrides concept mention
            if is_garbled:
                status = "GARBLED"
            elif concept_mentioned:
                status = "✓ concept"
            else:
                status = "neutral"
            logger.info(f"  Scale {scale:.1f}: {status:12s} | {text[:80]}...")

    # Validation: Compare null-diff vs traditional
    logger.info("\n" + "-" * 60)
    logger.info("VALIDATION: Null-diff vs Traditional contrastive")
    logger.info("-" * 60)

    test_scale = 1.5
    logger.info(f"\nUsing Dogs vector at scale {test_scale}:")

    # Dogs null-diff steering
    dogs_result = generator.generate(
        prompt="Write about",
        steering_vector=dogs_vector,
        scale=test_scale,
        max_new_tokens=40,
        temperature=0.7
    )
    logger.info(f"Null-diff method: {dogs_result['text'][:100]}...")

    # Traditional contrastive
    trad_result = generator.generate(
        prompt="Write about",
        steering_vector=traditional_diff,
        scale=test_scale,
        max_new_tokens=40,
        temperature=0.7
    )
    logger.info(f"Traditional diff: {trad_result['text'][:100]}...")

    # Save comprehensive summary
    logger.info("\n" + "-" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("-" * 60)
    summary_file = model_output_dir / "summary.txt"
    with open(summary_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("STEERING VECTORS EXPERIMENT RESULTS\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Model: {config['model_name']}\n")
        f.write(f"Description: {config['description']}\n")
        f.write(f"Layers: {model_handler.num_layers}\n")
        f.write(f"Hidden size: {model_handler.hidden_size}\n")
        f.write(f"Target layer: {target_layer}\n\n")

        f.write("=" * 80 + "\n")
        f.write("STEERING VECTORS CREATED\n")
        f.write("=" * 80 + "\n")
        f.write(f"1. Dogs (null-diff):          norm = {torch.norm(dogs_vector.vector).item():.4f}\n")
        f.write(f"2. Bridge (null-diff):        norm = {torch.norm(bridge_vector.vector).item():.4f}\n")
        f.write(f"3. Mean combination:          norm = {torch.norm(combinations['mean'].vector).item():.4f}\n")
        f.write(f"4. Max combination:           norm = {torch.norm(combinations['max'].vector).item():.4f}\n")
        f.write(f"5. Min combination:           norm = {torch.norm(combinations['min'].vector).item():.4f}\n")
        f.write(f"6. RMS-signed combination:    norm = {torch.norm(combinations['rms_signed'].vector).item():.4f}\n")
        f.write(f"7. Diff combination:          norm = {torch.norm(combinations['diff'].vector).item():.4f}\n")
        f.write(f"8. Abs-diff combination:      norm = {torch.norm(combinations['abs_diff'].vector).item():.4f}\n")
        f.write(f"9. Traditional contrastive:   norm = {torch.norm(traditional_diff.vector).item():.4f}\n\n")

        f.write("=" * 80 + "\n")
        f.write("SCALE TESTING RESULTS (all vectors × all scales)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Baseline: {baseline_result['text'][:100]}...\n\n")

        for vec_name, scales in scale_results.items():
            f.write(f"\n{vec_name} vector:\n")
            f.write("-" * 60 + "\n")
            for scale, text in scales.items():
                is_garbled = any(ord(c) < 32 and c not in '\n\t' for c in text[:50])
                concept = 'dog' in text.lower() or 'bridge' in text.lower()
                status = "[GARBLED]" if is_garbled else "[CONCEPT]" if concept else "[NEUTRAL]"
                f.write(f"Scale {scale:.1f} {status:12s}: {text[:120]}...\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("VALIDATION: Null-diff vs Traditional\n")
        f.write("=" * 80 + "\n")
        f.write(f"Null-diff (Dogs):  {dogs_result['text'][:150]}...\n")
        f.write(f"Traditional diff:  {trad_result['text'][:150]}...\n\n")

        f.write("=" * 80 + "\n")
        f.write("FILES SAVED\n")
        f.write("=" * 80 + "\n")
        f.write(f"Steering vectors: {vectors_dir}\n")
        f.write(f"Summary: {summary_file}\n\n")

    logger.info(f"Saved comprehensive summary to {summary_file}")
    logger.info(f"\nCompleted experiments for {model_key}")

    # Clean up
    del model_handler
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description="Run steering vector experiments across multiple models"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_CONFIGS.keys()) + ["all"],
        default=["qwen-3b"],
        help="Models to test (default: qwen-3b)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/multi_model",
        help="Base output directory (default: outputs/multi_model)"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit"
    )

    args = parser.parse_args()

    if args.list_models:
        list_available_models()
        return

    # Determine which models to run
    if "all" in args.models:
        models_to_run = list(MODEL_CONFIGS.keys())
    else:
        models_to_run = args.models

    logger.info(f"Running experiments for models: {', '.join(models_to_run)}")

    output_base_dir = Path(args.output_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    # Run for each model
    for model_key in models_to_run:
        try:
            run_for_model(model_key, output_base_dir)
        except Exception as e:
            logger.error(f"Error running {model_key}: {e}", exc_info=True)
            continue

    logger.info("\n" + "=" * 80)
    logger.info("ALL EXPERIMENTS COMPLETED")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {output_base_dir}")


if __name__ == "__main__":
    main()
