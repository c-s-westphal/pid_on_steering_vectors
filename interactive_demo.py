"""
Interactive demonstration script for exploring steering vectors.

This script provides a step-by-step walkthrough with detailed output
to help understand how each component works.
"""

import torch
import logging

from models import ModelHandler
from extraction import ActivationExtractor
from vectors import VectorComputer, SteeringVector
from steering import SteeredGenerator
from evaluation import SteeringEvaluator
from data import DatasetBuilder, EvaluationPrompts
from utils import analyze_vector_statistics, cosine_similarity

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_subsection(title):
    """Print a subsection header."""
    print("\n" + "-" * 80)
    print(f"  {title}")
    print("-" * 80)


def main():
    print_section("INTERACTIVE STEERING VECTORS DEMO")
    print("This demo walks through the entire workflow step-by-step.\n")
    print("Press Ctrl+C at any time to exit.\n")

    # ========================================================================
    # SETUP
    # ========================================================================
    print_section("STEP 1: Initialize Model and Components")

    print("Loading Qwen2.5-3B model...")
    print("(This will download ~6GB on first run)\n")

    model_handler = ModelHandler(
        model_name="Qwen/Qwen2.5-3B",
        torch_dtype=torch.float16
    )

    print(f"✓ Model loaded successfully")
    print(f"  - Device: {model_handler.device}")
    print(f"  - Layers: {model_handler.num_layers}")
    print(f"  - Hidden size: {model_handler.hidden_size}")

    # Create components
    extractor = ActivationExtractor(model_handler)
    generator = SteeredGenerator(model_handler)
    evaluator = SteeringEvaluator(model_handler, generator)

    print(f"\n✓ Components initialized")

    # Choose target layer
    layer_range = model_handler.get_layer_range("middle-late")
    target_layer = layer_range[len(layer_range) // 2]

    print(f"\n✓ Target layer: {target_layer}")
    print(f"  (from middle-late range: {layer_range[0]} to {layer_range[-1]})")

    # ========================================================================
    # CREATE DATASETS
    # ========================================================================
    print_section("STEP 2: Create Concept Datasets")

    dogs_dataset = DatasetBuilder.create_dog_dataset()
    bridge_dataset = DatasetBuilder.create_bridge_dataset()

    print(f"Dogs dataset: {len(dogs_dataset)} examples")
    print("  Examples:", dogs_dataset.examples[:5])

    print(f"\nBridge dataset: {len(bridge_dataset)} examples")
    print("  Examples:", bridge_dataset.examples[:3])

    # Show what prompts look like
    print_subsection("Generated Prompts")
    print("Using 'topic' template:")
    print("  Dogs:", dogs_dataset.get_prompts()[:3])
    print("  Bridge:", bridge_dataset.get_prompts()[:2])

    # ========================================================================
    # NULL VECTOR
    # ========================================================================
    print_section("STEP 3: Compute Null Vector")

    print("Computing null vector by averaging all token embeddings...")
    print("(This represents the 'average embedding space')\n")

    null_vector = extractor.compute_null_vector(target_layer)

    print(f"✓ Null vector computed")

    # Analyze null vector
    null_stats = analyze_vector_statistics(null_vector)
    print(f"\nNull vector statistics:")
    print(f"  Shape: {null_vector.shape}")
    print(f"  Norm: {null_stats['norm']:.4f}")
    print(f"  Mean: {null_stats['mean']:.6f}")
    print(f"  Std: {null_stats['std']:.6f}")
    print(f"  Range: [{null_stats['min']:.4f}, {null_stats['max']:.4f}]")

    # ========================================================================
    # EXTRACT ACTIVATIONS
    # ========================================================================
    print_section("STEP 4: Extract Concept Activations")

    print("Extracting activations for 'dogs' concept...")
    dogs_prompts = dogs_dataset.get_prompts()
    dogs_activation = extractor.extract_mean_activation(
        prompts=dogs_prompts,
        layer_idx=target_layer,
        position="last"
    )

    print(f"✓ Dogs activation extracted")
    dogs_stats = analyze_vector_statistics(dogs_activation)
    print(f"  Norm: {dogs_stats['norm']:.4f}")
    print(f"  Mean: {dogs_stats['mean']:.6f}")

    print("\nExtracting activations for 'Golden Gate Bridge' concept...")
    bridge_prompts = bridge_dataset.get_prompts()
    bridge_activation = extractor.extract_mean_activation(
        prompts=bridge_prompts,
        layer_idx=target_layer,
        position="last"
    )

    print(f"✓ Bridge activation extracted")
    bridge_stats = analyze_vector_statistics(bridge_activation)
    print(f"  Norm: {bridge_stats['norm']:.4f}")
    print(f"  Mean: {bridge_stats['mean']:.6f}")

    # Compare activations
    print_subsection("Activation Comparison")
    cos_sim = cosine_similarity(dogs_activation, bridge_activation)
    print(f"Cosine similarity (dogs vs bridge): {cos_sim:.4f}")
    print("(Lower values indicate more distinct concepts)")

    # ========================================================================
    # CREATE STEERING VECTORS
    # ========================================================================
    print_section("STEP 5: Create Steering Vectors (Null-Diff Method)")

    print("Creating dogs steering vector: dogs_activation - null_vector...")
    dogs_vector = VectorComputer.from_diff_with_null(
        concept_activation=dogs_activation,
        null_vector=null_vector,
        layer_idx=target_layer,
        concept="dogs"
    )

    print(f"✓ {dogs_vector}")
    dogs_vec_stats = analyze_vector_statistics(dogs_vector.vector)
    print(f"  Norm: {dogs_vec_stats['norm']:.4f}")

    print("\nCreating bridge steering vector: bridge_activation - null_vector...")
    bridge_vector = VectorComputer.from_diff_with_null(
        concept_activation=bridge_activation,
        null_vector=null_vector,
        layer_idx=target_layer,
        concept="golden_gate_bridge"
    )

    print(f"✓ {bridge_vector}")
    bridge_vec_stats = analyze_vector_statistics(bridge_vector.vector)
    print(f"  Norm: {bridge_vec_stats['norm']:.4f}")

    # ========================================================================
    # COMBINE VECTORS
    # ========================================================================
    print_section("STEP 6: Create Combined Steering Vectors")

    print("Creating combined vectors using different methods...\n")

    # Mean
    print("1. Mean combination: (dogs + bridge) / 2")
    mean_vec = VectorComputer.combine_vectors(
        dogs_vector, bridge_vector, method="mean"
    )
    print(f"   ✓ {mean_vec}")
    print(f"   Norm: {analyze_vector_statistics(mean_vec.vector)['norm']:.4f}")

    # Max
    print("\n2. Max combination: elementwise_max(dogs, bridge)")
    max_vec = VectorComputer.combine_vectors(
        dogs_vector, bridge_vector, method="max"
    )
    print(f"   ✓ {max_vec}")
    print(f"   Norm: {analyze_vector_statistics(max_vec.vector)['norm']:.4f}")

    # RMS-signed
    print("\n3. RMS-signed: sign(a+b) · sqrt((a² + b²) / 2)")
    rms_vec = VectorComputer.combine_vectors(
        dogs_vector, bridge_vector, method="rms_signed"
    )
    print(f"   ✓ {rms_vec}")
    print(f"   Norm: {analyze_vector_statistics(rms_vec.vector)['norm']:.4f}")

    # Diff
    print("\n4. Diff: dogs - bridge")
    diff_vec = VectorComputer.combine_vectors(
        dogs_vector, bridge_vector, method="diff"
    )
    print(f"   ✓ {diff_vec}")
    print(f"   Norm: {analyze_vector_statistics(diff_vec.vector)['norm']:.4f}")

    # ========================================================================
    # TEST GENERATION
    # ========================================================================
    print_section("STEP 7: Test Steered Generation")

    test_prompt = "Let me tell you about"
    print(f"Test prompt: \"{test_prompt}\"")
    print(f"Generating with max_new_tokens=40, scale=2.0\n")

    # Baseline
    print_subsection("Baseline (No Steering)")
    baseline = generator.generate(
        prompt=test_prompt,
        steering_vector=None,
        max_new_tokens=40,
        temperature=0.7
    )
    print(f"\"{baseline['text']}\"")

    # Dogs steering
    print_subsection("Dogs Steering")
    dogs_gen = generator.generate(
        prompt=test_prompt,
        steering_vector=dogs_vector,
        scale=2.0,
        max_new_tokens=40,
        temperature=0.7
    )
    print(f"\"{dogs_gen['text']}\"")

    # Bridge steering
    print_subsection("Bridge Steering")
    bridge_gen = generator.generate(
        prompt=test_prompt,
        steering_vector=bridge_vector,
        scale=2.0,
        max_new_tokens=40,
        temperature=0.7
    )
    print(f"\"{bridge_gen['text']}\"")

    # Mean combination
    print_subsection("Mean Combination (Dogs + Bridge)")
    mean_gen = generator.generate(
        prompt=test_prompt,
        steering_vector=mean_vec,
        scale=2.0,
        max_new_tokens=40,
        temperature=0.7
    )
    print(f"\"{mean_gen['text']}\"")

    # ========================================================================
    # PROBABILITY ANALYSIS
    # ========================================================================
    print_section("STEP 8: Analyze Token Probability Shifts")

    dog_tokens = ["dog", "dogs", "puppy"]
    print(f"Tracking tokens: {dog_tokens}")
    print(f"Testing scales: [1.0, 2.0, 5.0]\n")

    prob_analysis = evaluator.analyze_token_probability_shifts(
        prompt=test_prompt,
        steering_vector=dogs_vector,
        concept_tokens=dog_tokens,
        scales=[1.0, 2.0, 5.0],
        max_new_tokens=10
    )

    print("Results:")
    if 'statistics' in prob_analysis and 'per_token' in prob_analysis['statistics']:
        for token, stats in prob_analysis['statistics']['per_token'].items():
            print(f"\n  Token: '{token}'")
            print(f"    Baseline probability: {stats['baseline_avg_prob']:.6f}")

            if 'scale_effects' in stats:
                for scale, effect in stats['scale_effects'].items():
                    abs_change = effect['absolute_change']
                    rel_change = effect['relative_change'] * 100
                    print(f"    Scale {scale}: {effect['avg_prob']:.6f} "
                          f"({abs_change:+.6f}, {rel_change:+.1f}%)")

    # ========================================================================
    # CONCEPT SCORING
    # ========================================================================
    print_section("STEP 9: Evaluate Concept Presence")

    eval_prompts = EvaluationPrompts.get_neutral_prompts()[:3]
    print(f"Generating {len(eval_prompts)} samples to score concept presence...\n")

    quality_results = evaluator.evaluate_generation_quality(
        prompts=eval_prompts,
        steering_vector=dogs_vector,
        scale=2.0,
        max_new_tokens=50
    )

    scoring = evaluator.automated_concept_scoring(
        generations=quality_results['generations'],
        concept_keywords=dog_tokens
    )

    print("Concept presence scoring:")
    if 'baseline_avg' in scoring:
        print(f"  Baseline:")
        print(f"    Avg keyword occurrences: {scoring['baseline_avg']['avg_total_occurrences']:.2f}")
        print(f"    Presence rate: {scoring['baseline_avg']['avg_presence_rate']:.4f}")

    if 'steered_avg' in scoring:
        print(f"\n  Steered (dogs, scale=2.0):")
        print(f"    Avg keyword occurrences: {scoring['steered_avg']['avg_total_occurrences']:.2f}")
        print(f"    Presence rate: {scoring['steered_avg']['avg_presence_rate']:.4f}")

        if 'baseline_avg' in scoring:
            improvement = (scoring['steered_avg']['avg_total_occurrences'] /
                          scoring['baseline_avg']['avg_total_occurrences'] - 1) * 100
            print(f"\n  Improvement: {improvement:+.1f}%")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print_section("DEMO COMPLETE!")

    print("Summary of what we did:")
    print("  1. ✓ Loaded Qwen2.5-3B model")
    print("  2. ✓ Created concept datasets (dogs, Golden Gate Bridge)")
    print("  3. ✓ Computed null vector from average embeddings")
    print("  4. ✓ Extracted concept activations")
    print("  5. ✓ Created steering vectors using null-diff method")
    print("  6. ✓ Combined vectors with 4 different methods")
    print("  7. ✓ Generated text with and without steering")
    print("  8. ✓ Analyzed token probability shifts")
    print("  9. ✓ Evaluated concept presence in generations")

    print("\nNext steps:")
    print("  - Run example.py for full experiment with saved outputs")
    print("  - Try your own concepts with DatasetBuilder.create_custom_dataset()")
    print("  - Experiment with different combination methods")
    print("  - Analyze probability shifts at different scales")
    print("  - Compare null-diff method to traditional contrastive method")

    print("\nHappy researching!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
