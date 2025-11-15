"""
Simple test to verify steering works with raw activations.
Tests multiple scales to find the sweet spot.
"""

import torch
import logging
from models import ModelHandler
from extraction import ActivationExtractor
from vectors import SteeringVector
from steering import SteeredGenerator

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("="*80)
    logger.info("SIMPLE STEERING TEST - Finding the right scale")
    logger.info("="*80)

    # Load model
    logger.info("\n1. Loading model...")
    model_handler = ModelHandler("Qwen/Qwen2.5-3B")
    extractor = ActivationExtractor(model_handler)
    generator = SteeredGenerator(model_handler)

    # Use a late layer (better for semantic steering)
    layer_idx = 30
    logger.info(f"   Using layer {layer_idx}")

    # Create simple, natural prompts
    logger.info("\n2. Creating concept prompts...")
    dog_prompts = [
        "I love dogs",
        "Dogs are amazing",
        "My dog is wonderful",
        "Puppies are adorable",
        "The dog was happy"
    ]
    logger.info(f"   Dog prompts: {dog_prompts[:2]}...")

    # Extract activation (NO null subtraction, just raw)
    logger.info("\n3. Extracting dog concept activation...")
    dog_activation = extractor.extract_mean_activation(
        prompts=dog_prompts,
        layer_idx=layer_idx,
        position="last"
    )
    activation_norm = torch.norm(dog_activation).item()
    logger.info(f"   Activation norm: {activation_norm:.2f}")

    # Create steering vector (just the raw activation!)
    dog_vector = SteeringVector(
        vector=dog_activation,
        layer_idx=layer_idx,
        concept="dogs_raw",
        method="raw_activation"
    )
    logger.info(f"   Steering vector norm: {torch.norm(dog_vector.vector).item():.2f}")

    # Test prompt
    test_prompt = "Write a short paragraph about"

    # Test multiple scales
    scales_to_test = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]

    logger.info("\n" + "="*80)
    logger.info("TESTING DIFFERENT SCALES")
    logger.info("="*80)
    logger.info(f"Prompt: '{test_prompt}'\n")

    # Baseline
    logger.info("BASELINE (no steering):")
    logger.info("-"*80)
    baseline = generator.generate(
        prompt=test_prompt,
        steering_vector=None,
        max_new_tokens=40,
        temperature=0.7
    )
    logger.info(f"{baseline['text']}\n")

    # Test each scale
    for scale in scales_to_test:
        logger.info(f"SCALE {scale}:")
        logger.info("-"*80)
        result = generator.generate(
            prompt=test_prompt,
            steering_vector=dog_vector,
            scale=scale,
            max_new_tokens=40,
            temperature=0.7
        )

        # Check if output is valid (no control characters)
        text = result['text']
        has_control_chars = any(ord(c) < 32 and c not in '\n\t' for c in text)

        if has_control_chars:
            logger.info(f"⚠️  GARBLED OUTPUT (scale too high)")
            logger.info(f"{text[:100]}...\n")
        else:
            # Count dog-related words
            dog_words = ['dog', 'dogs', 'puppy', 'puppies', 'canine', 'pet']
            dog_count = sum(text.lower().count(word) for word in dog_words)

            logger.info(f"✓ Valid output (dog mentions: {dog_count})")
            logger.info(f"{text}\n")

    logger.info("="*80)
    logger.info("RECOMMENDATIONS:")
    logger.info("="*80)
    logger.info("- If you see garbled output, the scale is too high")
    logger.info("- Look for the highest scale that gives valid, dog-related text")
    logger.info("- Typical good range: 0.3 - 1.0 for raw activations")
    logger.info("="*80)

if __name__ == "__main__":
    main()
