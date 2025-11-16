#!/usr/bin/env python3
"""
Example: Train linear probe and extract steering vectors.
"""

import torch
import logging
from pathlib import Path

from models import ModelHandler
from extraction import ActivationExtractor
from vectors import VectorComputer, SteeringVector
from steering import SteeredGenerator
from probe_dataset import ProbeDatasetGenerator
from probe_trainer_mlp import MLPProbeTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("=" * 80)
    logger.info("LINEAR PROBE STEERING VECTOR EXTRACTION")
    logger.info("=" * 80)

    # ============================================================================
    # 1. Load model
    # ============================================================================
    logger.info("\nStep 1: Loading model...")
    model_handler = ModelHandler(model_name="Qwen/Qwen2.5-3B")
    extractor = ActivationExtractor(model_handler)
    generator = SteeredGenerator(model_handler)
    vector_computer = VectorComputer()

    target_layer = int(model_handler.num_layers * 2 / 3)
    logger.info(f"Using layer {target_layer} (out of {model_handler.num_layers})")

    # ============================================================================
    # 2. Generate training dataset
    # ============================================================================
    logger.info("\nStep 2: Generating training dataset...")
    dataset_gen = ProbeDatasetGenerator(seed=42)
    sentences, labels = dataset_gen.generate_dataset(num_samples_per_class=100)

    logger.info(f"Generated {len(sentences)} sentences")
    logger.info(f"Class distribution:")
    for class_idx in range(4):
        count = sum(1 for l in labels if l == class_idx)
        class_name = dataset_gen.get_class_name(class_idx)
        logger.info(f"  {class_name}: {count} samples")

    # Show examples
    logger.info("\nExample sentences:")
    for i in range(8):
        logger.info(f"  [{dataset_gen.get_class_name(labels[i]):12s}] {sentences[i]}")

    # ============================================================================
    # 3. Extract activations for all sentences
    # ============================================================================
    logger.info(f"\nStep 3: Extracting activations at layer {target_layer}...")

    all_activations = []
    for i, sentence in enumerate(sentences):
        if (i + 1) % 50 == 0:
            logger.info(f"  Processed {i+1}/{len(sentences)} sentences...")

        # Extract activation for this sentence
        activation = extractor.extract_mean_activation(
            prompts=[sentence],
            layer_idx=target_layer,
            position="last"
        )
        all_activations.append(activation)

    # Stack into tensor
    activations_tensor = torch.stack(all_activations)  # (num_samples, hidden_size)
    labels_tensor = torch.tensor(labels)

    logger.info(f"Activations shape: {activations_tensor.shape}")
    logger.info(f"Labels shape: {labels_tensor.shape}")

    # ============================================================================
    # 4. Train linear probe
    # ============================================================================
    logger.info("\nStep 4: Training linear probe...")

    # Split train/val (80/20)
    num_train = int(0.8 * len(activations_tensor))
    indices = torch.randperm(len(activations_tensor))
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]

    train_activations = activations_tensor[train_indices]
    train_labels = labels_tensor[train_indices]
    val_activations = activations_tensor[val_indices]
    val_labels = labels_tensor[val_indices]

    trainer = MLPProbeTrainer(
        input_dim=model_handler.hidden_size,
        hidden_dim=model_handler.hidden_size,  # Same as model width
        num_classes=4,
        learning_rate=1e-4,
        use_float32=True  # Force float32 for numerical stability
    )

    history = trainer.train(
        train_activations=train_activations,
        train_labels=train_labels,
        val_activations=val_activations,
        val_labels=val_labels,
        num_epochs=100,
        batch_size=32
    )

    # Save probe
    output_dir = Path("outputs/probe")
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save(output_dir / "linear_probe.pt")

    # ============================================================================
    # 5. Extract steering vectors from probe
    # ============================================================================
    logger.info("\nStep 5: Extracting steering vectors from probe weights...")

    probe_weights = trainer.get_steering_vectors()  # (4, hidden_size)
    logger.info(f"Probe weights shape: {probe_weights.shape}")

    # Create steering vectors for each class
    probe_neither = vector_computer.from_linear_probe(
        probe_weights, class_idx=0, layer_idx=target_layer, concept="probe_neither"
    )
    probe_dog = vector_computer.from_linear_probe(
        probe_weights, class_idx=1, layer_idx=target_layer, concept="probe_dog"
    )
    probe_bridge = vector_computer.from_linear_probe(
        probe_weights, class_idx=2, layer_idx=target_layer, concept="probe_bridge"
    )
    probe_both = vector_computer.from_linear_probe(
        probe_weights, class_idx=3, layer_idx=target_layer, concept="probe_both"
    )

    # Create difference vectors
    probe_dog_vs_neither = vector_computer.from_probe_difference(
        probe_weights, class_a_idx=1, class_b_idx=0,
        layer_idx=target_layer, concept="probe_dog_vs_neither"
    )
    probe_bridge_vs_neither = vector_computer.from_probe_difference(
        probe_weights, class_a_idx=2, class_b_idx=0,
        layer_idx=target_layer, concept="probe_bridge_vs_neither"
    )
    probe_both_vs_neither = vector_computer.from_probe_difference(
        probe_weights, class_a_idx=3, class_b_idx=0,
        layer_idx=target_layer, concept="probe_both_vs_neither"
    )

    logger.info("\nProbe-based steering vectors:")
    logger.info(f"  Neither:             norm = {torch.norm(probe_neither.vector).item():.4f}")
    logger.info(f"  Dog:                 norm = {torch.norm(probe_dog.vector).item():.4f}")
    logger.info(f"  Bridge:              norm = {torch.norm(probe_bridge.vector).item():.4f}")
    logger.info(f"  Both:                norm = {torch.norm(probe_both.vector).item():.4f}")
    logger.info(f"  Dog vs neither:      norm = {torch.norm(probe_dog_vs_neither.vector).item():.4f}")
    logger.info(f"  Bridge vs neither:   norm = {torch.norm(probe_bridge_vs_neither.vector).item():.4f}")
    logger.info(f"  Both vs neither:     norm = {torch.norm(probe_both_vs_neither.vector).item():.4f}")

    # Save all probe vectors
    probe_neither.save(output_dir / "probe_neither.pt")
    probe_dog.save(output_dir / "probe_dog.pt")
    probe_bridge.save(output_dir / "probe_bridge.pt")
    probe_both.save(output_dir / "probe_both.pt")
    probe_dog_vs_neither.save(output_dir / "probe_dog_vs_neither.pt")
    probe_bridge_vs_neither.save(output_dir / "probe_bridge_vs_neither.pt")
    probe_both_vs_neither.save(output_dir / "probe_both_vs_neither.pt")

    # ============================================================================
    # 6. Test steering with probe vectors
    # ============================================================================
    logger.info("\nStep 6: Testing probe-based steering...")

    test_prompt = "Write a short paragraph about"
    test_scale = 1.0

    # Baseline
    logger.info(f"\n{'='*60}")
    logger.info("BASELINE (no steering):")
    baseline = generator.generate(test_prompt, steering_vector=None, max_new_tokens=50)
    logger.info(f"  {baseline['text'][:150]}...")

    # Test each probe vector
    test_vectors = [
        ("Dog direction", probe_dog),
        ("Bridge direction", probe_bridge),
        ("Both direction", probe_both),
        ("Dog vs neither", probe_dog_vs_neither),
        ("Bridge vs neither", probe_bridge_vs_neither),
        ("Both vs neither", probe_both_vs_neither),
    ]

    for name, vec in test_vectors:
        logger.info(f"\n{'='*60}")
        logger.info(f"{name.upper()} (scale={test_scale}):")
        result = generator.generate(test_prompt, steering_vector=vec, scale=test_scale, max_new_tokens=50)
        logger.info(f"  {result['text'][:150]}...")

    # ============================================================================
    # 7. Summary
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Model: {model_handler.model_name}")
    logger.info(f"Training samples: {len(train_activations)}")
    logger.info(f"Validation samples: {len(val_activations)}")
    logger.info(f"Final train accuracy: {history['train_acc'][-1]:.2f}%")
    logger.info(f"Final val accuracy: {history['val_acc'][-1]:.2f}%")
    logger.info(f"\nSaved {7} probe-based steering vectors to {output_dir}/")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
