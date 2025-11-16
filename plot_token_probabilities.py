#!/usr/bin/env python3
"""
Plot token probability changes from multi-model experiments.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

def load_token_probabilities(json_path):
    """Load token probability data from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def plot_probability_heatmap(data, output_dir, model_name="Model"):
    """
    Create a heatmap showing probability changes for each token across vectors and scales.
    """
    baseline = data['baseline']
    concept_tokens = data['concept_tokens']
    test_scales = data['test_scales']
    results = data['results']

    # Get all vector names
    vector_names = list(results.keys())

    # Create a subplot for each token
    n_tokens = len(concept_tokens)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for token_idx, token in enumerate(concept_tokens):
        ax = axes[token_idx]

        # Build matrix: rows = vectors, cols = scales
        matrix = np.zeros((len(vector_names), len(test_scales)))

        for vec_idx, vec_name in enumerate(vector_names):
            for scale_idx, scale in enumerate(test_scales):
                scale_str = str(scale)
                if scale_str in results[vec_name]:
                    delta = results[vec_name][scale_str]['delta'].get(token, 0.0)
                    matrix[vec_idx, scale_idx] = delta

        # Plot heatmap
        im = ax.imshow(matrix, aspect='auto', cmap='RdBu_r',
                      vmin=-np.abs(matrix).max(), vmax=np.abs(matrix).max())

        ax.set_xticks(range(len(test_scales)))
        ax.set_xticklabels([f'{s:.1f}' for s in test_scales], rotation=45)
        ax.set_yticks(range(len(vector_names)))
        ax.set_yticklabels(vector_names, fontsize=8)

        ax.set_xlabel('Steering Scale')
        ax.set_ylabel('Vector Type')
        ax.set_title(f'Token: "{token}"\nBaseline prob: {baseline.get(token, 0):.2e}')

        plt.colorbar(im, ax=ax, label='Probability Change')

    plt.suptitle(f'{model_name}: Token Probability Changes by Vector and Scale',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / f'{model_name.lower().replace(" ", "_")}_token_prob_heatmap.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved heatmap to {output_path}")
    plt.close()

def plot_vector_comparison(data, output_dir, model_name="Model"):
    """
    Compare how different vectors affect dog vs bridge token probabilities.
    """
    baseline = data['baseline']
    test_scales = data['test_scales']
    results = data['results']

    # Focus on dog and bridge tokens
    dog_tokens = ['dog', 'dogs', 'puppy']
    bridge_tokens = ['bridge', 'bridges', 'Golden']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot for each vector
    for vec_name, vec_data in results.items():
        dog_probs = []
        bridge_probs = []

        for scale in test_scales:
            scale_str = str(scale)
            if scale_str in vec_data:
                # Average probability change across dog tokens
                dog_delta = np.mean([vec_data[scale_str]['delta'].get(t, 0.0) for t in dog_tokens])
                bridge_delta = np.mean([vec_data[scale_str]['delta'].get(t, 0.0) for t in bridge_tokens])

                dog_probs.append(dog_delta)
                bridge_probs.append(bridge_delta)
            else:
                dog_probs.append(0)
                bridge_probs.append(0)

        # Plot dog tokens
        axes[0].plot(test_scales, dog_probs, marker='o', label=vec_name, alpha=0.7)

        # Plot bridge tokens
        axes[1].plot(test_scales, bridge_probs, marker='s', label=vec_name, alpha=0.7)

    # Configure dog plot
    axes[0].set_xlabel('Steering Scale', fontsize=12)
    axes[0].set_ylabel('Average Probability Change', fontsize=12)
    axes[0].set_title('Dog-related Tokens (dog, dogs, puppy)', fontsize=13, fontweight='bold')
    axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    # Configure bridge plot
    axes[1].set_xlabel('Steering Scale', fontsize=12)
    axes[1].set_ylabel('Average Probability Change', fontsize=12)
    axes[1].set_title('Bridge-related Tokens (bridge, bridges, Golden)', fontsize=13, fontweight='bold')
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    plt.suptitle(f'{model_name}: Concept Token Probability Changes', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / f'{model_name.lower().replace(" ", "_")}_concept_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison to {output_path}")
    plt.close()

def plot_probe_vs_traditional(data, output_dir, model_name="Model"):
    """
    Compare probe-based vectors vs traditional methods.
    """
    test_scales = data['test_scales']
    results = data['results']

    # Categorize vectors
    probe_vectors = {k: v for k, v in results.items() if k.startswith('Probe')}
    traditional_vectors = {k: v for k, v in results.items() if not k.startswith('Probe')}

    dog_tokens = ['dog', 'dogs', 'puppy']
    bridge_tokens = ['bridge', 'bridges', 'Golden']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot traditional methods - dog tokens
    for vec_name, vec_data in traditional_vectors.items():
        dog_probs = []
        for scale in test_scales:
            scale_str = str(scale)
            if scale_str in vec_data:
                dog_delta = np.mean([vec_data[scale_str]['delta'].get(t, 0.0) for t in dog_tokens])
                dog_probs.append(dog_delta)
            else:
                dog_probs.append(0)
        axes[0, 0].plot(test_scales, dog_probs, marker='o', label=vec_name, alpha=0.7)

    # Plot traditional methods - bridge tokens
    for vec_name, vec_data in traditional_vectors.items():
        bridge_probs = []
        for scale in test_scales:
            scale_str = str(scale)
            if scale_str in vec_data:
                bridge_delta = np.mean([vec_data[scale_str]['delta'].get(t, 0.0) for t in bridge_tokens])
                bridge_probs.append(bridge_delta)
            else:
                bridge_probs.append(0)
        axes[0, 1].plot(test_scales, bridge_probs, marker='s', label=vec_name, alpha=0.7)

    # Plot probe methods - dog tokens
    for vec_name, vec_data in probe_vectors.items():
        dog_probs = []
        for scale in test_scales:
            scale_str = str(scale)
            if scale_str in vec_data:
                dog_delta = np.mean([vec_data[scale_str]['delta'].get(t, 0.0) for t in dog_tokens])
                dog_probs.append(dog_delta)
            else:
                dog_probs.append(0)
        axes[1, 0].plot(test_scales, dog_probs, marker='o', label=vec_name, alpha=0.7, linewidth=2)

    # Plot probe methods - bridge tokens
    for vec_name, vec_data in probe_vectors.items():
        bridge_probs = []
        for scale in test_scales:
            scale_str = str(scale)
            if scale_str in vec_data:
                bridge_delta = np.mean([vec_data[scale_str]['delta'].get(t, 0.0) for t in bridge_tokens])
                bridge_probs.append(bridge_delta)
            else:
                bridge_probs.append(0)
        axes[1, 1].plot(test_scales, bridge_probs, marker='s', label=vec_name, alpha=0.7, linewidth=2)

    # Configure plots
    titles = [
        'Traditional Methods: Dog Tokens',
        'Traditional Methods: Bridge Tokens',
        'MLP Probe Methods: Dog Tokens',
        'MLP Probe Methods: Bridge Tokens'
    ]

    for ax, title in zip(axes.flatten(), titles):
        ax.set_xlabel('Steering Scale', fontsize=11)
        ax.set_ylabel('Avg Probability Change', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.suptitle(f'{model_name}: MLP Probe vs Traditional Steering Methods',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / f'{model_name.lower().replace(" ", "_")}_probe_vs_traditional.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved probe comparison to {output_path}")
    plt.close()

def plot_best_performers(data, output_dir, model_name="Model"):
    """
    Identify and plot the best performing vectors for each concept.
    """
    test_scales = data['test_scales']
    results = data['results']

    dog_tokens = ['dog', 'dogs', 'puppy']
    bridge_tokens = ['bridge', 'bridges', 'Golden']

    # Calculate average effect at scale 1.0 for each vector
    vector_performance = {}

    for vec_name, vec_data in results.items():
        if '1.0' in vec_data:
            dog_effect = np.mean([vec_data['1.0']['delta'].get(t, 0.0) for t in dog_tokens])
            bridge_effect = np.mean([vec_data['1.0']['delta'].get(t, 0.0) for t in bridge_tokens])
            vector_performance[vec_name] = {
                'dog': dog_effect,
                'bridge': bridge_effect,
                'combined': dog_effect + bridge_effect
            }

    # Find top 5 for each concept
    top_dog = sorted(vector_performance.items(), key=lambda x: x[1]['dog'], reverse=True)[:5]
    top_bridge = sorted(vector_performance.items(), key=lambda x: x[1]['bridge'], reverse=True)[:5]
    top_combined = sorted(vector_performance.items(), key=lambda x: x[1]['combined'], reverse=True)[:5]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot top dog vectors
    for vec_name, _ in top_dog:
        dog_probs = []
        for scale in test_scales:
            scale_str = str(scale)
            if scale_str in results[vec_name]:
                dog_delta = np.mean([results[vec_name][scale_str]['delta'].get(t, 0.0) for t in dog_tokens])
                dog_probs.append(dog_delta)
            else:
                dog_probs.append(0)
        axes[0].plot(test_scales, dog_probs, marker='o', label=vec_name, linewidth=2)

    # Plot top bridge vectors
    for vec_name, _ in top_bridge:
        bridge_probs = []
        for scale in test_scales:
            scale_str = str(scale)
            if scale_str in results[vec_name]:
                bridge_delta = np.mean([results[vec_name][scale_str]['delta'].get(t, 0.0) for t in bridge_tokens])
                bridge_probs.append(bridge_delta)
            else:
                bridge_probs.append(0)
        axes[1].plot(test_scales, bridge_probs, marker='s', label=vec_name, linewidth=2)

    # Plot top combined vectors
    for vec_name, _ in top_combined:
        combined_probs = []
        for scale in test_scales:
            scale_str = str(scale)
            if scale_str in results[vec_name]:
                dog_delta = np.mean([results[vec_name][scale_str]['delta'].get(t, 0.0) for t in dog_tokens])
                bridge_delta = np.mean([results[vec_name][scale_str]['delta'].get(t, 0.0) for t in bridge_tokens])
                combined_probs.append(dog_delta + bridge_delta)
            else:
                combined_probs.append(0)
        axes[2].plot(test_scales, combined_probs, marker='D', label=vec_name, linewidth=2)

    # Configure plots
    axes[0].set_title('Top 5 for Dog Promotion', fontsize=12, fontweight='bold')
    axes[1].set_title('Top 5 for Bridge Promotion', fontsize=12, fontweight='bold')
    axes[2].set_title('Top 5 for Combined Promotion', fontsize=12, fontweight='bold')

    for ax in axes:
        ax.set_xlabel('Steering Scale', fontsize=11)
        ax.set_ylabel('Avg Probability Change', fontsize=11)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    plt.suptitle(f'{model_name}: Best Performing Steering Vectors (at scale=1.0)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / f'{model_name.lower().replace(" ", "_")}_best_performers.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved best performers to {output_path}")
    plt.close()

    # Print summary
    print(f"\n{model_name} - Best Performers at Scale 1.0:")
    print("\nTop 5 for Dog Promotion:")
    for vec_name, perf in top_dog:
        print(f"  {vec_name:30s}: {perf['dog']:+.6f}")
    print("\nTop 5 for Bridge Promotion:")
    for vec_name, perf in top_bridge:
        print(f"  {vec_name:30s}: {perf['bridge']:+.6f}")
    print("\nTop 5 for Combined Promotion:")
    for vec_name, perf in top_combined:
        print(f"  {vec_name:30s}: {perf['combined']:+.6f}")

def main():
    parser = argparse.ArgumentParser(description='Plot token probability results')
    parser.add_argument(
        '--results-dir',
        type=str,
        default='hpc_results/multi_model',
        help='Directory containing multi_model results'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='plots/token_probabilities',
        help='Output directory for plots'
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all token_probabilities.json files
    prob_files = list(results_dir.rglob('token_probabilities.json'))

    if not prob_files:
        print(f"No token_probabilities.json files found in {results_dir}")
        return

    print(f"Found {len(prob_files)} token probability files")

    for prob_file in prob_files:
        # Extract model name from path
        parts = prob_file.parts
        model_name = parts[-3]  # e.g., 'qwen-7b'
        timestamp = parts[-2]   # e.g., '20251116_200826'

        print(f"\n{'='*60}")
        print(f"Processing: {model_name} ({timestamp})")
        print(f"{'='*60}")

        # Load data
        data = load_token_probabilities(prob_file)

        # Create plots
        model_output_dir = output_dir / model_name / timestamp
        model_output_dir.mkdir(parents=True, exist_ok=True)

        plot_probability_heatmap(data, model_output_dir, model_name.upper())
        plot_vector_comparison(data, model_output_dir, model_name.upper())
        plot_probe_vs_traditional(data, model_output_dir, model_name.upper())
        plot_best_performers(data, model_output_dir, model_name.upper())

    print(f"\n{'='*60}")
    print(f"All plots saved to {output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
