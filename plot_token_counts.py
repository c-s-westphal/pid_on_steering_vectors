#!/usr/bin/env python3
"""
Plot quantitative analysis of how different vectors promote concept tokens.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re
from collections import Counter

def parse_summary_for_token_counts(summary_path):
    """Parse summary.txt and count concept mentions by vector and scale."""
    with open(summary_path, 'r') as f:
        content = f.read()

    # Extract scale testing results with actual text
    scale_results = {}
    current_vector = None

    lines = content.split('\n')
    in_scale_section = False

    for line in lines:
        if 'SCALE TESTING RESULTS' in line:
            in_scale_section = True
            continue

        if 'VALIDATION:' in line:
            break

        if in_scale_section:
            # Check if this is a vector name line
            if 'vector:' in line and '----' not in line:
                current_vector = line.split('vector:')[0].strip()
                scale_results[current_vector] = {}

            # Check if this is a scale result line
            elif current_vector and line.strip().startswith('Scale '):
                match = re.search(r'Scale ([\d.]+)\s+\[.*?\]\s*:\s*(.+)', line)
                if match:
                    scale = float(match.group(1))
                    text = match.group(2)
                    scale_results[current_vector][scale] = text

    # Count concept mentions
    concept_counts = {}
    for vector, scales in scale_results.items():
        concept_counts[vector] = {}
        for scale, text in scales.items():
            # Count dog-related words
            dog_count = (
                text.lower().count('dog') +
                text.lower().count('dogs') +
                text.lower().count('puppy') +
                text.lower().count('puppies') +
                text.lower().count('canine')
            )

            # Count bridge-related words
            bridge_count = (
                text.lower().count('bridge') +
                text.lower().count('bridges') +
                text.lower().count('golden gate')
            )

            # Total concept mentions
            total_concept = dog_count + bridge_count

            concept_counts[vector][scale] = {
                'dog': dog_count,
                'bridge': bridge_count,
                'total': total_concept,
                'text_length': len(text.split())  # For normalization
            }

    return concept_counts


def plot_concept_mentions_by_scale(concept_counts, output_path):
    """Plot total concept mentions by scale for each vector."""
    fig, ax = plt.subplots(figsize=(12, 7))

    scales = sorted(list(set(scale for v in concept_counts.values() for scale in v.keys())))
    vectors = list(concept_counts.keys())

    # Colors for different vector types
    color_map = {
        'Dogs': '#3498db',
        'Bridge': '#e74c3c',
        'Mean': '#f39c12',
        'Max': '#9b59b6',
        'Min': '#1abc9c',
        'RMS': '#34495e',
        'Diff': '#16a085',
        'AbsDiff': '#d35400',
        'Traditional': '#27ae60'
    }

    for vector in vectors:
        counts = [concept_counts[vector].get(scale, {}).get('total', 0) for scale in scales]
        color = color_map.get(vector, 'gray')
        ax.plot(scales, counts, marker='o', label=vector, linewidth=2, color=color, markersize=6)

    ax.set_xlabel('Steering Scale', fontsize=13, fontweight='bold')
    ax.set_ylabel('Total Concept Mentions (Count)', fontsize=13, fontweight='bold')
    ax.set_title('Concept Token Mentions by Vector and Scale\n(Dogs + Bridge mentions in generated text)',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', fontsize=10, ncol=2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved concept mentions plot to {output_path}")
    plt.close()


def plot_dog_vs_bridge_mentions(concept_counts, output_path):
    """Plot dog vs bridge mentions separately."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    scales = sorted(list(set(scale for v in concept_counts.values() for scale in v.keys())))
    vectors = list(concept_counts.keys())

    color_map = {
        'Dogs': '#3498db',
        'Bridge': '#e74c3c',
        'Mean': '#f39c12',
        'Max': '#9b59b6',
        'Min': '#1abc9c',
        'RMS': '#34495e',
        'Diff': '#16a085',
        'AbsDiff': '#d35400',
        'Traditional': '#27ae60'
    }

    # Plot dog mentions
    for vector in vectors:
        counts = [concept_counts[vector].get(scale, {}).get('dog', 0) for scale in scales]
        color = color_map.get(vector, 'gray')
        ax1.plot(scales, counts, marker='o', label=vector, linewidth=2, color=color, markersize=6)

    ax1.set_xlabel('Steering Scale', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Dog-related Mentions', fontsize=12, fontweight='bold')
    ax1.set_title('Dog Token Mentions by Vector', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper left', fontsize=9)

    # Plot bridge mentions
    for vector in vectors:
        counts = [concept_counts[vector].get(scale, {}).get('bridge', 0) for scale in scales]
        color = color_map.get(vector, 'gray')
        ax2.plot(scales, counts, marker='o', label=vector, linewidth=2, color=color, markersize=6)

    ax2.set_xlabel('Steering Scale', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Bridge-related Mentions', fontsize=12, fontweight='bold')
    ax2.set_title('Bridge Token Mentions by Vector', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='upper left', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved dog vs bridge plot to {output_path}")
    plt.close()


def plot_concept_heatmap_counts(concept_counts, output_path):
    """Heatmap showing total concept counts (not just binary)."""
    vectors = list(concept_counts.keys())
    scales = sorted(list(set(scale for v in concept_counts.values() for scale in v.keys())))

    # Create matrix of counts
    matrix = np.zeros((len(vectors), len(scales)))

    for i, vector in enumerate(vectors):
        for j, scale in enumerate(scales):
            if scale in concept_counts[vector]:
                matrix[i, j] = concept_counts[vector][scale]['total']

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')

    # Set ticks
    ax.set_xticks(range(len(scales)))
    ax.set_xticklabels([f'{s:.1f}' for s in scales])
    ax.set_yticks(range(len(vectors)))
    ax.set_yticklabels(vectors)

    ax.set_xlabel('Steering Scale', fontsize=12, fontweight='bold')
    ax.set_ylabel('Steering Vector', fontsize=12, fontweight='bold')
    ax.set_title('Total Concept Mentions (Quantitative)\n(Higher = More Dog/Bridge Tokens)',
                 fontsize=13, fontweight='bold')

    # Add text annotations
    for i in range(len(vectors)):
        for j in range(len(scales)):
            count = int(matrix[i, j])
            if count > 0:
                text_color = 'white' if count > matrix.max() * 0.5 else 'black'
                ax.text(j, i, str(count), ha='center', va='center',
                       color=text_color, fontsize=9, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Concept Token Count', fontsize=11, fontweight='bold')

    # Add grid
    ax.set_xticks([x - 0.5 for x in range(1, len(scales))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(vectors))], minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved quantitative heatmap to {output_path}")
    plt.close()


def plot_vector_comparison_bar(concept_counts, output_path):
    """Bar chart comparing total concept promotion across all scales."""
    vectors = list(concept_counts.keys())

    # Sum up total mentions across all scales
    total_mentions = []
    for vector in vectors:
        total = sum(data['total'] for data in concept_counts[vector].values())
        total_mentions.append(total)

    # Sort by total
    sorted_indices = np.argsort(total_mentions)[::-1]
    vectors_sorted = [vectors[i] for i in sorted_indices]
    mentions_sorted = [total_mentions[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#e74c3c' if 'Bridge' in v else '#3498db' if 'Dogs' in v else '#95a5a6'
              for v in vectors_sorted]

    bars = ax.barh(vectors_sorted, mentions_sorted, color=colors, alpha=0.8, edgecolor='black')

    ax.set_xlabel('Total Concept Mentions (All Scales Combined)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Steering Vector', fontsize=12, fontweight='bold')
    ax.set_title('Total Concept Promotion by Vector\n(Sum across all scales 0.1-2.0)',
                 fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # Add value labels
    for bar, count in zip(bars, mentions_sorted):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f' {count}',
                ha='left', va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved vector comparison bar chart to {output_path}")
    plt.close()


def main():
    # Find the most recent summary file
    hpc_results_dir = Path("/Users/charleswestphal/Documents/UCL/pid_on_steering_vectors/hpc_results")
    summary_files = list(hpc_results_dir.glob("**/summary.txt"))

    if not summary_files:
        print("No summary.txt files found!")
        return

    # Use the largest one (has most data)
    summary_file = max(summary_files, key=lambda p: p.stat().st_size)
    print(f"Using summary file: {summary_file}")

    # Parse data
    concept_counts = parse_summary_for_token_counts(summary_file)

    print(f"\nFound {len(concept_counts)} vectors")

    # Print summary
    print("\nConcept mention summary:")
    for vector, scales in concept_counts.items():
        total = sum(data['total'] for data in scales.values())
        print(f"  {vector:20s}: {total} total mentions")

    # Create plots directory
    plots_dir = Path("/Users/charleswestphal/Documents/UCL/pid_on_steering_vectors/plots")
    plots_dir.mkdir(exist_ok=True)

    # Generate plots
    plot_concept_mentions_by_scale(concept_counts, plots_dir / "concept_mentions_by_scale.png")
    plot_dog_vs_bridge_mentions(concept_counts, plots_dir / "dog_vs_bridge_mentions.png")
    plot_concept_heatmap_counts(concept_counts, plots_dir / "concept_counts_heatmap.png")
    plot_vector_comparison_bar(concept_counts, plots_dir / "vector_comparison_total.png")

    print(f"\n✓ All token count plots saved to {plots_dir}/")
    print("\nGenerated plots:")
    print("  1. concept_mentions_by_scale.png - Line plot of total mentions vs scale")
    print("  2. dog_vs_bridge_mentions.png - Separate plots for dog vs bridge tokens")
    print("  3. concept_counts_heatmap.png - Heatmap with actual counts (not binary)")
    print("  4. vector_comparison_total.png - Bar chart comparing total promotion")


if __name__ == "__main__":
    main()
