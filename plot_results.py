#!/usr/bin/env python3
"""
Plot steering vector experiment results.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re

def parse_summary_file(summary_path):
    """Parse summary.txt file to extract vector norms and scale results."""
    with open(summary_path, 'r') as f:
        content = f.read()

    # Extract vector norms
    vector_norms = {}
    norms_section = re.search(r'STEERING VECTORS CREATED.*?\n(.*?)\n\n', content, re.DOTALL)
    if norms_section:
        for line in norms_section.group(1).split('\n'):
            match = re.search(r'\d+\.\s+(.*?):\s+norm = ([\d.]+)', line)
            if match:
                name = match.group(1).strip()
                norm = float(match.group(2))
                vector_norms[name] = norm

    # Extract scale testing results
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
                match = re.search(r'Scale ([\d.]+)\s+\[(.*?)\]', line)
                if match:
                    scale = float(match.group(1))
                    status = match.group(2)
                    scale_results[current_vector][scale] = status

    return vector_norms, scale_results


def plot_vector_norms(vector_norms, output_path):
    """Plot vector norms as a bar chart."""
    fig, ax = plt.subplots(figsize=(12, 6))

    names = list(vector_norms.keys())
    norms = list(vector_norms.values())

    # Color by type
    colors = []
    for name in names:
        if 'null-diff' in name.lower():
            colors.append('steelblue')
        elif 'combination' in name.lower():
            colors.append('coral')
        elif 'traditional' in name.lower():
            colors.append('green')
        else:
            colors.append('gray')

    bars = ax.bar(range(len(names)), norms, color=colors, alpha=0.7, edgecolor='black')

    ax.set_xlabel('Steering Vector', fontsize=12, fontweight='bold')
    ax.set_ylabel('L2 Norm', fontsize=12, fontweight='bold')
    ax.set_title('Steering Vector Norms (Qwen 2.5 7B)', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for bar, norm in zip(bars, norms):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{norm:.1f}',
                ha='center', va='bottom', fontsize=9)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', alpha=0.7, label='Null-diff'),
        Patch(facecolor='coral', alpha=0.7, label='Combination'),
        Patch(facecolor='green', alpha=0.7, label='Traditional')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved vector norms plot to {output_path}")
    plt.close()


def plot_scale_heatmap(scale_results, output_path):
    """Plot heatmap showing which scales produce good/garbled output."""
    # Prepare data
    vectors = list(scale_results.keys())
    scales = sorted(list(set(scale for v in scale_results.values() for scale in v.keys())))

    # Create numeric matrix: 0=neutral, 1=concept, -1=garbled
    matrix = np.zeros((len(vectors), len(scales)))

    for i, vector in enumerate(vectors):
        for j, scale in enumerate(scales):
            if scale in scale_results[vector]:
                status = scale_results[vector][scale]
                if status == 'GARBLED':
                    matrix[i, j] = -1
                elif status == '✓ concept' or status == 'CONCEPT':
                    matrix[i, j] = 1
                else:  # neutral
                    matrix[i, j] = 0

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)

    # Set ticks
    ax.set_xticks(range(len(scales)))
    ax.set_xticklabels([f'{s:.1f}' for s in scales])
    ax.set_yticks(range(len(vectors)))
    ax.set_yticklabels(vectors)

    ax.set_xlabel('Steering Scale', fontsize=12, fontweight='bold')
    ax.set_ylabel('Steering Vector', fontsize=12, fontweight='bold')
    ax.set_title('Steering Quality by Vector and Scale\n(Green=Concept, Yellow=Neutral, Red=Garbled)',
                 fontsize=13, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(['Garbled', 'Neutral', 'Concept'])

    # Add grid
    ax.set_xticks([x - 0.5 for x in range(1, len(scales))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(vectors))], minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved scale heatmap to {output_path}")
    plt.close()


def plot_concept_success_rate(scale_results, output_path):
    """Plot success rate (concept mention) by scale across all vectors."""
    scales = sorted(list(set(scale for v in scale_results.values() for scale in v.keys())))

    success_rates = []
    garbled_rates = []
    neutral_rates = []

    for scale in scales:
        total = 0
        success = 0
        garbled = 0
        neutral = 0

        for vector_results in scale_results.values():
            if scale in vector_results:
                total += 1
                status = vector_results[scale]
                if status == '✓ concept' or status == 'CONCEPT':
                    success += 1
                elif status == 'GARBLED':
                    garbled += 1
                else:
                    neutral += 1

        success_rates.append(success / total * 100 if total > 0 else 0)
        garbled_rates.append(garbled / total * 100 if total > 0 else 0)
        neutral_rates.append(neutral / total * 100 if total > 0 else 0)

    # Plot stacked area
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.fill_between(scales, 0, garbled_rates, label='Garbled', color='#e74c3c', alpha=0.7)
    ax.fill_between(scales, garbled_rates,
                     [g + n for g, n in zip(garbled_rates, neutral_rates)],
                     label='Neutral', color='#f39c12', alpha=0.7)
    ax.fill_between(scales,
                     [g + n for g, n in zip(garbled_rates, neutral_rates)],
                     [100] * len(scales),
                     label='Concept Mentioned', color='#27ae60', alpha=0.7)

    ax.set_xlabel('Steering Scale', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage of Vectors (%)', fontsize=12, fontweight='bold')
    ax.set_title('Steering Output Quality by Scale (Aggregated Across All Vectors)',
                 fontsize=13, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.grid(axis='both', alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', fontsize=10)

    # Mark optimal zone
    ax.axvspan(0.3, 0.7, alpha=0.1, color='blue', label='Optimal range')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved concept success rate plot to {output_path}")
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
    vector_norms, scale_results = parse_summary_file(summary_file)

    print(f"\nFound {len(vector_norms)} vectors")
    print(f"Found scale results for {len(scale_results)} vectors")

    # Create plots directory
    plots_dir = Path("/Users/charleswestphal/Documents/UCL/pid_on_steering_vectors/plots")
    plots_dir.mkdir(exist_ok=True)

    # Generate plots
    plot_vector_norms(vector_norms, plots_dir / "vector_norms.png")
    plot_scale_heatmap(scale_results, plots_dir / "scale_heatmap.png")
    plot_concept_success_rate(scale_results, plots_dir / "concept_success_by_scale.png")

    print(f"\n✓ All plots saved to {plots_dir}/")
    print("\nGenerated plots:")
    print("  1. vector_norms.png - Bar chart of vector L2 norms")
    print("  2. scale_heatmap.png - Heatmap of output quality by vector and scale")
    print("  3. concept_success_by_scale.png - Stacked area chart of output types by scale")


if __name__ == "__main__":
    main()
