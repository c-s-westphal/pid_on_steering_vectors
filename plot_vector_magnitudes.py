#!/usr/bin/env python3
"""
Plot vector magnitude analysis from summary files.
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def parse_summary_file(summary_path):
    """Extract vector norms from summary file."""
    with open(summary_path, 'r') as f:
        content = f.read()

    vectors = {}

    # Parse traditional vectors
    trad_pattern = r'(\d+)\.\s+([^:]+):\s+norm = ([\d.]+)'
    for match in re.finditer(trad_pattern, content):
        idx, name, norm = match.groups()
        vectors[name.strip()] = float(norm)

    # Parse probe vectors
    probe_pattern = r'Probe ([^:]+):\s+norm = ([\d.]+)'
    for match in re.finditer(probe_pattern, content):
        name, norm = match.groups()
        vectors[f'Probe {name.strip()}'] = float(norm)

    return vectors

def plot_vector_magnitudes(summary_path, output_dir):
    """Create bar plot of vector magnitudes."""
    vectors = parse_summary_file(summary_path)

    # Separate traditional and probe vectors
    trad_vectors = {k: v for k, v in vectors.items() if not k.startswith('Probe')}
    probe_vectors = {k: v for k, v in vectors.items() if k.startswith('Probe')}

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot traditional vectors
    names = list(trad_vectors.keys())
    norms = list(trad_vectors.values())

    axes[0].barh(range(len(names)), norms, color='steelblue', alpha=0.7)
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels(names)
    axes[0].set_xlabel('Vector Norm (L2)', fontsize=12)
    axes[0].set_title('Traditional Steering Vectors', fontsize=13, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (name, norm) in enumerate(zip(names, norms)):
        axes[0].text(norm + 2, i, f'{norm:.1f}', va='center', fontsize=9)

    # Plot probe vectors
    probe_names = list(probe_vectors.keys())
    probe_norms = list(probe_vectors.values())

    axes[1].barh(range(len(probe_names)), probe_norms, color='coral', alpha=0.7)
    axes[1].set_yticks(range(len(probe_names)))
    axes[1].set_yticklabels(probe_names)
    axes[1].set_xlabel('Vector Norm (L2)', fontsize=12)
    axes[1].set_title('MLP Probe Steering Vectors', fontsize=13, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (name, norm) in enumerate(zip(probe_names, probe_norms)):
        axes[1].text(norm + 0.02, i, f'{norm:.2f}', va='center', fontsize=9)

    # Add annotation about magnitude difference
    if probe_norms and norms:
        avg_trad = np.mean(norms)
        avg_probe = np.mean(probe_norms)
        ratio = avg_trad / avg_probe

        fig.text(0.5, 0.02,
                f'Average traditional norm: {avg_trad:.1f}  |  Average probe norm: {avg_probe:.2f}  |  Ratio: {ratio:.1f}×',
                ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Steering Vector Magnitude Comparison\n(Probe vectors are ~100× smaller)',
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    # Extract model name from path
    parts = summary_path.parts
    model_name = parts[-3]
    timestamp = parts[-2]

    output_path = output_dir / model_name / timestamp / f'{model_name}_vector_magnitudes.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved magnitude plot to {output_path}")
    plt.close()

    return vectors

def main():
    results_dir = Path('hpc_results/multi_model')
    output_dir = Path('plots/token_probabilities')

    # Find all summary files
    summary_files = list(results_dir.rglob('summary.txt'))

    print(f"Found {len(summary_files)} summary files")

    for summary_file in summary_files:
        print(f"\nProcessing: {summary_file}")
        vectors = plot_vector_magnitudes(summary_file, output_dir)

        print("\nVector norms:")
        print("Traditional vectors:")
        for name, norm in vectors.items():
            if not name.startswith('Probe'):
                print(f"  {name:30s}: {norm:8.2f}")

        print("\nProbe vectors:")
        for name, norm in vectors.items():
            if name.startswith('Probe'):
                print(f"  {name:30s}: {norm:8.2f}")

if __name__ == "__main__":
    main()
