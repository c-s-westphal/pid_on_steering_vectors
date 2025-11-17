# Plots Directory Structure

All plots are automatically generated from HPC results data.

## Directory Organization

```
plots/
├── {model-name}/               # e.g., qwen-7b, qwen-14b
│   └── {timestamp}/            # e.g., 20251116_203923
│       ├── vector_norms.png
│       ├── scale_heatmap.png
│       └── concept_success_by_scale.png
│
└── token_probabilities/
    └── {model-name}/
        └── {timestamp}/
            ├── {model}_token_prob_heatmap.png
            ├── {model}_concept_comparison.png
            ├── {model}_split_half_vs_traditional.png
            └── {model}_best_performers.png
```

## Generating Plots

### From Summary Files

```bash
python plot_results.py
```

This automatically:
- Finds the latest summary.txt file
- Extracts model name and timestamp
- Creates plots in `plots/{model-name}/{timestamp}/`

### From Token Probability Data

```bash
python plot_token_probabilities.py --results-dir hpc_results/multi_model/{model}/{timestamp}
```

This creates plots in `plots/token_probabilities/{model-name}/{timestamp}/`

## Plot Types

### Vector Analysis Plots
1. **vector_norms.png** - Bar chart showing L2 norms of all steering vectors
2. **scale_heatmap.png** - Heatmap of steering quality (green=concept, red=garbled)
3. **concept_success_by_scale.png** - Stacked area chart showing % success vs garbled by scale

### Token Probability Plots
1. **token_prob_heatmap.png** - Heatmap of probability changes for dog/bridge tokens
2. **concept_comparison.png** - Comparison of concept promotion across methods
3. **split_half_vs_traditional.png** - Split-half probe vs traditional methods
4. **best_performers.png** - Top performing vectors for each concept
