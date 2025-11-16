#!/bin/bash

# Run multi-model experiments and save output
# Usage: ./run_experiments.sh

echo "Starting multi-model steering vector experiments..."
echo "Timestamp: $(date)"
echo ""

# Create outputs directory if it doesn't exist
mkdir -p outputs/logs

# Run on Qwen 3B and Llama 7B
python run_multi_model.py --models qwen-3b llama-7b 2>&1 | tee outputs/logs/multi_model_$(date +%Y%m%d_%H%M%S).out

echo ""
echo "Experiments completed!"
echo "Results saved to outputs/multi_model/"
echo "Logs saved to outputs/logs/"
