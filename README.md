# Steering Vectors Research Library

A PyTorch-based library for research on activation-based steering vectors for language models, with a focus on novel null-vector diffing methods and vector combination techniques.

## Overview

This library implements methods for:
1. **Null-vector based steering**: Diffing concept activations against an "average embedding space" null vector
2. **Multiple combination methods**: Mean, max, RMS-signed, and traditional difference
3. **Comprehensive evaluation**: Probability analysis, concept presence scoring, and LLM-based judging
4. **Flexible experimentation**: Support for any HuggingFace transformer model

## Key Features

### Novel Null-Vector Method
Instead of requiring contrastive pairs of datasets, this library allows you to:
- Extract activations from a single concept dataset
- Compute a "null vector" by averaging all token embeddings
- Create steering vectors by diffing concept activations from this null vector

### Vector Combination Methods
Combine multiple steering vectors using:
- **Mean**: `(a + b) / 2`
- **Max**: `elementwise_max(a, b)`
- **RMS-Signed**: `sign(a+b) · sqrt((a² + b²) / 2)`
- **Diff**: `a - b` (traditional contrastive method)

### Comprehensive Analysis
- Token probability shift analysis
- Concept presence scoring in generated text
- LLM-based generation quality evaluation
- Top-k token tracking with/without steering

## Installation

```bash
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.35+
- CUDA-capable GPU (recommended)

## Quick Start

```python
from models import ModelHandler
from extraction import ActivationExtractor
from vectors import VectorComputer
from steering import SteeredGenerator
from data import DatasetBuilder

# 1. Load model
model_handler = ModelHandler(model_name="Qwen/Qwen2.5-3B")

# 2. Create components
extractor = ActivationExtractor(model_handler)
generator = SteeredGenerator(model_handler)

# 3. Create a concept dataset
dogs_dataset = DatasetBuilder.create_dog_dataset()
prompts = dogs_dataset.get_prompts()

# 4. Extract activations
layer_idx = 20  # Choose a layer
dogs_activation = extractor.extract_mean_activation(
    prompts=prompts,
    layer_idx=layer_idx
)

# 5. Compute null vector
null_vector = extractor.compute_null_vector(layer_idx)

# 6. Create steering vector
dogs_vector = VectorComputer.from_diff_with_null(
    concept_activation=dogs_activation,
    null_vector=null_vector,
    layer_idx=layer_idx,
    concept="dogs"
)

# 7. Generate with steering
result = generator.generate(
    prompt="Let me tell you about",
    steering_vector=dogs_vector,
    scale=2.0,
    max_new_tokens=50
)

print(result['text'])
```

## Example Usage

Run the comprehensive example:

```bash
python example.py
```

This will:
- Load Qwen2.5-3B model
- Create "dogs" and "Golden Gate Bridge" concept datasets
- Compute null vector
- Extract activations and create steering vectors
- Test all combination methods (mean, max, RMS-signed, diff)
- Analyze probability shifts for concept tokens
- Evaluate generation quality
- Save all results to `outputs/`

## Library Structure

```
├── models.py          # Model loading and hook management
├── extraction.py      # Activation extraction and null vector computation
├── vectors.py         # Steering vector creation and combination
├── steering.py        # Steered generation
├── evaluation.py      # Evaluation and analysis tools
├── data.py           # Dataset builders and prompt templates
├── utils.py          # Utility functions
├── example.py        # Comprehensive example
└── requirements.txt  # Dependencies
```

## Core Components

### ModelHandler (`models.py`)
Manages model loading and provides hook registration for:
- Activation extraction
- Steering vector application during generation

### ActivationExtractor (`extraction.py`)
Extracts activations from model layers:
- Single prompt or batch extraction
- Configurable token positions (last, first, all)
- Null vector computation

### VectorComputer (`vectors.py`)
Creates and combines steering vectors:
- `from_diff_with_null()`: Novel null-vector method
- `from_traditional_diff()`: Contrastive baseline
- `combine_vectors()`: Multiple combination methods

### SteeredGenerator (`steering.py`)
Generates text with steering applied:
- Single or batch generation
- Configurable scaling factors
- Probability tracking and analysis

### SteeringEvaluator (`evaluation.py`)
Evaluates steering effectiveness:
- Token probability shift analysis
- Concept presence scoring
- LLM-based quality judging

### DatasetBuilder (`data.py`)
Creates concept datasets:
- Pre-built datasets (dogs, Golden Gate Bridge)
- Custom dataset creation
- Multiple prompt templates

## Research Applications

### 1. Null-Vector Validation
Compare null-vector method against traditional contrastive methods:

```python
# Traditional method
traditional = VectorComputer.from_traditional_diff(
    activation_a=concept_a_activation,
    activation_b=concept_b_activation,
    layer_idx=layer_idx,
    concept="A vs B"
)

# Null-vector method
null_a = VectorComputer.from_diff_with_null(
    concept_activation=concept_a_activation,
    null_vector=null_vector,
    layer_idx=layer_idx,
    concept="A"
)

null_b = VectorComputer.from_diff_with_null(
    concept_activation=concept_b_activation,
    null_vector=null_vector,
    layer_idx=layer_idx,
    concept="B"
)

# Compare: traditional vs (null_a - null_b)
```

### 2. Vector Combination Studies
Investigate how different combination methods affect steering:

```python
# Test all combination methods
mean_vec = VectorComputer.combine_vectors(vec_a, vec_b, method="mean")
max_vec = VectorComputer.combine_vectors(vec_a, vec_b, method="max")
rms_vec = VectorComputer.combine_vectors(vec_a, vec_b, method="rms_signed")

# Compare generation quality
for vec in [mean_vec, max_vec, rms_vec]:
    result = generator.generate(prompt, steering_vector=vec, scale=2.0)
    # Analyze results...
```

### 3. Probability Analysis
Study how steering affects token probabilities:

```python
analysis = evaluator.analyze_token_probability_shifts(
    prompt="Test prompt",
    steering_vector=my_vector,
    concept_tokens=["dog", "puppy", "canine"],
    scales=[0.5, 1.0, 2.0, 5.0],
    max_new_tokens=20
)

# Examine probability changes across scales
```

## Configuration

### Supported Models
The library works with any HuggingFace causal language model. Tested with:
- Qwen2.5 (1.5B, 3B, 7B)
- Llama-2/3 (7B, 13B)
- GPT-2 (medium, large, xl)
- Pythia (1B, 1.4B, 2.8B)

### Layer Selection
Use `model_handler.get_layer_range()` to select layers:
- `"all"`: All layers
- `"early"`: First 1/3 of layers
- `"middle"`: Middle 1/3 of layers
- `"late"`: Last 1/3 of layers
- `"middle-late"`: Middle and late layers (recommended)

### Steering Parameters
- **scale**: Magnitude multiplier (typical range: 0.5 - 5.0)
- **layer_idx**: Which layer to apply steering
- **position**: Token position for extraction ("last", "first", "all")

## Output Format

### Steering Vectors
Saved as PyTorch tensors with metadata:
```python
{
    'vector': torch.Tensor,
    'layer_idx': int,
    'concept': str,
    'method': str,
    'metadata': dict
}
```

### Evaluation Results
Saved as JSON with:
- Generation comparisons (baseline vs steered)
- Probability distributions
- Concept presence scores
- Statistics and aggregations

## Advanced Usage

### Custom Datasets
```python
from data import ConceptDataset

custom_dataset = ConceptDataset(
    concept="your_concept",
    examples=["example 1", "example 2", ...],
    template="topic"  # or "simple", "sentence", "question"
)
```

### Multi-Vector Combinations
```python
vectors = [vec1, vec2, vec3, ...]
combined = VectorComputer.combine_multiple_vectors(
    vectors=vectors,
    method="mean"
)
```

### Batch Evaluation
```python
results = evaluator.evaluate_generation_quality(
    prompts=test_prompts,
    steering_vector=my_vector,
    scale=2.0
)

scoring = evaluator.automated_concept_scoring(
    generations=results['generations'],
    concept_keywords=["keyword1", "keyword2"]
)
```

## Citation

If you use this library in your research, please cite:

```bibtex
@software{steering_vectors_research,
  title={Steering Vectors Research Library},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/steering-vectors}
}
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Troubleshooting

### CUDA Out of Memory
- Use smaller models (Qwen2.5-1.5B instead of 7B)
- Reduce batch sizes
- Use `torch.float16` instead of `torch.float32`

### Slow Generation
- Use smaller `max_new_tokens`
- Process prompts in smaller batches
- Consider using a faster model

### Import Errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (3.8+ required)

## Contact

For questions or issues, please open a GitHub issue.
