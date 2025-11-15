# Getting Started with Steering Vectors Research

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- PyTorch (2.0+)
- Transformers (4.35+)
- Accelerate
- SentencePiece
- Protobuf

### 2. Verify Installation

```bash
python test_imports.py
```

You should see:
```
✓ models.py
✓ extraction.py
✓ vectors.py
✓ steering.py
✓ evaluation.py
✓ data.py
✓ utils.py

All imports successful! Library is ready to use.
```

## Quick Start: First Experiment

### Option 1: Run the Full Example

```bash
python example.py
```

This comprehensive example will:
1. Load Qwen2.5-3B model
2. Create "dogs" and "Golden Gate Bridge" datasets
3. Compute null vector using average embedding method
4. Extract activations for both concepts
5. Create steering vectors using null-diff method
6. Test all combination methods:
   - Mean: `(dog + bridge) / 2`
   - Max: `elementwise_max(dog, bridge)`
   - RMS-signed: `sign(a+b) · sqrt((a² + b²) / 2)`
   - Diff: `dog - bridge`
7. Analyze probability shifts for concept tokens
8. Evaluate generation quality
9. Save all results to `outputs/`

**Expected runtime**: 10-20 minutes on GPU

**Expected outputs**:
- `outputs/steering_vectors/` - All steering vectors (.pt files)
- `outputs/evaluations/` - JSON files with analysis results

### Option 2: Quick Interactive Test

Create a file `quick_test.py`:

```python
import torch
from models import ModelHandler
from extraction import ActivationExtractor
from vectors import VectorComputer
from steering import SteeredGenerator
from data import DatasetBuilder

# Load model (this will download ~6GB on first run)
print("Loading model...")
model_handler = ModelHandler(model_name="Qwen/Qwen2.5-3B")

# Create components
extractor = ActivationExtractor(model_handler)
generator = SteeredGenerator(model_handler)

# Create "dogs" dataset
dogs_dataset = DatasetBuilder.create_dog_dataset()
prompts = dogs_dataset.get_prompts()
print(f"Created dataset with {len(prompts)} prompts")

# Choose a layer (middle-late range)
layers = model_handler.get_layer_range("middle-late")
layer_idx = layers[len(layers) // 2]
print(f"Using layer {layer_idx}")

# Extract activations
print("Extracting activations...")
dogs_activation = extractor.extract_mean_activation(
    prompts=prompts,
    layer_idx=layer_idx
)

# Compute null vector
print("Computing null vector...")
null_vector = extractor.compute_null_vector(layer_idx)

# Create steering vector
dogs_vector = VectorComputer.from_diff_with_null(
    concept_activation=dogs_activation,
    null_vector=null_vector,
    layer_idx=layer_idx,
    concept="dogs"
)
print(f"Created steering vector: {dogs_vector}")

# Test generation
test_prompt = "Let me tell you about"

print("\n" + "="*60)
print("BASELINE GENERATION")
print("="*60)
baseline = generator.generate(
    prompt=test_prompt,
    steering_vector=None,
    max_new_tokens=30
)
print(baseline['text'])

print("\n" + "="*60)
print("STEERED GENERATION (dogs, scale=2.0)")
print("="*60)
steered = generator.generate(
    prompt=test_prompt,
    steering_vector=dogs_vector,
    scale=2.0,
    max_new_tokens=30
)
print(steered['text'])

print("\n" + "="*60)
print("SUCCESS! The library is working correctly.")
print("="*60)
```

Run it:
```bash
python quick_test.py
```

## Understanding the Outputs

### Steering Vectors

Saved as `.pt` files containing:
```python
{
    'vector': torch.Tensor,      # The actual steering vector
    'layer_idx': int,             # Which layer it was extracted from
    'concept': str,               # Concept name (e.g., "dogs")
    'method': str,                # Creation method (e.g., "diff_from_null")
    'metadata': dict              # Additional information
}
```

Load them with:
```python
from vectors import SteeringVector
vec = SteeringVector.load("outputs/steering_vectors/dogs_vector.pt")
```

### Evaluation Results

JSON files containing:

**Probability Analysis** (`*_probability_analysis.json`):
- Token probability changes at different steering scales
- Per-token statistics showing baseline vs steered probabilities
- Relative and absolute changes

**Generation Quality** (`*_generation_quality.json`):
- Side-by-side comparisons of baseline vs steered generations
- Multiple test prompts
- Steering metadata

**Concept Scoring** (`*_concept_scoring.json`):
- Keyword occurrence counts
- Presence rates (keywords per word)
- Baseline vs steered comparisons

## Next Steps

### Experiment 1: Different Concepts

Create your own concept:

```python
from data import DatasetBuilder

# Create custom dataset
my_concept = DatasetBuilder.create_custom_dataset(
    concept="cats",
    variations=["cat", "cats", "feline", "kitten", "kittens"]
)

# Get prompts and extract activations
prompts = my_concept.get_prompts()
# ... continue with extraction
```

### Experiment 2: Compare Combination Methods

```python
# Create two concept vectors
dog_vec = ...  # from null-diff method
cat_vec = ...  # from null-diff method

# Test all combinations
mean_vec = VectorComputer.combine_vectors(dog_vec, cat_vec, method="mean")
max_vec = VectorComputer.combine_vectors(dog_vec, cat_vec, method="max")
rms_vec = VectorComputer.combine_vectors(dog_vec, cat_vec, method="rms_signed")

# Generate with each
for vec in [mean_vec, max_vec, rms_vec]:
    result = generator.generate(
        prompt="Tell me about",
        steering_vector=vec,
        scale=2.0
    )
    print(f"{vec.concept}: {result['text']}")
```

### Experiment 3: Probability Analysis

```python
from evaluation import SteeringEvaluator

evaluator = SteeringEvaluator(model_handler, generator)

# Analyze how dog-related tokens change in probability
analysis = evaluator.analyze_token_probability_shifts(
    prompt="Let me tell you about",
    steering_vector=dogs_vector,
    concept_tokens=["dog", "dogs", "puppy", "canine"],
    scales=[0.5, 1.0, 2.0, 5.0],
    max_new_tokens=20
)

# Check statistics
for token, stats in analysis['statistics']['per_token'].items():
    print(f"\nToken: {token}")
    print(f"Baseline: {stats['baseline_avg_prob']:.6f}")
    for scale, effect in stats['scale_effects'].items():
        change_pct = effect['relative_change'] * 100
        print(f"  Scale {scale}: {effect['avg_prob']:.6f} ({change_pct:+.1f}%)")
```

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution**: Use a smaller model or float16:
```python
model_handler = ModelHandler(
    model_name="Qwen/Qwen2.5-1.5B",  # Smaller model
    torch_dtype=torch.float16         # Half precision
)
```

### Issue: Model Download Too Slow

**Solution**: The first run downloads ~6GB. Consider:
1. Using a smaller model initially
2. Checking your internet connection
3. Using a cached model if available

### Issue: Generation Is Slow

**Solution**:
1. Reduce `max_new_tokens`
2. Use a smaller model
3. Reduce number of test prompts
4. Ensure CUDA is being used:
   ```python
   print(model_handler.device)  # Should show 'cuda'
   ```

### Issue: Steering Seems Ineffective

Try:
1. Increase scale: `scale=5.0` or `scale=10.0`
2. Use different layers (try late layers: `layer_idx=25` for a 32-layer model)
3. Check that concept tokens are increasing in probability
4. Verify steering vector norm isn't too small

## Key Research Questions

This library is designed to help answer:

1. **Does the null-vector method work?**
   - Compare null-diff vectors to traditional contrastive vectors
   - Measure correlation, generation quality, probability shifts

2. **Which combination method is best?**
   - Compare mean, max, RMS-signed, and diff
   - Which preserves information better?
   - Which produces stronger/cleaner steering?

3. **How do concepts combine?**
   - Does `mean(dog, bridge)` steer toward both concepts?
   - Do incompatible concepts interfere or coexist?
   - Is combination linear or nonlinear?

## Additional Resources

- **README.md** - Full library documentation
- **example.py** - Comprehensive example with all features
- **models.py through utils.py** - Source code with detailed docstrings
- **test_imports.py** - Verify installation

## Getting Help

If you encounter issues:

1. Check the error message carefully
2. Verify dependencies are installed: `pip list | grep -E "(torch|transformers)"`
3. Ensure GPU is available: `python -c "import torch; print(torch.cuda.is_available())"`
4. Try the quick test first before running full example
5. Check that you have enough disk space (~10GB for model + outputs)

Happy researching!
