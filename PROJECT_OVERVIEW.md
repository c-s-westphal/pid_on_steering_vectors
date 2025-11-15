# Project Overview: Steering Vectors Research Library

## 🎯 Project Goals

This library implements a novel approach to steering vector research with three main innovations:

### 1. **Null-Vector Diffing Method**
Instead of requiring contrastive dataset pairs (e.g., "love" vs "hate"), we:
- Use a single concept dataset (e.g., just "dogs")
- Compute a "null vector" representing the average embedding space
- Create steering vectors by: `steering_vector = concept_activation - null_vector`

**Hypothesis**: This should capture the essence of the concept without needing an opposite.

### 2. **Novel Vector Combination Methods**
Traditional methods use simple difference: `vector_a - vector_b`

We implement and compare:
- **Mean**: `(a + b) / 2` - averages the directions
- **Max**: `elementwise_max(a, b)` - takes maximum at each dimension
- **RMS-Signed**: `sign(a+b) · sqrt((a² + b²) / 2)` - preserves magnitude information with sign
- **Diff**: `a - b` - traditional baseline

**Hypothesis**: Different combination methods may preserve information better than simple differencing.

### 3. **Comprehensive Evaluation Framework**
- Token probability shift analysis
- Concept presence scoring
- LLM-based quality evaluation
- Quantitative metrics for comparing methods

## 📁 Project Structure

```
pid_on_steering_vectors/
├── Core Library Files
│   ├── models.py          # Model loading and hook management
│   ├── extraction.py      # Activation extraction & null vector computation
│   ├── vectors.py         # Steering vector creation & combination
│   ├── steering.py        # Steered text generation
│   ├── evaluation.py      # Evaluation and analysis tools
│   ├── data.py           # Dataset builders and prompt templates
│   └── utils.py          # Utility functions
│
├── Example & Demo Scripts
│   ├── example.py         # Comprehensive example workflow
│   ├── interactive_demo.py # Step-by-step demo with detailed output
│   └── test_imports.py    # Verify installation
│
├── Documentation
│   ├── README.md          # Full library documentation
│   ├── GETTING_STARTED.md # Installation and quick start guide
│   └── PROJECT_OVERVIEW.md # This file
│
└── Configuration
    ├── requirements.txt   # Python dependencies
    ├── __init__.py       # Package initialization
    └── .gitignore        # Git ignore rules
```

## 🔬 Research Questions

This library is designed to help answer:

### Primary Questions

1. **Does null-vector diffing work as well as traditional contrastive methods?**
   - Extract vectors using both methods
   - Compare generation quality, probability shifts
   - Measure correlation between the two approaches

2. **Which vector combination method preserves the most information?**
   - Compare mean, max, RMS-signed, and diff
   - Measure steering effectiveness for combined concepts
   - Analyze whether concepts interfere or cooperate

3. **How do steering vectors behave when combined?**
   - Can we steer toward two concepts simultaneously?
   - Does combination create emergent properties?
   - Is the relationship linear or nonlinear?

### Secondary Questions

4. **What role does the null vector play?**
   - Does it truly represent "average embedding space"?
   - How does it vary across layers?
   - Is it concept-agnostic?

5. **How do different scales affect steering?**
   - Is there an optimal scale for each concept?
   - Do scales transfer across concepts?
   - What happens at extreme scales?

6. **How layer-specific is steering?**
   - Do early/middle/late layers steer differently?
   - Is there a "sweet spot" layer?
   - Can we combine vectors from different layers?

## 🧪 Experimental Workflow

### Standard Experiment Pipeline

```python
# 1. Setup
model = ModelHandler("Qwen/Qwen2.5-3B")
extractor = ActivationExtractor(model)
generator = SteeredGenerator(model)
evaluator = SteeringEvaluator(model, generator)

# 2. Create datasets
concept_dataset = DatasetBuilder.create_custom_dataset("your_concept")

# 3. Compute null vector (once per layer)
null_vector = extractor.compute_null_vector(layer_idx)

# 4. Extract concept activations
concept_activation = extractor.extract_mean_activation(
    prompts=concept_dataset.get_prompts(),
    layer_idx=layer_idx
)

# 5. Create steering vector
steering_vector = VectorComputer.from_diff_with_null(
    concept_activation=concept_activation,
    null_vector=null_vector,
    layer_idx=layer_idx,
    concept="your_concept"
)

# 6. Generate and evaluate
results = generator.compare_generations(
    prompt="test prompt",
    steering_vectors=[steering_vector],
    scales=[1.0, 2.0, 5.0]
)

# 7. Analyze
prob_analysis = evaluator.analyze_token_probability_shifts(...)
concept_scoring = evaluator.automated_concept_scoring(...)
```

## 📊 Key Features

### Activation Extraction
- ✅ Extract from any layer
- ✅ Configurable token positions (last, first, all)
- ✅ Batch processing
- ✅ Automatic hook management

### Null Vector Computation
- ✅ Average all token embeddings
- ✅ Forward pass through model
- ✅ Layer-specific null vectors
- ✅ Cached for efficiency

### Steering Vector Creation
- ✅ Null-diff method
- ✅ Traditional contrastive method
- ✅ 4 combination methods (mean, max, RMS-signed, diff)
- ✅ Save/load functionality
- ✅ Metadata tracking

### Generation & Evaluation
- ✅ Configurable steering scales
- ✅ Side-by-side comparisons
- ✅ Token probability tracking
- ✅ Concept presence scoring
- ✅ LLM-based quality judging
- ✅ JSON export for analysis

## 🚀 Getting Started

### Quick Install
```bash
pip install -r requirements.txt
python test_imports.py  # Verify installation
```

### Run Your First Experiment
```bash
python interactive_demo.py  # Step-by-step walkthrough
# OR
python example.py  # Full experiment with saved outputs
```

### Create Custom Concept
```python
from data import DatasetBuilder

my_concept = DatasetBuilder.create_custom_dataset(
    concept="your_topic",
    variations=["variant1", "variant2", ...]
)
```

## 📈 Expected Results

### What Success Looks Like

1. **Effective Steering**
   - Baseline generation is neutral
   - Steered generation mentions the concept
   - Higher scales = stronger steering

2. **Probability Shifts**
   - Concept tokens increase in probability
   - Effect scales with steering strength
   - Shifts are measurable and significant

3. **Concept Presence**
   - Steered generations score higher on concept keywords
   - Quantifiable improvement over baseline
   - Consistent across multiple prompts

### Comparison Metrics

To validate null-vector method:
- **Correlation**: Do null-diff vectors correlate with traditional vectors?
- **Effectiveness**: Do they steer equally well?
- **Efficiency**: Is single dataset sufficient?

To compare combination methods:
- **Strength**: Which produces strongest steering?
- **Coherence**: Which generates most natural text?
- **Information**: Which preserves both concepts best?

## 🔧 Customization Points

### Models
Change in `models.py` or at initialization:
```python
ModelHandler(model_name="meta-llama/Llama-2-7b-hf")
```

### Layers
```python
# Try different layer ranges
early_layers = model.get_layer_range("early")
late_layers = model.get_layer_range("late")
```

### Prompts
```python
# Use different templates
dataset.get_prompts(template="question")  # vs "topic", "simple", etc.
```

### Evaluation
```python
# Custom concept keywords
evaluator.automated_concept_scoring(
    generations=results,
    concept_keywords=["custom", "keywords", "here"]
)
```

## 📝 Output Files

### After Running `example.py`

```
outputs/
├── steering_vectors/
│   ├── dogs_vector.pt              # Single concept vectors
│   ├── bridge_vector.pt
│   ├── dogs_bridge_mean.pt         # Combined vectors
│   ├── dogs_bridge_max.pt
│   ├── dogs_bridge_rms.pt
│   ├── dogs_bridge_diff.pt
│   └── traditional_diff.pt         # Baseline
│
└── evaluations/
    ├── dogs_probability_analysis.json
    ├── dogs_generation_quality.json
    └── dogs_concept_scoring.json
```

### File Sizes
- Steering vectors: ~10-50 MB each (depends on hidden size)
- Evaluation JSONs: ~100 KB - 1 MB each

## 🎓 Research Applications

### For Papers/Theses
- Novel null-vector method validation
- Comprehensive comparison of combination methods
- Quantitative steering vector analysis
- Replicable experimental framework

### For Exploration
- Test different concepts and their interactions
- Explore layer-wise steering differences
- Investigate scale effects
- Discover emergent properties of combinations

### For Development
- Baseline for steering vector research
- Modular components for custom methods
- Evaluation framework for new approaches

## 📚 Next Steps

1. **Install and test**: Run `interactive_demo.py`
2. **Understand workflow**: Read through `example.py`
3. **Try custom concepts**: Use `DatasetBuilder`
4. **Compare methods**: Test all combination approaches
5. **Analyze results**: Use evaluation tools
6. **Iterate**: Refine based on findings

## 🤝 Contributing

Key areas for extension:
- Additional combination methods
- New evaluation metrics
- Different null vector approaches
- Multi-layer steering
- Concept arithmetic exploration

## 📖 Additional Resources

- **README.md**: Detailed API documentation
- **GETTING_STARTED.md**: Installation and troubleshooting
- **Source code**: All files have extensive docstrings
- **Example scripts**: `example.py` and `interactive_demo.py`

---

**Happy researching! 🚀**
