# Steering Vector Performance Issues & Improvements

## Problem Diagnosis

Your steering vectors aren't working because:

1. **Null vector dominance**: The null vector has norm ~1279 while concept activations have norm ~93-95
   - When subtracting: `concept (95) - null (1279) ≈ -1279`
   - The result is essentially just the negated null vector
   - No concept-specific information is preserved

2. **Garbled output**: The generation shows control characters (^s^g ^y^w) indicating the model is generating invalid tokens
   - This suggests steering is pushing activations into invalid regions of the space

3. **Zero probability increases**: Concept tokens don't increase in probability at all

## Recommended Improvements

### 1. Remove Null Vector Subtraction (DONE ✅)

**Change**: Use `method="zeros"` for null vector computation
- This means we use the raw concept activations directly
- No subtraction, just: `steering_vector = concept_activation`

**Why**: Standard steering vector methods work this way - they use the activation difference between concepts OR just the raw activation

### 2. Better Prompt Templates

Current: `"The topic is dogs"` - too meta, not natural

**Improve to**:
```python
dogs_prompts = [
    "I love dogs",
    "Dogs are amazing",  
    "My dog is",
    "Puppies are so",
    "The dog was",
    "A golden retriever",
    "Pet dogs are",
    "Canine behavior shows"
]
```

**Why**: More natural language that actually activates dog-related features

### 3. Use Contrastive Pairs (Traditional Method)

Instead of null-diff, use proven contrastive method:
```python
dogs_activation = extract_mean_activation(dog_prompts)
generic_activation = extract_mean_activation(generic_prompts) 
steering_vector = dogs_activation - generic_activation
```

Where generic_prompts are neutral:
```python
generic_prompts = [
    "The thing is",
    "It was", 
    "This is",
    "That was",
    "Something about"
]
```

### 4. Increase Steering Scale

Current: scale=2.0
**Try**: scale=10.0 to 50.0

**Why**: Many steering vector papers use much higher scales

### 5. Try Different Layers

Current: Layer 24 (middle-late)
**Try**: 
- Later layers (28-32) - often better for semantic steering
- Earlier layers (8-16) - sometimes better for specific features

### 6. Normalize Vectors

Before applying, normalize to unit length:
```python
steering_vector = steering_vector / torch.norm(steering_vector)
```

**Why**: Prevents magnitude from dominating, focuses on direction

## Quick Fix for example.py

Change line 84:
```python
# OLD:
null_vector = extractor.compute_null_vector(target_layer)

# NEW:
null_vector = extractor.compute_null_vector(target_layer, method="zeros")
```

This will make it use raw activations without subtraction.

## Alternative: Contrastive Approach

For a working baseline, try:
```python
# Get dog activations
dogs_activation = extractor.extract_mean_activation(dog_prompts, layer_idx)

# Get neutral/generic activations  
neutral_prompts = ["The", "It", "This", "That", "Something"]
neutral_activation = extractor.extract_mean_activation(neutral_prompts, layer_idx)

# Create steering vector
dogs_vector = VectorComputer.from_traditional_diff(
    activation_a=dogs_activation,
    activation_b=neutral_activation,
    layer_idx=layer_idx,
    concept="dogs"
)

# Apply with higher scale
result = generator.generate(
    prompt="Write about",
    steering_vector=dogs_vector,
    scale=20.0  # Much higher!
)
```

## Expected Results After Fix

With these changes, you should see:
- ✅ Normal text generation (no control characters)
- ✅ Concept token probabilities increasing by 100-1000%+
- ✅ Generated text actually mentioning the concept
- ✅ Clear difference between baseline and steered

## Summary

**Root cause**: Null vector subtraction doesn't work because the null vector magnitude dominates

**Solution**: Either:
1. Use raw activations (zeros null vector) ✅ IMPLEMENTED
2. Use contrastive pairs (concept vs neutral) - proven method
3. Better prompts + higher scales + later layers

**Next step**: Update example.py to use `method="zeros"` and test
