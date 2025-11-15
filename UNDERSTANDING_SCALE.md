# Understanding the Steering Scale Parameter

## What Scale Means

```python
steered_activation = original_activation + (scale * steering_vector)
```

The `scale` parameter is a **multiplier** that controls how much we shift the model's internal activations:

- `scale = 0.0`: No steering (baseline generation)
- `scale = 0.5`: Add half the steering vector
- `scale = 1.0`: Add the full steering vector
- `scale = 2.0`: Add twice the steering vector

## Why Your Current Steering Fails

You're seeing garbled output (`^y^w ^s^g`) because:

**Problem**: Your steering vectors have huge norms (~1249) instead of ~93

**Root Cause**: You're loading old cached vectors from `outputs/` that were computed with the broken null vector method (before the fix)

**Solution**: Delete the old cached vectors!

```bash
rm -rf outputs/
python example.py
```

## Finding the Right Scale

The optimal scale depends on the **magnitude of your steering vector**:

### For Raw Activations (norm ~50-100)
- **Scale 0.1-0.3**: Very subtle steering
- **Scale 0.5-1.0**: Moderate steering (recommended starting point)
- **Scale 1.5-3.0**: Strong steering
- **Scale >3.0**: Often causes garbled output

### For Normalized Vectors (norm = 1.0)
- **Scale 5-20**: Moderate steering
- **Scale 20-50**: Strong steering
- **Scale >50**: Often too much

## Recommended Approach

### Option 1: Use Raw Activations (Simplest)

```python
# Just use the concept activation directly, no subtraction
dog_activation = extractor.extract_mean_activation(dog_prompts, layer_idx)

# Create steering vector (no null subtraction!)
dog_vector = SteeringVector(
    vector=dog_activation,  # Just the raw activation
    layer_idx=layer_idx,
    concept="dogs"
)

# Use scale 0.3-1.0
result = generator.generate(
    prompt="Write about",
    steering_vector=dog_vector,
    scale=0.5  # Start here
)
```

### Option 2: Normalize Then Scale Higher

```python
# Normalize the vector to unit length
dog_vector_normalized = dog_activation / torch.norm(dog_activation)

# Now you can use much higher scales
result = generator.generate(
    prompt="Write about",
    steering_vector=SteeringVector(dog_vector_normalized, layer_idx, "dogs"),
    scale=20.0  # Higher scales work with normalized vectors
)
```

### Option 3: Use Contrastive (Traditional Method)

```python
# Get neutral baseline
neutral_prompts = ["The", "It", "This", "That", "Something"]
neutral_activation = extractor.extract_mean_activation(neutral_prompts, layer_idx)

# Subtract to get direction
dog_vector = dog_activation - neutral_activation

# Use moderate scale
result = generator.generate(
    prompt="Write about",
    steering_vector=SteeringVector(dog_vector, layer_idx, "dogs"),
    scale=1.0
)
```

## Testing Different Scales

Use `simple_test.py` to find the right scale:

```bash
python simple_test.py
```

This tests scales from 0.1 to 2.0 and shows you:
- Which scales give valid output
- Which scales mention the concept
- When garbled output starts

## Signs You Need to Adjust Scale

### Scale Too High
- ❌ Garbled output (control characters: `^y^w ^s^g`)
- ❌ Nonsensical text
- ❌ Repetitive loops
- → **Solution**: Reduce scale by 50% (2.0 → 1.0)

### Scale Too Low
- ⚠️ Valid text but no concept mentions
- ⚠️ Identical to baseline
- → **Solution**: Increase scale by 2x (0.5 → 1.0)

### Scale Just Right
- ✅ Valid, coherent text
- ✅ Mentions concept naturally
- ✅ Retains model capability
- → **This is your target!**

## Example Outputs by Scale

Using raw dog activation (norm ~90):

**Scale 0.1** (too low):
```
Write a short paragraph about the importance of reading books...
```
No dog mentions.

**Scale 0.5** (good):
```
Write a short paragraph about my dog Max. He's a golden retriever...
```
Natural dog-related content!

**Scale 2.0** (too high):
```
Write a short paragraph about ^y^w ^s^g ^n dog dog dog ^w^~...
```
Garbled, repetitive.

## Quick Fix Commands

### 1. Delete old cached vectors
```bash
rm -rf outputs/
```

### 2. Test different scales
```bash
python simple_test.py
```

### 3. Once you find good scale, update example.py
Replace all `scale=1.5` with your optimal scale (probably 0.5-1.0)

## Summary

- **Scale is a multiplier**: How much steering to add
- **Your vectors are broken**: Old cached files with huge norms
- **Fix**: Delete `outputs/` and regenerate
- **Good scale**: 0.3-1.0 for raw activations (norm ~90)
- **Test first**: Use `simple_test.py` to find optimal scale
- **Watch for garbled output**: Sign scale is too high

The steering will work once you regenerate the vectors with the fixed code!
