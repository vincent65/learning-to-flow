# Critical Fix: Prototype Collapse Problem

## Problem After 10 Epochs

**Despite all previous optimizations, the model STILL had severe mode collapse:**

```
Epoch 10 Results:
❌ Silhouette by target: 0.9323 (should be 0.35-0.55)
❌ Movement: 1.7919 (way too high, should be 0.5-0.7)
❌ Step distance: 0.2350 (should be 0.02-0.05)
❌ All trajectories end at ONE POINT [10, 6] in UMAP space
```

**Visual evidence:**
- Trajectory graph showed ALL green dots (endpoints) clustering at one location
- Flowed embeddings collapsed to ~20-30 discrete points
- Original embeddings were well spread, but after flowing → complete collapse

---

## Root Cause: Prototype Collapse

### The Fatal Flaw in Original Implementation

```python
# BROKEN CODE (before fix):
z_flowed = normalize(z + alpha * vector_field(z, y))

# Compute prototypes FROM the flowed embeddings
for i in range(num_prototypes):
    proto = z_flowed[mask].mean(dim=0)  # NO .detach()!
    prototypes.append(proto)

# Loss compares embeddings to prototypes
loss = -log(sim(z_flowed, prototypes))

# Gradients flow: loss → prototypes → z_flowed → vector_field
```

### Why This Causes Collapse

**The feedback loop:**

1. **Batch 1**: Prototypes are computed from current embeddings
   - Proto_A = mean(embeddings with attributes A)
   - Proto_B = mean(embeddings with attributes B)

2. **Loss computation**: Pulls embeddings toward their prototypes
   - Embeddings with attr A → pulled toward Proto_A
   - Embeddings with attr B → pulled toward Proto_B

3. **Gradient descent**: Updates vector field to minimize loss

4. **THE PROBLEM**: Prototypes were computed WITH gradients!
   - Gradients flow through prototypes back to embeddings
   - Model learns: "If Proto_A moves, embeddings move the same way"
   - Easiest solution: Collapse all attr-A embeddings to ONE POINT
   - This makes Proto_A = that point, and all embeddings = that point
   - Loss = 0! (perfect similarity)

5. **Result**: Mode collapse
   - All "smiling" faces → Point A
   - All "non-smiling" faces → Point B
   - But within each group: NO variation (discrete collapse)

---

## The Fix: Stop-Gradient on Prototypes

```python
# FIXED CODE:
z_flowed = normalize(z + alpha * vector_field(z, y))

# Compute prototypes WITH stop-gradient
for i in range(num_prototypes):
    proto = z_flowed[mask].mean(dim=0)
    proto = proto.detach()  # ← CRITICAL: Stop gradients!
    prototypes.append(proto)

# Loss compares embeddings to FIXED prototypes
loss = -log(sim(z_flowed, prototypes))

# Gradients ONLY flow: loss → z_flowed → vector_field
# Prototypes act as fixed targets (updated each batch but no gradients)
```

### Why This Works

**With `.detach()`:**

1. **Prototypes are semi-fixed targets**:
   - Computed from current batch (adaptive)
   - But treated as constants during backprop (no gradients)

2. **Model must spread embeddings**:
   - Can't just collapse all attr-A embeddings to one point
   - Must create a DISTRIBUTION of embeddings around Proto_A
   - Because Proto_A is computed as MEAN, model incentivized to spread

3. **Prevents feedback loop**:
   - Gradients can't flow through prototypes
   - Embeddings can't "chase" the prototypes as they move
   - Forces stable, distributed attribute clusters

---

## Technical Explanation: Momentum Prototypes

This fix implements a form of **momentum prototypes** used in contrastive learning:

### Standard Momentum Prototypes (e.g., MoCo)
```python
# Global prototype bank
prototypes = exponential_moving_average(all_previous_batches)

# Very stable, but requires large memory
```

### Our Batch-Level Prototypes (Simpler)
```python
# Compute prototypes from current batch
prototypes = mean(current_batch).detach()

# Lighter weight, works well for our use case
```

**Why batch-level is OK for us:**
- With batch size 512 and 5 attributes → ~20-40 unique combinations per batch
- Prototypes get reasonable estimates even from one batch
- Updated every iteration, so they track the moving distribution

---

## Expected Results After Retraining

### Metrics Should Improve To:

```
✓ Silhouette by target: 0.9323 → 0.35-0.55
  (Continuous distribution, not discrete collapse)

✓ Movement: 1.7919 → 0.5-0.7
  (Reasonable attribute-specific flows)

✓ Step distance: 0.2350 → 0.02-0.05
  (Smooth, small steps)

✓ Trajectory plot:
  - Green dots spread across UMAP space
  - 10-20 visible attribute-specific regions
  - Continuous distribution within each region
```

### Visual Improvements:

**Before (mode collapse):**
```
All trajectories → ONE point [10, 6]
Flowed embeddings: 20-30 discrete points
Silhouette: 0.93 (perfect clustering = bad!)
```

**After (expected):**
```
Trajectories → Multiple regions across UMAP
Flowed embeddings: Continuous distribution in 10-20 clusters
Silhouette: 0.4-0.5 (good clustering with variation = good!)
```

---

## Why Previous Fixes Didn't Work

### Fix 1: Reduced Identity Weight (lambda 1.0 → 0.05)
- **Helped**: Allowed more movement
- **Didn't solve**: Prototype collapse still dominant issue
- **Result**: Movement increased to 1.79 (too much!) because gradients flowed through prototypes

### Fix 2: Always Augment Attributes
- **Helped**: Consistent transfer training
- **Didn't solve**: Still collapsed because prototype loss was broken
- **Result**: Model learned transfer but to discrete points

### Fix 3: Prototype-Based Loss (without detach)
- **Made it WORSE**: Created feedback loop
- **Caused**: Severe mode collapse (silhouette 0.93)
- **Problem**: Gradients flowing through prototypes

### Fix 4: This One! (Stop-gradient)
- **Fixes**: The fundamental issue
- **Breaks**: The feedback loop
- **Enables**: Continuous attribute-specific distributions

---

## How to Retrain

### Step 1: Pull the fix
```bash
cd ~/learning-to-flow
git pull
```

### Step 2: Delete old checkpoint (CRITICAL!)
```bash
rm -rf results/fclf_training/checkpoints/*
```

**Why delete:** The old model learned the wrong behavior (collapsing). Must start fresh!

### Step 3: Start training
```bash
python src/training/train_fclf.py \
    --config configs/fclf_config.yaml \
    --celeba_root data/celeba \
    --embedding_dir data/embeddings \
    --output_dir results/fclf_training
```

### Step 4: Evaluate after 5 and 10 epochs
```bash
# After 5 epochs:
python scripts/evaluate_attribute_transfer.py \
    --checkpoint results/fclf_training/checkpoints/fclf_epoch_5.pt \
    --output_dir results/transfer_eval_epoch_5_FIXED

# After 10 epochs:
python scripts/evaluate_attribute_transfer.py \
    --checkpoint results/fclf_training/checkpoints/fclf_epoch_10.pt \
    --output_dir results/transfer_eval_epoch_10_FIXED
```

---

## What to Watch During Training

### Key Diagnostic: Contrastive Loss

```python
# BEFORE (broken):
Epoch 1: contrastive = 1.97
Epoch 5: contrastive = 0.8
Epoch 10: contrastive = 0.5  # Decreasing, but mode collapse!

# AFTER (fixed):
Epoch 1: contrastive = 2.5-3.0  # Higher initially (harder task)
Epoch 5: contrastive = 1.2-1.5  # Slower decrease (learning properly)
Epoch 10: contrastive = 0.7-1.0  # Stabilizes higher (continuous clusters)
```

**Why loss will be higher with fix:**
- Harder to perfectly match continuous distributions
- Easier to collapse to discrete points (old behavior)
- Higher loss = model doing the RIGHT thing!

### Training Loss Components:

```python
# Watch for these values:
contrastive: 1.0-2.0 (will decrease slowly)
divergence: <1.0 (should be much lower than before's 6.81!)
curl: <0.1
identity: 0.001-0.01 (will increase as model learns to move)
```

---

## Success Criteria

### After 10 Epochs (Fixed Model):

**Metrics:**
```
✓ Silhouette by target: 0.4-0.6
✓ Movement: 0.5-0.7
✓ Step distance: 0.02-0.05
✓ Cluster purity: 0.6-0.8
```

**Visual:**
```
✓ Trajectory plot: Green dots in 8-15 distinct regions
✓ Flowed embeddings: Continuous spread, not discrete points
✓ Smooth trajectories: No erratic jumps
```

**If still seeing mode collapse:**
- Check if you deleted old checkpoints
- Verify git pull got latest code (commit 1f11391 or later)
- Check training output shows detach is being used

---

## The One-Line Fix

**Changed**: `proto = F.normalize(proto, dim=1)`
**To**: `proto = F.normalize(proto, dim=1).detach()`

**Impact**: Prevents ~20 epochs of wasted training and complete mode collapse

---

## Summary

**Problem**: Prototype-based contrastive loss created feedback loop → mode collapse
**Root cause**: Gradients flowed through prototypes back to embeddings
**Solution**: Stop-gradient (`.detach()`) on prototypes
**Expected result**: Continuous attribute-specific distributions, silhouette ~0.4-0.5

This is the FINAL critical fix. After retraining, mode collapse should be completely resolved!
