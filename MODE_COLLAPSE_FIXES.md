# Mode Collapse Fixes - Complete Guide

## Problem Summary

After 1 epoch of training with normalization fixes, the model showed:
- ✅ Good movement constraint (0.24, within bounds)
- ✅ Smooth trajectories (step distance 0.024)
- ❌ Severe mode collapse (all embeddings flow to same point [21,21])
- ❌ Silhouette by target = 1.0 (perfect clustering = collapsed)

**Root cause**: The model learned to move smoothly but NOT to different attribute-specific destinations.

---

## Root Cause Analysis

### 1. **Identity Loss Too Strong** (PRIMARY ISSUE)
- **Location**: `configs/fclf_config.yaml:16`
- **Problem**: `lambda_identity: 1.0` was TOO HIGH
- **Effect**: Penalized ANY movement away from original, preventing attribute-specific flows
- **Why it matters**: The contrastive loss pulls toward attribute clusters, but identity loss anchors embeddings to their starting point. When identity is stronger, model learns "don't move much" instead of "move to the right attribute-specific location"

### 2. **Weak Contrastive Loss**
- **Location**: `src/losses/contrastive_flow_loss.py:107-168`
- **Problem**: Old `AttributeContrastiveLoss` only compared pairwise similarities using MSE
- **Effect**: No explicit target directions for different attribute combinations
- **Why it matters**: Model had no clear signal about WHERE to move embeddings with different attributes

### 3. **Inconsistent Attribute Augmentation**
- **Location**: `src/training/train_fclf.py:223`
- **Problem**: Only 50% of batches did attribute flipping
- **Effect**: Half the training data reinforced identity preservation, not transfer
- **Why it matters**: Model never learned consistent attribute transfer behavior

### 4. **No Prototype Guidance**
- **Problem**: Loss didn't create explicit attribute-specific "target zones"
- **Effect**: Model could satisfy loss by moving everything to one point
- **Why it matters**: Without explicit targets, mode collapse is a valid solution

---

## Applied Fixes

### **FIX 1: Reduce Identity Loss Weight** ⭐ CRITICAL

**File**: `configs/fclf_config.yaml`

**Change**:
```yaml
# BEFORE
lambda_identity: 1.0  # Too strong!

# AFTER
lambda_identity: 0.05  # Reduced by 20x
```

**Rationale**:
- Identity loss prevents unbounded movement (which normalization already handles)
- But strong identity prevents attribute-specific movement
- Reduced to 0.05 to allow diverse flows while maintaining smoothness

**Expected effect**:
- Model can now learn different flow directions for different attributes
- Movement will increase slightly (0.3-0.6 range) but stay bounded
- Green dots will start spreading instead of clustering at one point

---

### **FIX 2: Always Apply Attribute Augmentation** ⭐ CRITICAL

**File**: `src/training/train_fclf.py:220-228`

**Change**:
```python
# BEFORE
target_attributes = attributes.clone()
if torch.rand(1).item() > 0.5:  # Only 50% of time!
    batch_size = attributes.size(0)
    for i in range(batch_size):
        num_flips = torch.randint(1, 3, (1,)).item()
        attrs_to_flip = torch.randperm(5)[:num_flips]
        for attr_idx in attrs_to_flip:
            target_attributes[i, attr_idx] = 1 - target_attributes[i, attr_idx]

# AFTER
target_attributes = attributes.clone()
batch_size = attributes.size(0)
for i in range(batch_size):  # ALWAYS flip!
    num_flips = torch.randint(1, 3, (1,)).item()
    attrs_to_flip = torch.randperm(5)[:num_flips]
    for attr_idx in attrs_to_flip:
        target_attributes[i, attr_idx] = 1 - target_attributes[i, attr_idx]
```

**Rationale**:
- Model must learn attribute TRANSFER, not just attribute clustering
- Every training sample should teach the model to change attributes
- Consistent training signal prevents mode collapse

**Expected effect**:
- Model learns that its job is to TRANSFORM attributes, not preserve them
- Better attribute-specific flow directions

---

### **FIX 3: Prototype-Based Contrastive Loss** ⭐⭐ MAJOR IMPROVEMENT

**File**: `src/losses/contrastive_flow_loss.py:107-226`

**Change**: Complete rewrite of `AttributeContrastiveLoss`

**New approach**:

```python
# OLD: Just compared pairwise similarities
loss = MSE(embedding_similarity, attribute_similarity)
# Problem: No explicit targets, weak signal

# NEW: Creates explicit attribute prototypes
1. Compute prototype = mean(embeddings with same attributes)
2. Pull each embedding toward its target attribute prototype
3. Push embeddings away from other attribute prototypes
4. Add pairwise contrastive term for fine-grained clustering

# InfoNCE-style loss:
loss = -log(sim_to_correct_prototype / sum_of_all_prototypes)
```

**Key improvements**:

1. **Explicit targets**: Each attribute combination gets a prototype (centroid)
2. **Clear separation**: Model learns to separate different attribute combinations
3. **Prevents mode collapse**: Loss is minimized when embeddings form distinct attribute-specific clusters

**Expected effect**:
- Green dots will spread into 10-20 distinct regions (one per attribute combo)
- Silhouette by target will drop from 1.0 to 0.35-0.55 (good clustering without collapse)
- Each region contains embeddings with same attributes but different identities

---

## How These Fixes Work Together

### The Problem: Conflicting Objectives

**Old setup**:
```
Contrastive loss: "Cluster by attributes" (weak signal)
Identity loss (λ=1.0): "Don't move!" (STRONG signal)
Result: Model learns "move slightly to one nearby point" (mode collapse)
```

**New setup**:
```
Prototype contrastive: "Move to attribute-specific prototype" (STRONG signal)
Identity loss (λ=0.05): "Don't move TOO far" (weak constraint)
Result: Model learns "move to the right attribute-specific location"
```

### Training Progression (Expected)

**Epoch 1-3**:
- Model learns to respect normalization constraint
- Green dots start spreading from single point to 2-3 clusters

**Epoch 4-7**:
- Prototypes stabilize, creating clear attribute-specific targets
- Green dots spread into 5-10 visible regions
- Silhouette by target drops to ~0.7

**Epoch 8-15**:
- Fine-grained attribute combinations separate
- Green dots in 10-15 distinct regions
- Silhouette by target reaches 0.4-0.6 (optimal)

**Epoch 15-50**:
- Refinement: smoother flows, better separation
- Continuous distribution within each attribute cluster

---

## Expected Metrics After Retraining

### After 5 Epochs (Early)
```
Movement:                     0.3 - 0.5
Silhouette by original:       0.15 - 0.25 (good - left original clusters)
Silhouette by target:         0.6 - 0.8 (improving but still collapsing)
Step distance:                0.02 - 0.04
Cluster purity:               40% - 60%

Trajectory plot:
- Green dots in 3-5 visible clusters
- Some overlap but clear separation emerging
```

### After 15 Epochs (Mid)
```
Movement:                     0.4 - 0.6
Silhouette by original:       0.18 - 0.28
Silhouette by target:         0.4 - 0.6 ✓ OPTIMAL!
Step distance:                0.02 - 0.04
Cluster purity:               65% - 80%

Trajectory plot:
- Green dots in 8-12 distinct regions
- Clear attribute-specific clustering
- Continuous within each cluster
```

### After 30+ Epochs (Converged)
```
Movement:                     0.5 - 0.7
Silhouette by original:       0.20 - 0.30
Silhouette by target:         0.35 - 0.55 ✓ IDEAL!
Step distance:                0.02 - 0.04
Cluster purity:               75% - 90%

Trajectory plot:
- Green dots spread across entire UMAP space
- 10-20 visible attribute-specific regions
- Smooth continuous distribution within regions
- No sklearn warnings!
```

**Key insight**: Silhouette by target should be 0.35-0.55, NOT 1.0!
- 1.0 = mode collapse (discrete points)
- 0.35-0.55 = good clustering with continuous variation (success!)
- <0.3 = too spread out, attributes not learned

---

## Retraining Instructions

### Step 1: Delete old checkpoint (IMPORTANT!)
```bash
# The old model learned bad behavior, must start fresh
rm -rf results/fclf_training/checkpoints/*
```

### Step 2: Start training
```bash
python scripts/train_fclf.sh
```

### Step 3: Monitor progress
```bash
# Evaluate every 5 epochs
python scripts/evaluate_attribute_transfer.py \
    --checkpoint results/fclf_training/checkpoints/fclf_latest.pt \
    --output_dir results/transfer_eval_latest

# Look for:
# - Movement: Should gradually increase from 0.24 → 0.5
# - Silhouette by target: Should decrease from 1.0 → 0.4-0.6
# - Green dots: Should spread from one point → multiple regions
```

### Step 4: When to stop
```
GOOD - Continue training:
- Silhouette by target > 0.6 (still collapsing)
- Green dots in <8 regions
- Sklearn warnings still appearing

SUCCESS - Training complete:
- Silhouette by target = 0.35-0.55
- Green dots in 10-20 regions
- No sklearn warnings
- Cluster purity >75%

OVERTRAINING - Stop and use earlier checkpoint:
- Silhouette by target <0.3
- Green dots completely spread out, no visible clusters
- This means model "forgot" attributes
```

---

## Advanced Fix (Optional - Try if Above Fixes Don't Work)

If after 20 epochs you still see mode collapse (silhouette > 0.7), try this:

### **FIX 4: Negative Sampling**

Add explicit repulsion between different attribute combinations:

**File**: `src/losses/combined_loss.py:107-113`

```python
# Add after line 113
# Explicit repulsion between different attribute combinations
if self.lambda_separation > 0:
    # Sample random attributes different from target
    random_attrs = attributes.clone()
    for i in range(attributes.size(0)):
        flip_idx = torch.randint(0, 5, (1,)).item()
        random_attrs[i, flip_idx] = 1 - random_attrs[i, flip_idx]

    # Compute flow toward random (wrong) attributes
    v_wrong = vector_field(z, random_attrs)
    z_wrong = torch.nn.functional.normalize(z + self.alpha * v_wrong, dim=1)

    # Repulsion loss: flowed embedding should be FAR from wrong attribute flow
    separation_loss = -torch.mean((z_flowed - z_wrong) ** 2)

    total_loss += self.lambda_separation * separation_loss
```

**Config**:
```yaml
loss:
  lambda_separation: 0.2
```

This adds explicit repulsion, but try the first 3 fixes first!

---

## Understanding the Fixes: Visual Analogy

### Old Model (Mode Collapse)
```
All embeddings → [21, 21]

Like a GPS that says "go to Times Square" for EVERY destination
- Smiling young woman? → Times Square
- Old bearded man?      → Times Square
- Teenager with glasses? → Times Square
```

### New Model (Correct Behavior)
```
Attribute combination 1 → Region A
Attribute combination 2 → Region B
...
Attribute combination 20 → Region T

Like a GPS that gives you the CORRECT destination:
- "Smiling young woman"  → SoHo (Region A)
- "Old bearded man"      → Upper East Side (Region B)
- "Teenager with glasses" → Brooklyn (Region C)

And within SoHo, each address is slightly different (identity preserved)
```

---

## Technical Details: Why Prototypes Work

### The Math

**Old loss**:
```python
similarity_matrix = z_flowed @ z_flowed.T
attribute_similarity = attributes @ attributes.T
loss = MSE(similarity_matrix, attribute_similarity)
```

Problem: This only says "similar attributes → similar embeddings"
It doesn't create DISTINCT clusters, just relative relationships.

**New loss**:
```python
# Compute prototype for each attribute combination
prototypes[attr_combo] = mean(z_flowed[attr == attr_combo])

# Pull each embedding to its prototype, push away from others
loss = -log(
    exp(sim(z_flowed, correct_prototype)) /
    sum(exp(sim(z_flowed, all_prototypes)))
)
```

Advantage: Creates explicit targets that:
1. Pull embeddings toward attribute-specific locations
2. Push embeddings away from wrong attributes
3. Allow continuous variation within each attribute cluster

### Why Identity Loss Interfered

```python
# Training objective becomes:
total_loss = contrastive_loss + 1.0 * identity_loss

# If identity_loss is strong:
total_loss ≈ 1.0 * ||z_flowed - z||²

# Model learns: "minimize movement"
# Result: Flow to nearest safe point (mode collapse)

# With reduced identity loss:
total_loss ≈ contrastive_loss + 0.05 * ||z_flowed - z||²

# Model learns: "go to correct attribute cluster, but don't go crazy"
# Result: Attribute-specific flows with bounded movement
```

---

## Debugging: What to Check If Still Collapsing

### After 10 epochs, if silhouette by target > 0.7:

1. **Check contrastive loss value**:
```bash
tensorboard --logdir results/fclf_training/logs
```
Look for `Train/Contrastive` - should be decreasing from ~2.0 → ~0.5

2. **Check identity loss isn't dominating**:
`Train/Identity` should be similar magnitude to `Train/Contrastive`
If `Identity` >> `Contrastive`, reduce `lambda_identity` further (try 0.01)

3. **Check number of prototypes per batch**:
Add debug print in contrastive_flow_loss.py:154
```python
print(f"Num prototypes this batch: {num_prototypes}")
```
Should see 30-60 prototypes per batch (128 batch size, 5 attributes = ~20-40 combos)

4. **Visualize prototype distribution**:
After 5 epochs, run:
```python
# In evaluation script, add:
print("Unique attribute combinations:", unique_attrs.size(0))
```
Should see 15-25 unique combinations in test set

5. **Check for NaN/exploding gradients**:
If loss becomes NaN, reduce learning rate:
```yaml
training:
  learning_rate: 0.00005  # Reduce from 0.0001
```

---

## Summary: The Three Critical Changes

1. **Reduce identity loss**: `lambda_identity: 1.0 → 0.05`
   - Allows attribute-specific movement

2. **Always augment attributes**: Remove `if torch.rand(1).item() > 0.5:`
   - Consistent transfer training

3. **Prototype-based contrastive loss**: Explicit attribute-specific targets
   - Creates distinct clusters, prevents mode collapse

**Expected result**: After 15 epochs, green dots spread into 10+ regions with silhouette ~0.4-0.6

Good luck with retraining! The mode collapse should be completely resolved.
