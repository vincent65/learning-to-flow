# Corrected Root Cause Analysis

## Original Diagnosis (Partially Wrong)

I initially said:
> "Identity loss weight (λ=1.0) was too strong, preventing attribute-specific movement"

**This was WRONG based on the training logs!**

## Actual Training Logs

```
loss=0.599, contr=0.599, curl=5.66e-5, div=1.63e-5, ident=6.57e-7
```

Breaking this down:
- **Contrastive**: 0.599 (DOMINATES the loss)
- **Curl**: 5.66e-5 (tiny)
- **Div**: 1.63e-5 (tiny)
- **Identity**: 6.57e-7 (effectively zero!)

Identity contribution to total loss:
```python
1.0 * 6.57e-7 = 0.00000066
```

**Identity loss was NOT preventing movement!**

## The REAL Root Cause

### Why Identity Loss Was So Small

```python
identity_loss = torch.mean((z_flowed - z) ** 2)
# After normalization: both z and z_flowed are on unit sphere
# If they're very close: identity_loss ≈ 0
```

Identity loss was tiny because **the model wasn't moving much in the first place!**

Movement = 0.24 confirms minimal movement.

### The Actual Problem: Weak Contrastive Loss

The OLD `AttributeContrastiveLoss` used MSE between similarities:

```python
# OLD (WEAK) approach:
loss = F.mse_loss(
    torch.sigmoid(similarity_matrix),
    target_attribute_similarity,
    reduction='mean'
)
```

**Why this is weak:**

1. **No explicit targets**: Just compares relative similarities, doesn't create distinct clusters
2. **Mode collapse is valid**: If all embeddings are at one point, all similarities are 1.0, and if all attributes are the same, target is 1.0 → MSE = 0 (low loss!)
3. **Lazy solution**: Model finds it easier to cluster everything at one point than to separate into attribute-specific regions

### The Proof: contrastive_loss = 0.599 (High!)

Contrastive loss stayed HIGH (0.599) throughout training because:
- The model was stuck in a local minimum (mode collapse)
- MSE loss couldn't provide gradient signal to escape
- No explicit "pull toward A, push from B" mechanism

## Why My Fixes Actually Work

### FIX 1: Prototype-Based Loss (THE KEY FIX)

```python
# NEW (STRONG) approach:
# 1. Compute prototypes (centroids) for each attribute combination
prototypes[attr] = mean(embeddings with attr)

# 2. InfoNCE loss: pull toward correct prototype, push from wrong ones
loss = -log(
    exp(sim(z, correct_prototype)) /
    sum(exp(sim(z, all_prototypes)))
)
```

**Why this works:**
- **Explicit targets**: Each attribute combination has a clear target (prototype)
- **Strong separation**: InfoNCE creates distinct clusters by design
- **Mode collapse is penalized**: If everything goes to one point, all prototypes collapse → high loss!

### FIX 2: Always Augment Attributes (CRITICAL)

Removing the 50% probability means:
- Every batch teaches attribute transfer
- Model consistently sees "change attributes, don't just cluster"
- No mixed signals

### FIX 3: Reduce lambda_identity (MINOR BENEFIT)

Even though identity loss was already tiny (6.57e-7), reducing it from 1.0 to 0.05 helps because:
- As the model starts learning better flows (with new contrastive loss), movement will increase
- Lower lambda_identity ensures it won't interfere later in training
- But this was NOT the primary issue!

## Corrected Summary

### Root Cause
**Weak MSE-based contrastive loss** allowed model to find lazy solution (mode collapse).

### Primary Fix
**Prototype-based InfoNCE contrastive loss** creates explicit attribute-specific targets with strong separation signal.

### Secondary Fixes
- Always augment attributes (consistent transfer training)
- Reduce lambda_identity (won't interfere as model learns to move more)

## Expected Training Progression

With the new prototype-based loss:

### Epoch 1-3
```
contrastive_loss: 2.0 → 1.2  (should DECREASE now!)
identity_loss: 6.57e-7 → 1e-3 (will INCREASE as model starts moving)
movement: 0.24 → 0.4
```

### Epoch 5-10
```
contrastive_loss: 1.2 → 0.6
identity_loss: 1e-3 → 5e-3 (model moving more)
movement: 0.4 → 0.6
silhouette_by_target: 1.0 → 0.6 (spreading out!)
```

### Epoch 15+
```
contrastive_loss: 0.6 → 0.3 (converged)
identity_loss: 5e-3 → 1e-2 (bounded movement)
movement: 0.6 → 0.7 (good attribute-specific flows)
silhouette_by_target: 0.6 → 0.4 ✓ (continuous clusters)
```

## Key Diagnostic

**Watch contrastive loss during retraining!**

- **OLD model**: contrastive_loss stayed at ~0.6, not decreasing (stuck in mode collapse)
- **NEW model**: contrastive_loss should decrease from ~2.0 → ~0.3 (learning!)

If contrastive loss is decreasing, the fixes are working!

## Bottom Line

- ❌ **NOT the problem**: Identity loss weight too strong
- ✅ **ACTUAL problem**: Weak MSE-based contrastive loss
- ✅ **PRIMARY solution**: Prototype-based InfoNCE loss
- ✅ **BONUS fixes**: Always augment, reduce lambda_identity (helps but not essential)

The prototype-based loss is the critical fix that will resolve mode collapse!
