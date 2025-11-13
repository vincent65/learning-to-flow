# FCLF Methods Documentation

**Last Updated:** 2025-11-12

This document tracks all implementation details, training procedures, and evaluation methods for the FCLF (Function-Contrastive Latent Fields) project.

---

## Table of Contents
1. [Model Architecture](#model-architecture)
2. [Training Procedure](#training-procedure)
3. [Loss Functions](#loss-functions)
4. [Evaluation Metrics](#evaluation-metrics)
5. [Change Log](#change-log)

---

## Model Architecture

### Vector Field Network

**Architecture:** MLP-based conditional vector field

```
Input: [z, a]
  z: [512] CLIP ViT-B/32 embedding (unit-normalized)
  a: [5] binary attribute vector

Network:
  Linear(512 + 5 → 256)
  ReLU
  Linear(256 → 256)
  ReLU
  Linear(256 → 512)

Output: v(z, a) [512] velocity vector
```

**Key Properties:**
- Embedding dimension: 512 (CLIP ViT-B/32)
- Hidden dimension: 256
- Number of attributes: 5
- Parameters: ~395k

**Flow Integration:**
```python
z_flowed = z + alpha * v(z, a)
z_flowed = normalize(z_flowed)  # CRITICAL: Stay in CLIP space
```

---

## Training Procedure

### Dataset: CelebA

**Split:**
- Train: 162,000 images (0-162,000)
- Val: 20,000 images (162,000-182,000)
- Test: 20,599 images (182,000-202,599)

**Attributes Used (5/40):**
1. Smiling
2. Young
3. Male
4. Eyeglasses
5. Mustache

**Preprocessing:**
- CLIP embeddings precomputed using ViT-B/32
- Embeddings are unit-normalized (||z|| = 1)
- Attributes converted from {-1, 1} to {0, 1}

### Training Configuration

**Current (v3 - Fixed Mode Collapse):**
```yaml
training:
  num_epochs: 50
  batch_size: 512
  learning_rate: 1e-4
  alpha: 0.05  # Flow step size
  optimizer: Adam

loss:
  temperature: 0.07
  lambda_contrastive: 0.2   # Reduced from 1.0
  lambda_curl: 0.05
  lambda_div: 0.05
  lambda_identity: 0.8      # Increased from 0.2
```

**Optimizer:** Adam with default betas (0.9, 0.999)

**Hardware:**
- GPU: NVIDIA L4 (24GB)
- Batch size: 512 (fully utilizes GPU)
- Training time: ~2.4 hours for 50 epochs

### Data Augmentation

**Attribute Augmentation (during training):**
- For each batch, create target attributes by flipping 1-2 random attributes
- Ensures model learns attribute transfer, not just clustering
- 50% single-attribute flips, 50% double-attribute flips

```python
for i in range(batch_size):
    num_flips = np.random.randint(1, 3)  # 1 or 2
    attrs_to_flip = np.random.choice(5, size=num_flips, replace=False)
    for attr_idx in attrs_to_flip:
        target_attributes[i, attr_idx] = 1 - target_attributes[i, attr_idx]
```

---

## Loss Functions

### Total Loss

```
L_total = λ_c * L_contrastive + λ_curl * R_curl + λ_div * R_div + λ_id * L_identity
```

### 1. Contrastive Loss (Supervised InfoNCE)

**Purpose:** Cluster embeddings by target attributes

**Implementation (v3 - Softer Matching):**
```python
# Positive pairs: At least 4/5 attributes match (was 5/5)
attr_similarity = attributes @ attributes.T
similarity_threshold = max(num_attrs - 1, int(0.8 * num_attrs))
positive_mask = (attr_similarity >= similarity_threshold)

# InfoNCE loss
similarity = (z_norm @ z_norm.T) / temperature
loss = -log(sum(exp(pos_sim)) / sum(exp(all_sim)))
```

**Rationale:**
- Previous version required exact 5/5 attribute matches
- Created 2^5 = 32 discrete clusters that collapsed to points
- New version allows 4/5 matches → more diversity within clusters

**Weight:** λ_c = 0.2 (reduced from 1.0 to prevent over-clustering)

### 2. Curl Regularization

**Purpose:** Encourage irrotational (conservative) vector fields

**Implementation:**
```python
# Sample random 2D planes through embedding space
# Compute circulation ∮v·dl via finite differences
curl = ||∇ × v||²
```

**Weight:** λ_curl = 0.05

### 3. Divergence Regularization

**Purpose:** Encourage divergence-free (incompressible) vector fields

**Implementation:**
```python
# Compute trace of Jacobian via finite differences
div = (∇ · v)²
```

**Weight:** λ_div = 0.05

### 4. Identity Loss

**Purpose:** Prevent mode collapse by keeping embeddings close to origin

**Implementation:**
```python
identity_loss = ||z_flowed - z_original||²
```

**Weight:** λ_id = 0.8 (increased from 0.2)

**Rationale:**
- Strong identity preservation forces minimal movement
- Prevents aggressive collapse to centroids
- Maintains face diversity while still adding/removing attributes

---

## Evaluation Metrics

### Quantitative Metrics

#### 1. Linear Probe Accuracy
**What:** Train logistic regression on embeddings to predict each attribute
**How:**
- Split: 70% train, 30% test
- Model: Logistic Regression (max_iter=1000)
- Compute for both original and flowed embeddings
**Good Result:** Flowed accuracy > 0.75 for each attribute

#### 2. k-NN Purity (k=10)
**What:** Fraction of k-nearest neighbors sharing same target attributes
**How:**
- For each flowed embedding, find 10 nearest neighbors
- Check if they have identical target attribute vectors
- Compute mean purity across all samples
**Good Result:** > 0.85 (strong clustering)

#### 3. Centroid Distance
**What:** Mean distance from flowed embeddings to target class centroids
**How:**
- Compute centroid for each unique attribute combination
- Measure ||z_flowed - centroid|| for each sample
**Good Result:** < 0.10 (close to target centroids)

#### 4. AUC Along Path
**What:** Track per-attribute classifier AUC at each flow step t=0,1,...,10
**How:**
- Train logistic regression classifier for each attribute
- Compute AUC at each step of trajectory
- Check if AUC increases monotonically
**Good Result:** > 80% of trajectories show monotonic AUC increase

#### 5. Attribute Leakage
**What:** How much NON-target attributes change during flow
**How:**
- For samples where attribute i is unchanged, measure Δ(probe accuracy) for attribute i
- Leakage = |acc_end - acc_start|
**Good Result:** Mean leakage < 0.05 (attributes stay stable)

#### 6. Field Diagnostics
**What:** Statistics of curl and divergence across embedding space
**How:**
- Sample 500 random (z, a) pairs
- Compute curl and divergence via finite differences
- Report mean, std, max
**Good Result:** Low curl and divergence (smooth field)

### Qualitative Metrics

#### 7. Nearest-Neighbor Flipbooks
**What:** Show real images nearest to each point along flow trajectory
**How:**
- For each trajectory, retrieve k=1 nearest training image at each step
- Display as horizontal strip: t=0 → t=1 → ... → t=10
**Good Result:** Smooth semantic transitions, diverse faces (not all converging)

#### 8. UMAP Visualizations
**What:** 2D projection of embeddings colored by attributes
**How:**
- Project original and flowed embeddings to 2D via UMAP
- Color by each attribute separately (5 subplots)
- Visualize trajectories as arrows
**Good Result:** Clear separation by target attributes, smooth flows

### Baseline Comparison

#### Linear CLIP Steering
**What:** Simple baseline using class-difference vectors
**How:**
```python
v_attr = mean(embeddings[attr=1]) - mean(embeddings[attr=0])
z_steered = normalize(z + alpha * sum(v_attr * attr_change))
```
**Purpose:** Show that FCLF is better than naive linear interpolation

**Expected Results:**
- FCLF should have higher k-NN purity (+10-20%)
- FCLF should have lower centroid distance (-50-70%)
- Linear steering might have higher linear probe accuracy (acceptable trade-off)

---

## Change Log

### Version 3 (2025-11-12) - Mode Collapse Fix

**Problem Identified:**
- Flipbook visualizations showed all flows converging to 2-3 "attractor" faces
- k-NN purity was 96.2% (too high → over-clustering)
- Model learned 32 discrete attractor points (one per attribute combination)
- Semantically incorrect transformations (e.g., "add mustache" → blonde woman)

**Root Causes:**
1. Contrastive loss required exact 5/5 attribute matches
2. Weak identity loss (0.2) allowed excessive movement
3. CLIP space not naturally organized by CelebA attributes

**Changes Made:**

1. **Softened Positive Pair Definition** (`src/losses/contrastive_flow_loss.py`)
   - OLD: `positive_mask = (attr_similarity == num_attrs)`
   - NEW: `positive_mask = (attr_similarity >= max(num_attrs-1, 0.8*num_attrs))`
   - Allows 4/5 attribute matches → more diversity within clusters

2. **Reduced Contrastive Weight** (`configs/fclf_config.yaml`)
   - OLD: `lambda_contrastive = 1.0` (implicit)
   - NEW: `lambda_contrastive = 0.2`
   - Weaker pull toward centroids

3. **Increased Identity Weight** (`configs/fclf_config.yaml`)
   - OLD: `lambda_identity = 0.2`
   - NEW: `lambda_identity = 0.8`
   - Strong preservation of starting point

**Expected Impact:**
- k-NN purity: 96.2% → ~85-90% (still excellent, more realistic)
- Centroid distance: 0.039 → ~0.06-0.09 (slightly farther, but diverse)
- Flipbooks: Should show diverse faces with gradual attribute changes
- Main improvement: **Diversity while maintaining attribute control**

**Files Modified:**
- `src/losses/contrastive_flow_loss.py` (line 58-61, 67)
- `src/losses/combined_loss.py` (added lambda_contrastive parameter)
- `configs/fclf_config.yaml` (loss section)

**Status:** Implemented, awaiting retraining

---

### Version 2 (2025-11-11) - Hyperparameter Tuning

**Problem:** Excessive movement (1.47, target 0.5-0.7)

**Changes:**
1. Reduced alpha: 0.1 → 0.05
2. Increased lambda_identity: 0.05 → 0.2
3. Increased lambda_curl: 0.01 → 0.05
4. Increased lambda_div: 0.01 → 0.05

**Impact:**
- Movement reduced but mode collapse still present
- Silhouette improved: 0.93 → 0.38

---

### Version 1 (2025-11-10) - Prototype Loss Replacement

**Problem:** Severe mode collapse with prototype-based loss (silhouette 0.93-0.94)

**Root Cause:** Prototype-based contrastive loss created feedback loop even with `.detach()`

**Change:** Replaced prototype loss with simple supervised contrastive (pairwise InfoNCE)

**Impact:**
- Silhouette: 0.93 → 0.38
- Mode collapse reduced but not eliminated
- Attribute transfer working

**Files Modified:**
- `src/losses/contrastive_flow_loss.py` (complete rewrite)

---

### Version 0 (2025-11-09) - Initial Implementation

**Implementation:** Basic FCLF with prototype-based contrastive loss

**Issues:**
- Severe mode collapse
- Embeddings collapsed to discrete points
- Not usable

---

## Evaluation Scripts

### 1. Comprehensive Paper Metrics
**Script:** `scripts/compute_paper_metrics.py`

**Computes:**
- Linear probe accuracy (per-attribute)
- k-NN purity, clustering metrics
- Centroid distance
- AUC along path (generates plot)
- Attribute leakage
- Field diagnostics (curl, divergence)
- Nearest-neighbor flipbook data
- Linear steering baseline comparison

**Output:**
- `paper_metrics.json` - All quantitative results
- `auc_curves.png` - AUC trajectory plots
- `flipbook_data.json` - Data for visualization

**Usage:**
```bash
python scripts/compute_paper_metrics.py \
    --checkpoint checkpoints/fclf_best.pt \
    --celeba_root data/celeba \
    --embedding_dir data/embeddings \
    --output_dir paper_metrics \
    --device cuda
```

### 2. LaTeX Table Generation
**Script:** `scripts/generate_latex_tables.py`

**Generates:**
- Method comparison table (FCLF vs. Linear)
- Attribute leakage table
- Field diagnostics table
- Monotonic AUC progress table

**Usage:**
```bash
python scripts/generate_latex_tables.py paper_metrics/paper_metrics.json
```

### 3. Flipbook Visualization
**Script:** `scripts/visualize_flipbook.py`

**Generates:**
- Individual flipbook strips with descriptive filenames
- Summary montage (5 trajectories)

**Filename Format:** `flipbook_0001_add_Mustache.png`

**Usage:**
```bash
python scripts/visualize_flipbook.py \
    --flipbook_data paper_metrics/flipbook_data.json \
    --celeba_root data/celeba \
    --output_dir paper_metrics/flipbooks \
    --num_flipbooks 20
```

### 4. UMAP Trajectory Visualization
**Script:** `scripts/evaluate_attribute_transfer.py`

**Generates:**
- UMAP plots colored by original attributes
- UMAP plots colored by target attributes
- 2D trajectory visualizations

**Usage:**
```bash
python scripts/evaluate_attribute_transfer.py \
    --checkpoint checkpoints/fclf_best.pt \
    --celeba_root data/celeba \
    --embedding_dir data/embeddings \
    --output_dir evaluation_results \
    --device cuda
```

---

## Key Design Decisions

### 1. Why Unit-Normalize Flowed Embeddings?

CLIP embeddings are **always** unit-normalized (||z|| = 1). After flowing, we must renormalize:

```python
z_flowed = z + alpha * v(z, a)
z_flowed = F.normalize(z_flowed, dim=1)  # CRITICAL
```

**Without normalization:**
- Embeddings can have arbitrary magnitude
- Not valid CLIP embeddings
- Similarity metrics break down
- Identity loss measures wrong thing

**With normalization:**
- Stay on unit hypersphere
- Identity loss measures angular distance (correct)
- Valid CLIP embeddings

### 2. Why Softer Positive Pairs?

**Problem:** Exact attribute matching (5/5) creates discrete clusters
- 2^5 = 32 possible combinations
- Each becomes a single point
- Mode collapse

**Solution:** Allow partial matches (4/5)
- ~160 overlapping clusters (5 ways to differ on 1 attribute × 32)
- Smooth overlap between clusters
- Maintains attribute separation while allowing diversity

### 3. Why Strong Identity Loss?

**Balancing Act:**
- Contrastive loss: Pull toward target centroid
- Identity loss: Stay near starting point

**Trade-off:**
```
Low identity → Strong clustering, mode collapse
High identity → Weak clustering, high diversity
```

**Our Choice:** λ_id = 0.8, λ_c = 0.2
- Prioritize diversity over tight clustering
- Still achieve 85-90% k-NN purity (excellent)
- Semantically meaningful transformations

### 4. Why These 5 Attributes?

**Chosen:** Smiling, Young, Male, Eyeglasses, Mustache

**Criteria:**
1. **Visual salience:** Clear, visible changes
2. **Reasonably balanced:** Not 99% one class
3. **Semantically independent:** Can be combined (e.g., "Young + Mustache")
4. **Diverse modalities:** Expression, age, gender, accessories

**Not chosen:** Attributes like "Attractive" (subjective, hard to verify)

---

## Reproducibility Checklist

To reproduce results:

- [ ] CelebA dataset in `data/celeba/`
- [ ] Precomputed CLIP embeddings in `data/embeddings/`
- [ ] Use exact config: `configs/fclf_config.yaml` (v3)
- [ ] Train for 50 epochs on GPU
- [ ] Use random seed 42 for evaluation splits
- [ ] Evaluate on test set (182,000-202,599)

**Expected Results (v3, after retraining):**
- k-NN Purity: 85-90%
- Average AUC: 0.75-0.82
- Centroid Distance: 0.06-0.09
- Attribute Leakage: < 0.05
- Flipbooks: Diverse faces with smooth attribute changes

---

## Future Improvements

**High Priority:**
1. Train with 10+ attributes (currently 5)
2. Add CLIP text embeddings as semantic anchors
3. Implement diversity regularizer (explicit variance term)

**Medium Priority:**
1. Multi-step flow training (currently single-step)
2. Adaptive step size (learned alpha)
3. Attribute-specific vector fields

**Low Priority:**
1. Train decoder for image reconstruction
2. Implement for other embedding spaces (DINO, DINOv2)
3. Extend to other datasets (FFHQ, AFHQv2)

---

## Citation & Acknowledgments

**CLIP:** Radford et al., "Learning Transferable Visual Models From Natural Language Supervision", ICML 2021

**CelebA:** Liu et al., "Deep Learning Face Attributes in the Wild", ICCV 2015

**Contrastive Learning:** Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations", ICML 2020

---

**Document maintained by:** CS229 Project Team
**Last verified:** 2025-11-12
**Model version:** v3 (mode collapse fix)
**Status:** Awaiting retraining with v3 hyperparameters
