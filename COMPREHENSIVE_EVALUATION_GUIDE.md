# Comprehensive FCLF Evaluation Guide

This guide implements all the quantitative metrics and visualizations suggested for a rigorous evaluation.

## Available CelebA Attributes

CelebA has 40 binary attributes. Currently, your model uses 5:
- `Smiling` (index 31)
- `Young` (index 39)
- `Male` (index 20)
- `Eyeglasses` (index 15)
- `Mustache` (index 22)

**Fun attribute pairs for visualization:**
- Smiling ↔ Not Smiling
- Young ↔ Old
- Male ↔ Female
- Eyeglasses ↔ No Glasses
- Mustache ↔ No Mustache
- Bald ↔ Hair
- Heavy_Makeup ↔ No Makeup
- Wearing_Hat ↔ No Hat

---

## 1. Separability & Clustering Metrics

### A. Linear Probe Accuracy

**What it measures:** How well a linear classifier can predict attributes from embeddings.

**Expected behavior:**
- ✅ Target attribute accuracy ↑ after flow
- ✅ Original attribute accuracy ↓ after flow (disentanglement)
- ✅ Non-target attributes unchanged (selectivity)

**Implementation:**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Train on original embeddings
clf_orig = LogisticRegression(max_iter=500)
clf_orig.fit(original_emb_train, target_attrs_train[:, attr_idx])
acc_orig = clf_orig.score(original_emb_test, target_attrs_test[:, attr_idx])

# Train on flowed embeddings
clf_flow = LogisticRegression(max_iter=500)
clf_flow.fit(flowed_emb_train, target_attrs_train[:, attr_idx])
acc_flow = clf_flow.score(flowed_emb_test, target_attrs_test[:, attr_idx])

print(f"Target attribute accuracy: {acc_orig:.3f} → {acc_flow:.3f}")
# Should increase!
```

### B. Clustering Quality

**Metrics:**
1. **Silhouette Score** [-1, 1]: Higher = better separated clusters
2. **Calinski-Harabasz**: Higher = better defined clusters
3. **Davies-Bouldin**: Lower = better separation

**Expected behavior:**
```
By target attributes:
  Original: Low scores (random w.r.t. target)
  Flowed:   High scores (clustered by target)

By original attributes:
  Original: High scores (CLIP clusters by attributes)
  Flowed:   Low scores (moved away from original)
```

**Implementation:**
```python
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Convert attributes to cluster labels
def attrs_to_labels(attrs):
    # Treats each unique attribute combination as a cluster
    return [''.join(map(str, row.astype(int))) for row in attrs]

orig_labels = attrs_to_labels(original_attrs)
target_labels = attrs_to_labels(target_attrs)

# Metrics for original embeddings by target attrs (should be random/low)
sil_orig_target = silhouette_score(original_emb, target_labels)

# Metrics for flowed embeddings by target attrs (should be high)
sil_flow_target = silhouette_score(flowed_emb, target_labels)

print(f"Silhouette by target: {sil_orig_target:.3f} → {sil_flow_target:.3f}")
# Should increase!
```

### C. k-NN Class Purity

**What it measures:** For each point, what fraction of k nearest neighbors share the same attribute?

**Expected:**
- Original embeddings by target: ~50% (random)
- Flowed embeddings by target: >80% (well-clustered)

**Implementation:**
```python
from sklearn.neighbors import KNeighborsClassifier

def knn_purity(embeddings, labels, k=10):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(embeddings, labels)

    purities = []
    for i in range(len(embeddings)):
        neighbors = knn.kneighbors([embeddings[i]], return_distance=False)[0]
        purity = (labels[neighbors] == labels[i]).mean()
        purities.append(purity)

    return np.mean(purities)

purity_orig = knn_purity(original_emb, target_labels, k=10)
purity_flow = knn_purity(flowed_emb, target_labels, k=10)

print(f"k-NN purity (k=10): {purity_orig:.3f} → {purity_flow:.3f}")
# Should increase!
```

### D. Distance to Class Centroids

**What it measures:** Mean distance from each point to its class centroid.

**Expected:** Decreases after flow (tighter clusters).

**Implementation:**
```python
def centroid_distance(embeddings, labels):
    unique_labels = np.unique(labels)
    centroids = {lab: embeddings[labels == lab].mean(axis=0) for lab in unique_labels}

    distances = [np.linalg.norm(embeddings[i] - centroids[labels[i]])
                 for i in range(len(embeddings))]
    return np.mean(distances)

dist_orig = centroid_distance(original_emb, target_labels)
dist_flow = centroid_distance(flowed_emb, target_labels)

print(f"Mean distance to centroid: {dist_orig:.3f} → {dist_flow:.3f}")
# Should decrease!
```

---

## 2. Path Quality Metrics

### A. Monotonic Progress

**What it measures:** Does distance to target centroid decrease at every step?

**Expected:** >70% of trajectories should show monotonic decrease.

**Implementation:**
```python
def monotonic_progress(trajectories, target_attrs):
    # Compute centroids for each attribute combination
    unique_attrs = np.unique(target_attrs, axis=0)
    centroids = {}
    for attr_vec in unique_attrs:
        mask = np.all(target_attrs == attr_vec, axis=1)
        centroids[tuple(attr_vec)] = trajectories[mask, -1, :].mean(axis=0)

    monotonic_count = 0
    for i in range(len(trajectories)):
        centroid = centroids[tuple(target_attrs[i])]

        # Distance at each step
        distances = [np.linalg.norm(trajectories[i, step] - centroid)
                    for step in range(trajectories.shape[1])]

        # Check monotonic decrease
        is_monotonic = all(distances[j] >= distances[j+1] for j in range(len(distances)-1))
        if is_monotonic:
            monotonic_count += 1

    return monotonic_count / len(trajectories)

frac_monotonic = monotonic_progress(trajectories, target_attrs)
print(f"Fraction with monotonic progress: {frac_monotonic:.1%}")
# Should be >70%
```

### B. Path Smoothness

**Metrics:**
1. **Cosine similarity** between successive steps (should be >0, ideally >0.8)
2. **Path efficiency**: straight-line distance / path length (should be close to 1.0)

**Implementation:**
```python
def path_smoothness(trajectories):
    cosine_sims = []
    efficiencies = []

    for traj in trajectories:
        # Cosine between successive direction vectors
        cosines = []
        for i in range(len(traj) - 2):
            v1 = traj[i+1] - traj[i]
            v2 = traj[i+2] - traj[i+1]

            v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
            v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)

            cosine = np.dot(v1_norm, v2_norm)
            cosines.append(cosine)

        cosine_sims.append(np.mean(cosines))

        # Path efficiency
        straight_line = np.linalg.norm(traj[-1] - traj[0])
        path_length = sum(np.linalg.norm(traj[i+1] - traj[i]) for i in range(len(traj)-1))
        efficiencies.append(straight_line / (path_length + 1e-8))

    return {
        'mean_cosine': np.mean(cosine_sims),
        'mean_efficiency': np.mean(efficiencies)
    }

smoothness = path_smoothness(trajectories)
print(f"Mean cosine similarity: {smoothness['mean_cosine']:.3f}")  # Should be >0.5
print(f"Mean path efficiency: {smoothness['mean_efficiency']:.3f}")  # Should be >0.8
```

### C. Curl/Divergence Stability

**What it measures:** Check if regularization actually constrains the field.

**Implementation:** Already tracked during training! Check TensorBoard:
```bash
tensorboard --logdir results/simple_final/logs

# Look at:
# - Train/Curl (should be <0.1)
# - Train/Div (should be <1.0)
```

**Ablation:** Compare models trained with different λ_curl and λ_div.

---

## 3. Control / Falsification Tests

### A. Shuffled Labels

**Test:** Train with shuffled attribute labels. Performance should drop to random.

**Implementation:**
```python
# In training, before loss computation:
if use_shuffled_labels:
    target_attributes = target_attributes[torch.randperm(len(target_attributes))]
    # This breaks attribute-embedding correspondence

# Expected: All metrics drop to baseline (silhouette~0, accuracy~50%)
```

### B. Random Vector Field Baseline

**Test:** Replace learned field with random Gaussian noise.

**Implementation:**
```python
def random_vector_field(z, y):
    """Random baseline."""
    return torch.randn_like(z) * 0.1

# Use this instead of model during evaluation
# Expected: All metrics at random baseline level
```

### C. Identity-Level Split

**Test:** Split train/val/test by person identity (not by image).

**Problem:** CelebA doesn't have identity labels! But you can approximate:
```python
# Ensure same person doesn't appear in train and test
# For CelebA, images are roughly chronological per person
# Use first 162k for train, next 20k for val, last 20k for test
# (Already done in standard split)
```

---

## 4. Visualization Improvements

### A. Multiple Dimensionality Reduction Methods

Compare UMAP, PCA, t-SNE to ensure trends are consistent:

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

# PCA (deterministic, linear)
pca = PCA(n_components=2, random_state=42)
coords_pca = pca.fit_transform(flowed_emb)

# t-SNE (non-linear, preserves local structure)
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
coords_tsne = tsne.fit_transform(flowed_emb)

# UMAP (non-linear, preserves global + local)
reducer = umap.UMAP(random_state=42, n_neighbors=15, min_dist=0.1)
coords_umap = reducer.fit_transform(flowed_emb)

# Plot all three side-by-side
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, coords, title in zip(axes, [coords_pca, coords_tsne, coords_umap],
                              ['PCA', 't-SNE', 'UMAP']):
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=target_labels, cmap='tab10', s=1)
    ax.set_title(title)

# Expected: All three should show similar clustering trends
```

### B. Class Centroids + Covariance Ellipses

Show cluster centers and spread:

```python
from matplotlib.patches import Ellipse

def plot_with_centroids(coords, labels, ax):
    unique_labels = np.unique(labels)

    for label in unique_labels:
        mask = labels == label
        points = coords[mask]

        # Plot points
        ax.scatter(points[:, 0], points[:, 1], label=label, s=10, alpha=0.5)

        # Compute centroid
        centroid = points.mean(axis=0)
        ax.scatter(*centroid, marker='X', s=200, edgecolors='black', linewidths=2)

        # Covariance ellipse (2 std devs)
        cov = np.cov(points.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(*eigenvectors[:, 1][::-1]))
        width, height = 2 * 2 * np.sqrt(eigenvalues)  # 2 std devs

        ellipse = Ellipse(centroid, width, height, angle=angle,
                         alpha=0.2, facecolor='none', edgecolor='black', linewidth=2)
        ax.add_patch(ellipse)

# Use this to visualize cluster quality
```

### C. Linear Probe Decision Boundaries

Visualize what the linear classifier learned:

```python
from sklearn.linear_model import LogisticRegression

# Train linear probe in 2D reduced space
clf = LogisticRegression()
clf.fit(coords_2d, target_labels)

# Plot decision boundary
x_min, x_max = coords_2d[:, 0].min() - 1, coords_2d[:, 0].max() + 1
y_min, y_max = coords_2d[:, 1].min() - 1, coords_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(coords_2d[:, 0], coords_2d[:, 1], c=target_labels, edgecolors='k', s=20)
plt.title("Linear Probe Decision Boundary")
```

---

## 5. Quick Implementation Script

I've created `scripts/comprehensive_evaluation.py` which:
1. Collects all necessary data
2. Computes all metrics
3. Generates all visualizations

**Usage:**
```bash
# Collect evaluation data
python scripts/comprehensive_evaluation.py \
    --checkpoint results/simple_final/checkpoints/fclf_best.pt \
    --output_dir results/comprehensive_eval \
    --num_samples 2000

# This will create:
# - evaluation_data.npz (all embeddings, trajectories)
# - metrics.json (all quantitative metrics)
# - figures/ (all visualizations)
```

---

## 6. Expected Results Summary

| Metric | Original→Target | Flowed→Target | Improvement |
|--------|-----------------|---------------|-------------|
| Linear probe acc | ~50% (random) | >75% | ✅ +25% |
| Silhouette | ~0 (no cluster) | 0.3-0.5 | ✅ +0.4 |
| k-NN purity | ~50% | >80% | ✅ +30% |
| Centroid dist | 1.0 | <0.5 | ✅ -50% |
| Monotonic % | N/A | >70% | ✅ |
| Path cosine | N/A | >0.5 | ✅ |
| Path efficiency | N/A | >0.8 | ✅ |

---

## 7. Presentation Tips

### For Each Metric, Show:
1. **Quantitative table** with confidence intervals
2. **Before/after comparison** (original vs flowed)
3. **Per-attribute breakdown** (does it work for all 5 attributes?)
4. **Statistical significance** (t-test, p-values)

### Visualization Best Practices:
1. **Use multiple methods** (UMAP, PCA, t-SNE) to show consistency
2. **Show trajectories** as arrows, not just endpoints
3. **Color by both** original and target attributes
4. **Include error bars** and confidence regions
5. **Larger N** (show 2000+ points, not just 1000)

This comprehensive evaluation will make your CS229 project publication-quality!
