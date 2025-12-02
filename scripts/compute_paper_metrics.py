"""
Comprehensive evaluation metrics for FCLF paper.
Includes attribute leakage, AUC curves, field diagnostics, baselines, and flipbook data.

Usage:
    python scripts/compute_paper_metrics.py \
        --checkpoint checkpoints/fclf_epoch_20.pt \
        --output_dir paper_metrics
"""

import os
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import sys
# Add project root and src directory to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from models.vector_field import VectorFieldNetwork
from data.celeba_dataset import get_dataloader


ATTRIBUTE_NAMES = ['Smiling', 'Young', 'Male', 'Eyeglasses', 'Mustache']


def compute_attribute_leakage(trajectories, original_attrs, target_attrs, num_samples=2000):
    """
    Measure how much NON-target attributes change during flow.

    For each attribute that is NOT being changed, measure the probe accuracy
    before and after flow. Leakage = |ΔAccuracy| (should be ~0).

    Returns:
        Dict with per-attribute leakage scores
    """
    results = {}
    num_attrs = original_attrs.shape[1]

    z_start = trajectories[:, 0, :]  # [N, 512]
    z_end = trajectories[:, -1, :]   # [N, 512]

    for attr_idx in range(num_attrs):
        # Find samples where this attribute was NOT changed
        unchanged_mask = (original_attrs[:, attr_idx] == target_attrs[:, attr_idx])

        if unchanged_mask.sum() < 50:
            results[ATTRIBUTE_NAMES[attr_idx]] = {
                'leakage': 0.0,
                'n_samples': int(unchanged_mask.sum()),
                'note': 'insufficient samples'
            }
            continue

        # Get embeddings and labels for unchanged samples
        z_start_unchanged = z_start[unchanged_mask]
        z_end_unchanged = z_end[unchanged_mask]
        labels = original_attrs[unchanged_mask, attr_idx].numpy()

        # Train probe on start embeddings
        X_train_start, X_test_start, y_train, y_test = train_test_split(
            z_start_unchanged.numpy(), labels, test_size=0.3, random_state=42
        )
        clf_start = LogisticRegression(max_iter=1000, random_state=42)
        clf_start.fit(X_train_start, y_train)
        acc_start = clf_start.score(X_test_start, y_test)

        # Train probe on end embeddings
        X_train_end, X_test_end = train_test_split(
            z_end_unchanged.numpy(), test_size=0.3, random_state=42
        )[0], train_test_split(z_end_unchanged.numpy(), test_size=0.3, random_state=42)[1]
        clf_end = LogisticRegression(max_iter=1000, random_state=42)
        clf_end.fit(X_train_end, y_train)
        acc_end = clf_end.score(X_test_end, y_test)

        # Leakage = absolute change in accuracy
        leakage = abs(acc_end - acc_start)

        results[ATTRIBUTE_NAMES[attr_idx]] = {
            'accuracy_start': float(acc_start),
            'accuracy_end': float(acc_end),
            'leakage': float(leakage),
            'n_samples': int(unchanged_mask.sum())
        }

    # Overall leakage score
    leakages = [v['leakage'] for v in results.values() if 'note' not in v]
    results['_overall'] = {
        'mean_leakage': float(np.mean(leakages)),
        'max_leakage': float(np.max(leakages))
    }

    return results


def compute_linear_steering_baseline(original_emb, original_attrs, target_attrs, alpha=0.5):
    """
    Compute linear CLIP steering baseline.

    For each attribute, compute:
        v_attr = mean(attr=1) - mean(attr=0)

    Then steer: z_steered = normalize(z + alpha * sum(v_attr * (target - original)))

    Returns:
        z_steered: [N, 512] steered embeddings
    """
    num_attrs = original_attrs.shape[1]
    device = original_emb.device

    # Compute class-difference vectors for each attribute
    steering_vectors = []
    for attr_idx in range(num_attrs):
        pos_mask = original_attrs[:, attr_idx] == 1
        neg_mask = original_attrs[:, attr_idx] == 0

        if pos_mask.sum() > 0 and neg_mask.sum() > 0:
            v_pos = original_emb[pos_mask].mean(dim=0)
            v_neg = original_emb[neg_mask].mean(dim=0)
            v_diff = v_pos - v_neg
            v_diff = F.normalize(v_diff, dim=0)
        else:
            v_diff = torch.zeros(original_emb.shape[1], device=device)

        steering_vectors.append(v_diff)

    steering_vectors = torch.stack(steering_vectors)  # [5, 512]

    # Compute steering direction for each sample
    z_steered = original_emb.clone()
    for i in range(len(original_emb)):
        # Compute attribute changes
        attr_change = (target_attrs[i] - original_attrs[i]).to(original_emb.dtype)  # [5], match dtype

        # Weighted sum of steering vectors
        steering_direction = (steering_vectors.T @ attr_change).squeeze()  # [512]

        # Apply steering
        z_steered[i] = original_emb[i] + alpha * steering_direction

        # Normalize to stay in CLIP space
        z_steered[i] = F.normalize(z_steered[i], dim=0)

    return z_steered


def compute_auc_along_path(trajectories, target_attrs, num_steps=10):
    """
    Compute per-attribute AUC at each step along the trajectory.

    Returns:
        auc_curves: Dict[attr_name -> List[float]] AUC at each step
        monotonic_frac: Dict[attr_name -> float] fraction of paths with monotonic AUC increase
    """
    num_attrs = target_attrs.shape[1]
    num_samples = trajectories.shape[0]

    auc_curves = {attr: [] for attr in ATTRIBUTE_NAMES}
    monotonic_counts = {attr: 0 for attr in ATTRIBUTE_NAMES}

    # For each time step
    for step in range(num_steps + 1):
        z_t = trajectories[:, step, :].numpy()  # [N, 512]

        # For each attribute
        for attr_idx in range(num_attrs):
            y_true = target_attrs[:, attr_idx].numpy()

            # Train classifier
            X_train, X_test, y_train, y_test = train_test_split(
                z_t, y_true, test_size=0.3, random_state=42
            )

            clf = LogisticRegression(max_iter=1000, random_state=42)
            clf.fit(X_train, y_train)

            # Compute AUC
            if len(np.unique(y_test)) > 1:
                y_pred_proba = clf.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_pred_proba)
            else:
                auc = 0.5  # undefined

            auc_curves[ATTRIBUTE_NAMES[attr_idx]].append(float(auc))

    # Check monotonicity
    for attr_idx in range(num_attrs):
        attr_name = ATTRIBUTE_NAMES[attr_idx]
        aucs = auc_curves[attr_name]

        # Count how many have monotonically increasing AUC
        is_monotonic = all(aucs[i] <= aucs[i+1] for i in range(len(aucs)-1))
        if is_monotonic:
            monotonic_counts[attr_name] += 1

    monotonic_frac = {
        attr: monotonic_counts[attr] / 1.0  # Only 1 curve per attribute
        for attr in ATTRIBUTE_NAMES
    }

    return auc_curves, monotonic_frac


def compute_field_diagnostics(model, embeddings, attributes, device, grid_size=500):
    """
    Sample points in embedding space and compute curl/divergence statistics.

    For vector field v(z, a):
        - Divergence: ∇·v = sum_i ∂v_i/∂z_i
        - Curl magnitude: ||∇×v||

    Returns:
        Dict with mean, std, max of divergence and curl
    """
    model.eval()

    # Sample random embeddings and attributes
    indices = torch.randperm(len(embeddings))[:grid_size]
    z_samples = embeddings[indices].to(device)
    a_samples = attributes[indices].to(device)

    # Enable gradient tracking
    z_samples.requires_grad_(True)

    # Compute vector field
    v = model(z_samples, a_samples)  # [grid_size, 512]

    # Compute divergence (trace of Jacobian)
    divergences = []
    for i in range(grid_size):
        # Compute gradient of v[i] w.r.t. z_samples[i]
        div = 0.0
        for dim in range(min(512, 10)):  # Sample 10 dimensions to save compute
            grad_outputs = torch.zeros_like(v)
            grad_outputs[i, dim] = 1.0

            grad = torch.autograd.grad(
                outputs=v,
                inputs=z_samples,
                grad_outputs=grad_outputs,
                retain_graph=True,
                create_graph=False
            )[0]

            div += grad[i, dim].item()

        divergences.append(div)

    divergences = np.array(divergences)

    # Compute curl magnitude (simplified: use off-diagonal Jacobian elements)
    # Full curl in high-D is expensive, so we approximate with Frobenius norm of antisymmetric part
    curls = []
    for i in range(min(grid_size, 100)):  # Sample 100 points to save time
        # Compute Jacobian for this point
        J = []
        for dim in range(min(512, 10)):  # Sample 10 dimensions
            grad_outputs = torch.zeros_like(v)
            grad_outputs[i, dim] = 1.0

            grad = torch.autograd.grad(
                outputs=v,
                inputs=z_samples,
                grad_outputs=grad_outputs,
                retain_graph=True,
                create_graph=False
            )[0]

            J.append(grad[i, :10].detach().cpu().numpy())

        J = np.array(J)  # [10, 10]

        # Antisymmetric part: (J - J.T) / 2
        curl_matrix = (J - J.T) / 2
        curl_magnitude = np.linalg.norm(curl_matrix, 'fro')
        curls.append(curl_magnitude)

    curls = np.array(curls)

    return {
        'divergence': {
            'mean': float(np.mean(divergences)),
            'std': float(np.std(divergences)),
            'max': float(np.max(np.abs(divergences)))
        },
        'curl': {
            'mean': float(np.mean(curls)),
            'std': float(np.std(curls)),
            'max': float(np.max(curls))
        }
    }


def compute_nearest_neighbor_flipbook(trajectories, all_embeddings, original_attrs, target_attrs, num_paths=50, k=1):
    """
    For each trajectory, find the k-nearest training images at each step.

    Returns:
        flipbook_data: Dict with indices, distances, and attribute changes for visualization
    """
    num_steps = trajectories.shape[1]
    all_embeddings_np = all_embeddings.numpy()

    # Pre-normalize all training embeddings once (MAJOR SPEEDUP!)
    all_emb_norm = all_embeddings_np / (np.linalg.norm(all_embeddings_np, axis=1, keepdims=True) + 1e-8)

    # Select random paths
    indices = np.random.choice(len(trajectories), size=min(num_paths, len(trajectories)), replace=False)

    flipbook_data = []

    for idx in tqdm(indices, desc="Computing nearest neighbors"):
        path = trajectories[idx]  # [num_steps, 512]

        nearest_indices = []
        distances = []

        for step in range(num_steps):
            z_t = path[step].numpy()  # [512]

            # Compute distances using cosine similarity (better for normalized embeddings)
            # cosine_sim = dot(a, b) / (||a|| * ||b||)
            # For unit vectors: cosine_sim = dot(a, b)
            # distance = 1 - cosine_sim
            z_t_norm = z_t / (np.linalg.norm(z_t) + 1e-8)  # Normalize query
            cosine_sim = np.dot(all_emb_norm, z_t_norm)  # Use pre-normalized embeddings
            dists = 1 - cosine_sim  # Convert similarity to distance

            # Find k nearest
            k_nearest = np.argsort(dists)[:k]

            nearest_indices.append(k_nearest.tolist())
            distances.append(dists[k_nearest].tolist())

        # Determine which attributes changed
        orig_attr = original_attrs[idx].numpy()
        targ_attr = target_attrs[idx].numpy()
        attr_changes = []

        for attr_idx, attr_name in enumerate(ATTRIBUTE_NAMES):
            if orig_attr[attr_idx] != targ_attr[attr_idx]:
                direction = "add" if targ_attr[attr_idx] == 1 else "remove"
                attr_changes.append(f"{direction}_{attr_name}")

        # Create descriptive name
        change_str = "_".join(attr_changes) if attr_changes else "no_change"

        flipbook_data.append({
            'trajectory_idx': int(idx),
            'nearest_indices': nearest_indices,  # [num_steps, k]
            'distances': distances,  # [num_steps, k]
            'original_attributes': orig_attr.tolist(),
            'target_attributes': targ_attr.tolist(),
            'attribute_changes': attr_changes,
            'change_string': change_str
        })

    return flipbook_data


def compute_global_geometry_metrics(embeddings, attributes):
    """
    Compute global within-class and between-class distances.

    Following cs229.ipynb approach for geometry analysis.

    Args:
        embeddings: [N, D] tensor of embeddings
        attributes: [N, num_attrs] tensor of binary attributes

    Returns:
        within_class_dist: Mean distance from samples to their class centroid
        between_class_dist: Mean pairwise distance between class centroids
        geometry_ratio: between / within (higher is better clustering)
    """
    # Convert attributes to string labels (for 32 unique combinations)
    def attrs_to_labels(attrs):
        return np.array([''.join(map(str, row.astype(int))) for row in attrs])

    labels = attrs_to_labels(attributes.numpy())
    unique_labels = np.unique(labels)

    # Compute centroids and within-class distances
    within_dists = []
    centroids = {}

    for label in unique_labels:
        mask = labels == label
        if mask.sum() < 2:
            continue

        class_emb = embeddings[mask]
        centroid = class_emb.mean(dim=0)
        centroids[label] = centroid

        # Within-class: mean distance to centroid
        distances = torch.norm(class_emb - centroid, dim=1)
        within_dists.extend(distances.numpy())

    within_class_dist = float(np.mean(within_dists)) if within_dists else 0.0

    # Compute between-class distances (pairwise centroid distances)
    between_dists = []
    centroid_list = list(centroids.values())

    for i in range(len(centroid_list)):
        for j in range(i + 1, len(centroid_list)):
            dist = torch.norm(centroid_list[i] - centroid_list[j]).item()
            between_dists.append(dist)

    between_class_dist = float(np.mean(between_dists)) if between_dists else 0.0
    geometry_ratio = between_class_dist / (within_class_dist + 1e-8)

    return within_class_dist, between_class_dist, geometry_ratio


def evaluate_method(embeddings, trajectories, original_attrs, target_attrs, method_name="FCLF"):
    """
    Evaluate a method (FCLF or baseline) on all metrics.

    Returns comprehensive metrics dict.
    """
    z_end = trajectories[:, -1, :] if len(trajectories.shape) == 3 else trajectories

    # 1. Linear probe accuracy per attribute
    linear_probe = {}
    for attr_idx in range(original_attrs.shape[1]):
        X_train, X_test, y_train, y_test = train_test_split(
            z_end.numpy(), target_attrs[:, attr_idx].numpy(), test_size=0.3, random_state=42
        )
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)

        linear_probe[ATTRIBUTE_NAMES[attr_idx]] = float(acc)

    # 2. k-NN purity
    from sklearn.neighbors import KNeighborsClassifier
    def attrs_to_labels(attrs):
        return np.array([''.join(map(str, row.astype(int))) for row in attrs])

    target_labels_str = attrs_to_labels(target_attrs.numpy())
    unique_labels = np.unique(target_labels_str)
    label_map = {lab: i for i, lab in enumerate(unique_labels)}
    target_labels_int = np.array([label_map[lab] for lab in target_labels_str])

    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(z_end.numpy(), target_labels_int)

    purities = []
    for i in range(len(z_end)):
        neighbors = knn.kneighbors([z_end[i].numpy()], return_distance=False)[0]
        purity = (target_labels_int[neighbors] == target_labels_int[i]).mean()
        purities.append(purity)

    knn_purity = float(np.mean(purities))

    # 3. Centroid distance
    centroids = {}
    for lab in unique_labels:
        mask = target_labels_str == lab
        centroids[lab] = z_end[mask].mean(dim=0).numpy()

    distances = [np.linalg.norm(z_end[i].numpy() - centroids[target_labels_str[i]])
                 for i in range(len(z_end))]
    centroid_dist = float(np.mean(distances))

    # 4. Geometry metrics (cs229.ipynb-style)
    within_dist, between_dist, geo_ratio = compute_global_geometry_metrics(z_end, target_attrs)

    return {
        'linear_probe': linear_probe,
        'knn_purity': knn_purity,
        'centroid_distance': centroid_dist,
        'within_class_distance': within_dist,
        'between_class_distance': between_dist,
        'geometry_ratio': geo_ratio
    }


def plot_auc_curves(auc_curves, output_path):
    """Plot AUC curves for each attribute."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, attr in enumerate(ATTRIBUTE_NAMES):
        ax = axes[idx]
        aucs = auc_curves[attr]
        steps = list(range(len(aucs)))

        ax.plot(steps, aucs, marker='o', linewidth=2, markersize=8)
        ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Random')
        ax.set_xlabel('Flow Step', fontsize=12)
        ax.set_ylabel('AUC', fontsize=12)
        ax.set_title(f'{attr}', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.set_ylim([0.4, 1.0])
        ax.legend()

    # Hide extra subplot
    axes[-1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--celeba_root', type=str, default='data/celeba')
    parser.add_argument('--embedding_dir', type=str, default='data/embeddings')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=2000)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print("Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    config = checkpoint['config']

    model = VectorFieldNetwork(
        embedding_dim=config['model']['embedding_dim'],
        num_attributes=config['model']['num_attributes'],
        hidden_dim=config['model']['hidden_dim']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    model.eval()

    # Load data
    print("Loading test data...")
    test_loader = get_dataloader(
        root_dir=args.celeba_root,
        split='test',
        batch_size=128,
        embedding_path=os.path.join(args.embedding_dir, 'test_embeddings.pt'),
        load_images=False,
        num_workers=0,
        shuffle=False
    )

    # Load TRAINING embeddings for flipbook nearest-neighbor search
    print("Loading training embeddings for flipbook search...")
    train_embeddings = torch.load(os.path.join(args.embedding_dir, 'train_embeddings.pt'))
    print(f"  Loaded {len(train_embeddings)} training embeddings")

    # Collect data
    print(f"\nCollecting {args.num_samples} samples...")
    original_embeddings = []
    original_attributes = []
    target_attributes = []
    fclf_trajectories = []

    count = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Collecting data"):
            if count >= args.num_samples:
                break

            embeddings = batch['embedding'].to(args.device)
            attributes = batch['attributes'].to(args.device)
            batch_size = embeddings.size(0)

            # Create target: flip exactly 1 random attribute
            target_attrs = attributes.clone()
            for i in range(batch_size):
                num_flips = 1  # Changed: only flip one attribute at a time
                attrs_to_flip = np.random.choice(5, size=num_flips, replace=False)
                for attr_idx in attrs_to_flip:
                    target_attrs[i, attr_idx] = 1 - target_attrs[i, attr_idx]

            # Get FCLF trajectory (use config values if available)
            num_steps = config.get('inference', {}).get('num_flow_steps', 10)
            step_size = config.get('inference', {}).get('step_size', 0.1)
            trajectory = model.get_trajectory(embeddings, target_attrs, num_steps=num_steps, step_size=step_size)

            # Store
            original_embeddings.append(embeddings.cpu())
            original_attributes.append(attributes.cpu())
            target_attributes.append(target_attrs.cpu())
            fclf_trajectories.append(trajectory.cpu())

            count += batch_size

    # Concatenate
    original_embeddings = torch.cat(original_embeddings)[:args.num_samples]
    original_attributes = torch.cat(original_attributes)[:args.num_samples]
    target_attributes = torch.cat(target_attributes)[:args.num_samples]
    fclf_trajectories = torch.cat(fclf_trajectories)[:args.num_samples]

    print("\n" + "="*80)
    print("COMPUTING PAPER METRICS")
    print("="*80)

    # 1. Attribute Leakage
    print("\n[1/7] Computing attribute leakage...")
    leakage = compute_attribute_leakage(fclf_trajectories, original_attributes, target_attributes)

    # 2. Linear Steering Baseline
    print("[2/7] Computing linear steering baseline...")
    z_linear_steered = compute_linear_steering_baseline(
        original_embeddings, original_attributes, target_attributes, alpha=0.5
    )

    # 3. AUC Along Path
    print("[3/7] Computing AUC along path...")
    auc_curves, monotonic_frac = compute_auc_along_path(fclf_trajectories, target_attributes, num_steps=10)

    # 4. Field Diagnostics
    print("[4/7] Computing field diagnostics...")
    field_stats = compute_field_diagnostics(model, original_embeddings, original_attributes, args.device)

    # 5. Nearest Neighbor Flipbook
    print("[5/7] Computing nearest-neighbor flipbook data...")
    flipbook = compute_nearest_neighbor_flipbook(
        fclf_trajectories, train_embeddings, original_attributes, target_attributes, num_paths=50, k=1
    )

    # 6. Evaluate FCLF
    print("[6/7] Evaluating FCLF method...")
    fclf_metrics = evaluate_method(
        original_embeddings, fclf_trajectories, original_attributes, target_attributes, "FCLF"
    )

    # 7. Evaluate Linear Steering
    print("[7/7] Evaluating linear steering baseline...")
    linear_metrics = evaluate_method(
        original_embeddings, z_linear_steered, original_attributes, target_attributes, "Linear"
    )

    # Compile results
    results = {
        'attribute_leakage': leakage,
        'auc_curves': auc_curves,
        'monotonic_auc_fraction': monotonic_frac,
        'field_diagnostics': field_stats,
        'comparison': {
            'fclf': fclf_metrics,
            'linear_steering': linear_metrics
        }
    }

    # Save results
    results_file = os.path.join(args.output_dir, 'paper_metrics.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Save flipbook data
    flipbook_file = os.path.join(args.output_dir, 'flipbook_data.json')
    with open(flipbook_file, 'w') as f:
        json.dump(flipbook, f, indent=2)

    # Plot AUC curves
    print("\nGenerating AUC curve plots...")
    plot_auc_curves(auc_curves, os.path.join(args.output_dir, 'auc_curves.png'))

    # Print summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    print("\n1. ATTRIBUTE LEAKAGE (lower is better):")
    for attr in ATTRIBUTE_NAMES:
        if 'note' in leakage[attr]:
            print(f"   {attr}: {leakage[attr]['note']}")
        else:
            print(f"   {attr}: {leakage[attr]['leakage']:.4f} "
                  f"(acc: {leakage[attr]['accuracy_start']:.3f} → {leakage[attr]['accuracy_end']:.3f})")
    print(f"   Overall Mean: {leakage['_overall']['mean_leakage']:.4f}")
    print(f"   Overall Max:  {leakage['_overall']['max_leakage']:.4f}")

    print("\n2. MONOTONIC AUC PROGRESS:")
    for attr in ATTRIBUTE_NAMES:
        print(f"   {attr}: {monotonic_frac[attr]:.1%}")

    print("\n3. FIELD DIAGNOSTICS:")
    print(f"   Divergence: mean={field_stats['divergence']['mean']:.4f}, "
          f"std={field_stats['divergence']['std']:.4f}, "
          f"max={field_stats['divergence']['max']:.4f}")
    print(f"   Curl:       mean={field_stats['curl']['mean']:.4f}, "
          f"std={field_stats['curl']['std']:.4f}, "
          f"max={field_stats['curl']['max']:.4f}")

    print("\n4. METHOD COMPARISON:")
    print("\n   Linear Probe Accuracy:")
    print("   " + "-"*60)
    print(f"   {'Attribute':<15} {'FCLF':<10} {'Linear':<10} {'Difference':<10}")
    print("   " + "-"*60)
    for attr in ATTRIBUTE_NAMES:
        fclf_acc = fclf_metrics['linear_probe'][attr]
        linear_acc = linear_metrics['linear_probe'][attr]
        diff = fclf_acc - linear_acc
        print(f"   {attr:<15} {fclf_acc:.4f}     {linear_acc:.4f}     {diff:+.4f}")

    fclf_avg = np.mean([fclf_metrics['linear_probe'][a] for a in ATTRIBUTE_NAMES])
    linear_avg = np.mean([linear_metrics['linear_probe'][a] for a in ATTRIBUTE_NAMES])
    print("   " + "-"*60)
    print(f"   {'AVERAGE':<15} {fclf_avg:.4f}     {linear_avg:.4f}     {fclf_avg - linear_avg:+.4f}")

    print("\n   Other Metrics:")
    print(f"   k-NN Purity:       FCLF={fclf_metrics['knn_purity']:.4f}  "
          f"Linear={linear_metrics['knn_purity']:.4f}  "
          f"(Δ={fclf_metrics['knn_purity'] - linear_metrics['knn_purity']:+.4f})")
    print(f"   Centroid Distance: FCLF={fclf_metrics['centroid_distance']:.4f}  "
          f"Linear={linear_metrics['centroid_distance']:.4f}  "
          f"(Δ={fclf_metrics['centroid_distance'] - linear_metrics['centroid_distance']:+.4f})")

    print("\n   Geometry Metrics (cs229.ipynb-style):")
    print(f"   Within-class:      FCLF={fclf_metrics['within_class_distance']:.4f}  "
          f"Linear={linear_metrics['within_class_distance']:.4f}  "
          f"(Δ={fclf_metrics['within_class_distance'] - linear_metrics['within_class_distance']:+.4f})")
    print(f"   Between-class:     FCLF={fclf_metrics['between_class_distance']:.4f}  "
          f"Linear={linear_metrics['between_class_distance']:.4f}  "
          f"(Δ={fclf_metrics['between_class_distance'] - linear_metrics['between_class_distance']:+.4f})")
    print(f"   Geometry Ratio:    FCLF={fclf_metrics['geometry_ratio']:.2f}  "
          f"Linear={linear_metrics['geometry_ratio']:.2f}  "
          f"(Δ={fclf_metrics['geometry_ratio'] - linear_metrics['geometry_ratio']:+.2f})")

    print(f"\n✅ Results saved to: {results_file}")
    print(f"✅ Flipbook data saved to: {flipbook_file}")
    print(f"✅ AUC curves saved to: {os.path.join(args.output_dir, 'auc_curves.png')}")

    print("\n" + "="*80)
    print("Next steps:")
    print("  1. Run: python scripts/generate_latex_tables.py paper_metrics/paper_metrics.json")
    print("  2. Run: python scripts/visualize_flipbook.py paper_metrics/flipbook_data.json")
    print("="*80)


if __name__ == '__main__':
    main()
