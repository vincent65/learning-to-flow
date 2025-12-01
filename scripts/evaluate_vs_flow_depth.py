"""
Evaluate FCLF model at different flow depths K.

Following cs229.ipynb approach: train fresh classifier at each K value
to determine optimal flow depth.

Usage:
    python scripts/evaluate_vs_flow_depth.py \
        --checkpoint outputs/v4/checkpoints/fclf_best.pt \
        --celeba_root data/celeba \
        --embedding_dir data/embeddings \
        --output_dir outputs/v4/flow_depth \
        --k_values 0 1 2 5 10 15 20 \
        --device cuda
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
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from models.vector_field import VectorFieldNetwork
from data.celeba_dataset import get_dataloader

ATTRIBUTE_NAMES = ['Smiling', 'Young', 'Male', 'Eyeglasses', 'Mustache']

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300


def compute_geometry_metrics(embeddings, attributes):
    """
    Compute global within-class and between-class distances.

    Args:
        embeddings: [N, D] embeddings
        attributes: [N, num_attrs] binary attributes

    Returns:
        within_class_dist: Mean distance to centroid within each class
        between_class_dist: Mean pairwise distance between class centroids
    """
    # Convert attributes to string labels (for 32 unique combinations)
    attr_strings = [''.join(map(str, row.astype(int))) for row in attributes.numpy()]
    unique_labels = list(set(attr_strings))

    # Compute centroids and within-class distances
    within_dists = []
    centroids = {}

    for label in unique_labels:
        mask = np.array([s == label for s in attr_strings])
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

    return within_class_dist, between_class_dist


def evaluate_at_depth_K(model, embeddings, attributes, K, device):
    """
    Evaluate model at flow depth K.

    Args:
        model: VectorFieldNetwork
        embeddings: [N, D] original embeddings
        attributes: [N, num_attrs] attributes (used as flow targets)
        K: Number of flow steps
        device: torch device

    Returns:
        Dict with per-attribute accuracy, geometry metrics
    """
    model.eval()

    with torch.no_grad():
        if K == 0:
            # No flow - just use original embeddings
            z_flowed = embeddings
        else:
            # Apply K-step flow
            z_flowed = model.get_trajectory(embeddings.to(device), attributes.to(device), num_steps=K)
            z_flowed = z_flowed[:, -1, :].cpu()  # Take final step

    # Train fresh classifier for each attribute
    results = {'K': K, 'per_attribute': {}}

    for attr_idx, attr_name in enumerate(ATTRIBUTE_NAMES):
        y = attributes[:, attr_idx].numpy()

        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            z_flowed.numpy(), y, test_size=0.3, random_state=42, stratify=y
        )

        # Train logistic regression
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)

        # Evaluate
        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)

        # AUC
        if len(np.unique(y_test)) > 1:
            y_pred_proba = clf.predict_proba(X_test)[:, 1]
            test_auc = roc_auc_score(y_test, y_pred_proba)
        else:
            test_auc = 0.5

        results['per_attribute'][attr_name] = {
            'train_acc': float(train_acc),
            'test_acc': float(test_acc),
            'test_auc': float(test_auc)
        }

    # Compute average accuracy
    train_accs = [v['train_acc'] for v in results['per_attribute'].values()]
    test_accs = [v['test_acc'] for v in results['per_attribute'].values()]
    test_aucs = [v['test_auc'] for v in results['per_attribute'].values()]

    results['mean_train_acc'] = float(np.mean(train_accs))
    results['mean_test_acc'] = float(np.mean(test_accs))
    results['mean_test_auc'] = float(np.mean(test_aucs))

    # Geometry metrics
    within_dist, between_dist = compute_geometry_metrics(z_flowed, attributes)
    results['within_class_dist'] = within_dist
    results['between_class_dist'] = between_dist
    results['geometry_ratio'] = between_dist / (within_dist + 1e-8)

    return results


def plot_accuracy_vs_K(all_results, output_path):
    """
    Plot per-attribute test accuracy vs K.

    Args:
        all_results: List of results dicts (one per K)
        output_path: Where to save figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    K_values = [r['K'] for r in all_results]

    # Plot per-attribute curves
    for idx, attr_name in enumerate(ATTRIBUTE_NAMES):
        ax = axes[idx]

        train_accs = [r['per_attribute'][attr_name]['train_acc'] for r in all_results]
        test_accs = [r['per_attribute'][attr_name]['test_acc'] for r in all_results]

        ax.plot(K_values, train_accs, 'o--', label='Train', linewidth=2, markersize=8, alpha=0.7)
        ax.plot(K_values, test_accs, 'o-', label='Test', linewidth=2, markersize=8)

        ax.set_xlabel('Flow Steps (K)', fontsize=11)
        ax.set_ylabel('Accuracy', fontsize=11)
        ax.set_title(f'{attr_name}', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend(loc='best')
        ax.set_ylim([0.5, 1.0])

    # Plot mean accuracy in last subplot
    ax = axes[5]
    mean_train = [r['mean_train_acc'] for r in all_results]
    mean_test = [r['mean_test_acc'] for r in all_results]

    ax.plot(K_values, mean_train, 'o--', label='Train (mean)', linewidth=2, markersize=8, alpha=0.7)
    ax.plot(K_values, mean_test, 'o-', label='Test (mean)', linewidth=2, markersize=8)

    ax.set_xlabel('Flow Steps (K)', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title('Mean Across Attributes', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(loc='best')
    ax.set_ylim([0.5, 1.0])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved accuracy vs K plot to: {output_path}")


def plot_geometry_vs_K(all_results, output_path):
    """
    Plot within/between class distances vs K.

    Args:
        all_results: List of results dicts (one per K)
        output_path: Where to save figure
    """
    K_values = [r['K'] for r in all_results]
    within_dists = [r['within_class_dist'] for r in all_results]
    between_dists = [r['between_class_dist'] for r in all_results]
    ratios = [r['geometry_ratio'] for r in all_results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Within-class distance
    axes[0].plot(K_values, within_dists, 'o-', linewidth=2, markersize=8, color='tab:blue')
    axes[0].set_xlabel('Flow Steps (K)', fontsize=12)
    axes[0].set_ylabel('Distance', fontsize=12)
    axes[0].set_title('Within-Class Distance\n(lower = tighter clusters)', fontsize=12, fontweight='bold')
    axes[0].grid(alpha=0.3)

    # Between-class distance
    axes[1].plot(K_values, between_dists, 'o-', linewidth=2, markersize=8, color='tab:orange')
    axes[1].set_xlabel('Flow Steps (K)', fontsize=12)
    axes[1].set_ylabel('Distance', fontsize=12)
    axes[1].set_title('Between-Class Distance\n(higher = better separation)', fontsize=12, fontweight='bold')
    axes[1].grid(alpha=0.3)

    # Ratio (separation / compactness)
    axes[2].plot(K_values, ratios, 'o-', linewidth=2, markersize=8, color='tab:green')
    axes[2].set_xlabel('Flow Steps (K)', fontsize=12)
    axes[2].set_ylabel('Ratio', fontsize=12)
    axes[2].set_title('Separation / Compactness Ratio\n(higher = better clustering)', fontsize=12, fontweight='bold')
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved geometry vs K plot to: {output_path}")


def find_optimal_K(all_results):
    """
    Recommend optimal K based on multiple criteria.

    Args:
        all_results: List of results dicts

    Returns:
        optimal_K: Recommended K value
        analysis: Dict explaining the choice
    """
    K_values = [r['K'] for r in all_results]
    mean_test_accs = [r['mean_test_acc'] for r in all_results]
    geometry_ratios = [r['geometry_ratio'] for r in all_results]

    # Find K with best test accuracy
    best_acc_idx = np.argmax(mean_test_accs)
    K_best_acc = K_values[best_acc_idx]
    best_acc = mean_test_accs[best_acc_idx]

    # Find K with best geometry ratio
    best_geo_idx = np.argmax(geometry_ratios)
    K_best_geo = K_values[best_geo_idx]
    best_geo = geometry_ratios[best_geo_idx]

    # Check for saturation (accuracy stops improving)
    saturation_K = None
    for i in range(1, len(mean_test_accs)):
        if mean_test_accs[i] - mean_test_accs[i-1] < 0.005:  # Less than 0.5% improvement
            saturation_K = K_values[i]
            break

    # Recommend K (prefer lower K if accuracy is similar)
    threshold = 0.98  # Within 2% of best
    candidates = []
    for k, acc in zip(K_values, mean_test_accs):
        if acc >= threshold * best_acc and k > 0:  # Exclude K=0
            candidates.append(k)

    optimal_K = min(candidates) if candidates else K_best_acc

    analysis = {
        'optimal_K': int(optimal_K),
        'K_best_accuracy': int(K_best_acc),
        'best_accuracy': float(best_acc),
        'K_best_geometry': int(K_best_geo),
        'best_geometry_ratio': float(best_geo),
        'saturation_K': int(saturation_K) if saturation_K else None,
        'reasoning': f"K={optimal_K} achieves {mean_test_accs[K_values.index(optimal_K)]:.4f} accuracy "
                    f"(within 2% of best {best_acc:.4f}) while being minimal."
    }

    return optimal_K, analysis


def main():
    parser = argparse.ArgumentParser(description='Evaluate FCLF at different flow depths')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--celeba_root', type=str, default='data/celeba', help='CelebA dataset root')
    parser.add_argument('--embedding_dir', type=str, default='data/embeddings', help='Embedding directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--k_values', type=int, nargs='+', default=[0, 1, 2, 5, 10, 15, 20],
                       help='K values to test')
    parser.add_argument('--num_samples', type=int, default=5000, help='Number of test samples')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("="*80)
    print("FLOW DEPTH EVALUATION (cs229.ipynb-style)")
    print("="*80)
    print(f"\nTesting K values: {args.k_values}")
    print(f"Using {args.num_samples} test samples")

    # Load model
    print("\nLoading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    config = checkpoint['config']

    model = VectorFieldNetwork(
        embedding_dim=config['model']['embedding_dim'],
        num_attributes=config['model']['num_attributes'],
        hidden_dim=config['model']['hidden_dim'],
        projection_radius=config['model'].get('projection_radius', 1.0)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    model.eval()

    # Load test data
    print("Loading test data...")
    test_loader = get_dataloader(
        root_dir=args.celeba_root,
        split='test',
        batch_size=512,
        embedding_path=os.path.join(args.embedding_dir, 'test_embeddings.pt'),
        load_images=False,
        num_workers=0,
        shuffle=False
    )

    # Collect embeddings
    print(f"Collecting {args.num_samples} samples...")
    all_embeddings = []
    all_attributes = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Loading data"):
            embeddings = batch['embedding']
            attributes = batch['attributes']

            all_embeddings.append(embeddings)
            all_attributes.append(attributes)

            if sum(len(e) for e in all_embeddings) >= args.num_samples:
                break

    all_embeddings = torch.cat(all_embeddings)[:args.num_samples]
    all_attributes = torch.cat(all_attributes)[:args.num_samples]

    print(f"Collected {len(all_embeddings)} samples")

    # Evaluate at each K
    print("\n" + "="*80)
    print("EVALUATING AT DIFFERENT K VALUES")
    print("="*80)

    all_results = []

    for K in args.k_values:
        print(f"\n--- K = {K} ---")
        results = evaluate_at_depth_K(model, all_embeddings, all_attributes, K, args.device)
        all_results.append(results)

        print(f"Mean Test Accuracy:  {results['mean_test_acc']:.4f}")
        print(f"Mean Test AUC:       {results['mean_test_auc']:.4f}")
        print(f"Within-class dist:   {results['within_class_dist']:.4f}")
        print(f"Between-class dist:  {results['between_class_dist']:.4f}")
        print(f"Geometry ratio:      {results['geometry_ratio']:.2f}")

    # Find optimal K
    print("\n" + "="*80)
    print("OPTIMAL K ANALYSIS")
    print("="*80)

    optimal_K, analysis = find_optimal_K(all_results)

    print(f"\n✅ Recommended K: {optimal_K}")
    print(f"\nAnalysis:")
    print(f"  Best accuracy: K={analysis['K_best_accuracy']} ({analysis['best_accuracy']:.4f})")
    print(f"  Best geometry: K={analysis['K_best_geometry']} (ratio={analysis['best_geometry_ratio']:.2f})")
    if analysis['saturation_K']:
        print(f"  Saturation point: K={analysis['saturation_K']}")
    print(f"\n  {analysis['reasoning']}")

    # Save results
    results_file = os.path.join(args.output_dir, 'flow_depth_analysis.json')
    output_data = {
        'k_values': args.k_values,
        'results': all_results,
        'optimal_K_analysis': analysis
    }

    with open(results_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\n✅ Results saved to: {results_file}")

    # Generate plots
    print("\nGenerating plots...")

    plot_accuracy_vs_K(
        all_results,
        os.path.join(args.output_dir, 'accuracy_vs_K.png')
    )

    plot_geometry_vs_K(
        all_results,
        os.path.join(args.output_dir, 'geometry_vs_K.png')
    )

    print("\n" + "="*80)
    print("FLOW DEPTH EVALUATION COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
