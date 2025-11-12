"""
Compute all comprehensive metrics in one script.
Usage: python scripts/compute_all_metrics.py --checkpoint <path> --output_dir <path>
"""

import os
import argparse
import json
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vector_field import VectorFieldNetwork
from data.celeba_dataset import get_dataloader


def attrs_to_labels(attrs):
    """Convert attribute vectors to string labels for clustering."""
    return np.array([''.join(map(str, row.astype(int))) for row in attrs])


def compute_linear_probe_metrics(original_emb, flowed_emb, original_attrs, target_attrs):
    """Compute linear probe accuracy for each attribute."""
    results = {}
    num_attrs = original_attrs.shape[1]

    for attr_idx in range(num_attrs):
        # Split data
        X_train_orig, X_test_orig, y_train, y_test = train_test_split(
            original_emb, target_attrs[:, attr_idx], test_size=0.2, random_state=42
        )
        X_train_flow, X_test_flow = train_test_split(
            flowed_emb, test_size=0.2, random_state=42
        )[0], train_test_split(flowed_emb, test_size=0.2, random_state=42)[1]

        # Train on original
        clf_orig = LogisticRegression(max_iter=1000, random_state=42)
        clf_orig.fit(X_train_orig, y_train)
        acc_orig = clf_orig.score(X_test_orig, y_test)

        # Train on flowed
        clf_flow = LogisticRegression(max_iter=1000, random_state=42)
        clf_flow.fit(X_train_flow, y_train)
        acc_flow = clf_flow.score(X_test_flow, y_test)

        results[f'attr_{attr_idx}'] = {
            'original_accuracy': float(acc_orig),
            'flowed_accuracy': float(acc_flow),
            'improvement': float(acc_flow - acc_orig)
        }

    return results


def compute_clustering_metrics(embeddings, labels_str):
    """Compute silhouette, Calinski-Harabasz, Davies-Bouldin."""
    if len(np.unique(labels_str)) < 2:
        return {'silhouette': -1.0, 'calinski_harabasz': 0.0, 'davies_bouldin': float('inf')}

    # Convert string labels to integers
    unique_labels = np.unique(labels_str)
    label_map = {lab: i for i, lab in enumerate(unique_labels)}
    labels_int = np.array([label_map[lab] for lab in labels_str])

    return {
        'silhouette': float(silhouette_score(embeddings, labels_int)),
        'calinski_harabasz': float(calinski_harabasz_score(embeddings, labels_int)),
        'davies_bouldin': float(davies_bouldin_score(embeddings, labels_int))
    }


def compute_knn_purity(embeddings, labels_str, k=10):
    """Compute k-NN class purity."""
    # Convert to int labels
    unique_labels = np.unique(labels_str)
    label_map = {lab: i for i, lab in enumerate(unique_labels)}
    labels_int = np.array([label_map[lab] for lab in labels_str])

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(embeddings, labels_int)

    purities = []
    for i in range(len(embeddings)):
        neighbors = knn.kneighbors([embeddings[i]], return_distance=False)[0]
        purity = (labels_int[neighbors] == labels_int[i]).mean()
        purities.append(purity)

    return float(np.mean(purities))


def compute_centroid_distance(embeddings, labels_str):
    """Mean distance to class centroid."""
    unique_labels = np.unique(labels_str)
    centroids = {}

    for lab in unique_labels:
        mask = labels_str == lab
        centroids[lab] = embeddings[mask].mean(axis=0)

    distances = [np.linalg.norm(embeddings[i] - centroids[labels_str[i]])
                 for i in range(len(embeddings))]

    return float(np.mean(distances))


def compute_monotonic_progress(trajectories, target_attrs):
    """Fraction of trajectories with monotonically decreasing distance to target."""
    # Compute centroids for each unique attribute combination
    unique_attrs = np.unique(target_attrs, axis=0)
    centroids = {}

    for attr_vec in unique_attrs:
        mask = np.all(target_attrs == attr_vec, axis=1)
        if mask.sum() > 0:
            centroids[tuple(attr_vec)] = trajectories[mask, -1, :].mean(axis=0)

    monotonic_count = 0
    for i in range(len(trajectories)):
        target_tuple = tuple(target_attrs[i])
        if target_tuple not in centroids:
            continue

        centroid = centroids[target_tuple]

        # Distance at each step
        distances = [np.linalg.norm(trajectories[i, step] - centroid)
                    for step in range(trajectories.shape[1])]

        # Check monotonic decrease
        is_monotonic = all(distances[j] >= distances[j+1] for j in range(len(distances)-1))
        if is_monotonic:
            monotonic_count += 1

    return float(monotonic_count / len(trajectories))


def compute_path_smoothness(trajectories):
    """Compute path smoothness metrics."""
    cosine_sims = []
    efficiencies = []

    for traj in trajectories:
        # Cosine similarity between successive steps
        cosines = []
        for i in range(len(traj) - 2):
            v1 = traj[i+1] - traj[i]
            v2 = traj[i+2] - traj[i+1]

            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)

            if norm1 > 1e-8 and norm2 > 1e-8:
                cosine = np.dot(v1, v2) / (norm1 * norm2)
                cosines.append(cosine)

        if cosines:
            cosine_sims.append(np.mean(cosines))

        # Path efficiency
        straight_line = np.linalg.norm(traj[-1] - traj[0])
        path_length = sum(np.linalg.norm(traj[i+1] - traj[i]) for i in range(len(traj)-1))

        if path_length > 1e-8:
            efficiencies.append(straight_line / path_length)

    return {
        'mean_cosine_similarity': float(np.mean(cosine_sims)) if cosine_sims else 0.0,
        'mean_path_efficiency': float(np.mean(efficiencies)) if efficiencies else 0.0
    }


def collect_data(model, dataloader, device, num_samples=2000):
    """Collect embeddings and trajectories."""
    original_embeddings = []
    original_attributes = []
    target_attributes = []
    flowed_embeddings = []
    trajectories = []

    model.eval()
    count = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting data"):
            if count >= num_samples:
                break

            embeddings = batch['embedding'].to(device)
            attributes = batch['attributes'].to(device)
            batch_size = embeddings.size(0)

            # Create target: flip random attribute
            target_attrs = attributes.clone()
            for i in range(batch_size):
                attr_idx = np.random.randint(0, 5)
                target_attrs[i, attr_idx] = 1 - target_attrs[i, attr_idx]

            # Get trajectory
            trajectory = model.get_trajectory(embeddings, target_attrs, num_steps=10)

            # Store
            original_embeddings.append(embeddings.cpu())
            original_attributes.append(attributes.cpu())
            target_attributes.append(target_attrs.cpu())
            flowed_embeddings.append(trajectory[:, -1].cpu())
            trajectories.append(trajectory.cpu())

            count += batch_size

    return {
        'original_emb': torch.cat(original_embeddings)[:num_samples].numpy(),
        'original_attrs': torch.cat(original_attributes)[:num_samples].numpy(),
        'target_attrs': torch.cat(target_attributes)[:num_samples].numpy(),
        'flowed_emb': torch.cat(flowed_embeddings)[:num_samples].numpy(),
        'trajectories': torch.cat(trajectories)[:num_samples].numpy(),
    }


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

    # Collect data
    print(f"Collecting {args.num_samples} samples...")
    data = collect_data(model, test_loader, args.device, args.num_samples)

    # Convert to labels
    orig_labels = attrs_to_labels(data['original_attrs'])
    target_labels = attrs_to_labels(data['target_attrs'])

    print("\n" + "="*60)
    print("COMPREHENSIVE EVALUATION METRICS")
    print("="*60)

    # 1. Linear Probe
    print("\n[1/6] Computing linear probe accuracy...")
    linear_probe = compute_linear_probe_metrics(
        data['original_emb'], data['flowed_emb'],
        data['original_attrs'], data['target_attrs']
    )

    # 2. Clustering metrics
    print("[2/6] Computing clustering metrics...")
    clustering_orig_target = compute_clustering_metrics(data['original_emb'], target_labels)
    clustering_flow_target = compute_clustering_metrics(data['flowed_emb'], target_labels)
    clustering_orig_orig = compute_clustering_metrics(data['original_emb'], orig_labels)
    clustering_flow_orig = compute_clustering_metrics(data['flowed_emb'], orig_labels)

    # 3. k-NN purity
    print("[3/6] Computing k-NN purity...")
    knn_orig_target = compute_knn_purity(data['original_emb'], target_labels, k=10)
    knn_flow_target = compute_knn_purity(data['flowed_emb'], target_labels, k=10)

    # 4. Centroid distance
    print("[4/6] Computing centroid distances...")
    centroid_orig_target = compute_centroid_distance(data['original_emb'], target_labels)
    centroid_flow_target = compute_centroid_distance(data['flowed_emb'], target_labels)

    # 5. Monotonic progress
    print("[5/6] Computing monotonic progress...")
    monotonic_frac = compute_monotonic_progress(data['trajectories'], data['target_attrs'])

    # 6. Path smoothness
    print("[6/6] Computing path smoothness...")
    smoothness = compute_path_smoothness(data['trajectories'])

    # Compile results
    results = {
        'linear_probe': linear_probe,
        'clustering': {
            'original_by_target': clustering_orig_target,
            'flowed_by_target': clustering_flow_target,
            'original_by_original': clustering_orig_orig,
            'flowed_by_original': clustering_flow_orig,
        },
        'knn_purity_k10': {
            'original_by_target': knn_orig_target,
            'flowed_by_target': knn_flow_target,
        },
        'centroid_distance': {
            'original_to_target': centroid_orig_target,
            'flowed_to_target': centroid_flow_target,
        },
        'path_quality': {
            'fraction_monotonic': monotonic_frac,
            **smoothness
        }
    }

    # Save results
    results_file = os.path.join(args.output_dir, 'comprehensive_metrics.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    print("\n1. Linear Probe Accuracy (per attribute):")
    for attr_idx in range(5):
        metrics = linear_probe[f'attr_{attr_idx}']
        print(f"  Attribute {attr_idx}: {metrics['original_accuracy']:.3f} → {metrics['flowed_accuracy']:.3f} "
              f"({metrics['improvement']:+.3f})")

    print("\n2. Clustering Quality (by target attributes):")
    print(f"  Silhouette: {clustering_orig_target['silhouette']:.3f} → {clustering_flow_target['silhouette']:.3f}")
    print(f"  Calinski-Harabasz: {clustering_orig_target['calinski_harabasz']:.1f} → {clustering_flow_target['calinski_harabasz']:.1f}")
    print(f"  Davies-Bouldin: {clustering_orig_target['davies_bouldin']:.3f} → {clustering_flow_target['davies_bouldin']:.3f}")

    print("\n3. k-NN Purity (k=10, by target):")
    print(f"  {knn_orig_target:.3f} → {knn_flow_target:.3f}")

    print("\n4. Centroid Distance (to target):")
    print(f"  {centroid_orig_target:.3f} → {centroid_flow_target:.3f}")

    print("\n5. Path Quality:")
    print(f"  Fraction monotonic: {monotonic_frac:.1%}")
    print(f"  Mean cosine similarity: {smoothness['mean_cosine_similarity']:.3f}")
    print(f"  Mean path efficiency: {smoothness['mean_path_efficiency']:.3f}")

    print(f"\n✅ Results saved to: {results_file}")
    print(f"✅ Data saved to: {os.path.join(args.output_dir, 'evaluation_data.npz')}")

    # Save data
    np.savez(os.path.join(args.output_dir, 'evaluation_data.npz'), **data)


if __name__ == '__main__':
    main()
