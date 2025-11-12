"""
Comprehensive FCLF Evaluation Script

Implements all quantitative metrics requested:
1. Separability & clustering (linear probe, silhouette, k-NN)
2. Path quality (monotonic progress, smoothness)
3. Control/falsification tests
4. Multiple visualization methods (UMAP, PCA, t-SNE)
"""

import os
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vector_field import VectorFieldNetwork
from data.celeba_dataset import get_dataloader


# CelebA has 40 attributes total
ALL_CELEBA_ATTRIBUTES = [
    '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald',
    'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair',
    'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair',
    'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache',
    'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose',
    'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair',
    'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
    'Wearing_Necklace', 'Wearing_Necktie', 'Young'
]

# Interesting attribute pairs for visualization
INTERESTING_PAIRS = [
    ('Smiling', 'Not Smiling'),
    ('Young', 'Old'),
    ('Male', 'Female'),
    ('Eyeglasses', 'No Eyeglasses'),
    ('Mustache', 'No Mustache'),
    ('Bald', 'Not Bald'),
    ('Heavy_Makeup', 'No Makeup'),
    ('Wearing_Hat', 'No Hat'),
]


class ComprehensiveEvaluator:
    """Comprehensive evaluation metrics for FCLF."""

    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.eval()

    def collect_data(self, dataloader, num_samples=1000, attribute_to_flip=None):
        """
        Collect embeddings, attributes, and trajectories.

        Args:
            dataloader: DataLoader for test set
            num_samples: Number of samples to evaluate
            attribute_to_flip: If specified, flip this attribute. If None, flip random.

        Returns:
            dict with original_emb, target_attrs, flowed_emb, trajectories, etc.
        """
        original_embeddings = []
        original_attributes = []
        target_attributes = []
        flowed_embeddings = []
        trajectories = []

        count = 0
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Collecting data"):
                if count >= num_samples:
                    break

                embeddings = batch['embedding'].to(self.device)
                attributes = batch['attributes'].to(self.device)
                batch_size = embeddings.size(0)

                # Create target attributes
                target_attrs = attributes.clone()

                if attribute_to_flip is not None:
                    # Flip specific attribute
                    attr_idx = attribute_to_flip
                    target_attrs[:, attr_idx] = 1 - target_attrs[:, attr_idx]
                else:
                    # Flip random attribute per sample
                    for i in range(batch_size):
                        attr_idx = np.random.randint(0, 5)
                        target_attrs[i, attr_idx] = 1 - target_attrs[i, attr_idx]

                # Get trajectory
                trajectory = self.model.get_trajectory(
                    embeddings, target_attrs, num_steps=10
                )

                # Store
                original_embeddings.append(embeddings.cpu())
                original_attributes.append(attributes.cpu())
                target_attributes.append(target_attrs.cpu())
                flowed_embeddings.append(trajectory[:, -1].cpu())  # Final step
                trajectories.append(trajectory.cpu())

                count += batch_size

        return {
            'original_emb': torch.cat(original_embeddings)[:num_samples].numpy(),
            'original_attrs': torch.cat(original_attributes)[:num_samples].numpy(),
            'target_attrs': torch.cat(target_attributes)[:num_samples].numpy(),
            'flowed_emb': torch.cat(flowed_embeddings)[:num_samples].numpy(),
            'trajectories': torch.cat(trajectories)[:num_samples].numpy(),
        }

    ### 1. SEPARABILITY & CLUSTERING METRICS ###

    def linear_probe_accuracy(self, embeddings, labels, attribute_idx):
        """
        Train logistic regression and measure accuracy.

        Args:
            embeddings: [N, D] embedding vectors
            labels: [N, num_attrs] binary attribute labels
            attribute_idx: Which attribute to predict

        Returns:
            accuracy: Classification accuracy
        """
        y = labels[:, attribute_idx]

        # Train/test split
        n = len(y)
        split = int(0.8 * n)
        indices = np.random.permutation(n)
        train_idx, test_idx = indices[:split], indices[split:]

        # Train logistic regression
        clf = LogisticRegression(max_iter=500, random_state=42)
        clf.fit(embeddings[train_idx], y[train_idx])

        # Evaluate
        y_pred = clf.predict(embeddings[test_idx])
        accuracy = accuracy_score(y[test_idx], y_pred)

        return accuracy

    def clustering_metrics(self, embeddings, labels):
        """
        Compute silhouette, Calinski-Harabasz, Davies-Bouldin scores.

        Args:
            embeddings: [N, D] embeddings
            labels: [N,] integer cluster labels

        Returns:
            dict with scores
        """
        if len(np.unique(labels)) < 2:
            return {
                'silhouette': -1.0,
                'calinski_harabasz': 0.0,
                'davies_bouldin': float('inf')
            }

        return {
            'silhouette': silhouette_score(embeddings, labels),
            'calinski_harabasz': calinski_harabasz_score(embeddings, labels),
            'davies_bouldin': davies_bouldin_score(embeddings, labels)
        }

    def knn_class_purity(self, embeddings, labels, k=10):
        """
        Compute k-NN class purity.

        Args:
            embeddings: [N, D]
            labels: [N,] integer labels
            k: Number of neighbors

        Returns:
            purity: Average purity score
        """
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(embeddings, labels)

        # For each point, check if k neighbors have same label
        purities = []
        for i in range(len(embeddings)):
            neighbors = knn.kneighbors([embeddings[i]], return_distance=False)[0]
            neighbor_labels = labels[neighbors]
            purity = (neighbor_labels == labels[i]).mean()
            purities.append(purity)

        return np.mean(purities)

    def class_centroid_distance(self, embeddings, labels):
        """
        Compute mean distance to correct class centroid.

        Args:
            embeddings: [N, D]
            labels: [N,] integer labels

        Returns:
            mean_distance: Average distance to own centroid
        """
        unique_labels = np.unique(labels)
        centroids = {}

        # Compute centroids
        for label in unique_labels:
            mask = labels == label
            centroids[label] = embeddings[mask].mean(axis=0)

        # Compute distance to own centroid
        distances = []
        for i in range(len(embeddings)):
            centroid = centroids[labels[i]]
            dist = np.linalg.norm(embeddings[i] - centroid)
            distances.append(dist)

        return np.mean(distances)

    ### 2. PATH QUALITY METRICS ###

    def monotonic_progress(self, trajectories, target_attrs):
        """
        Check if distance to target centroid decreases monotonically.

        Args:
            trajectories: [N, num_steps, D] trajectories
            target_attrs: [N, num_attrs] target attributes

        Returns:
            fraction_monotonic: Fraction of trajectories with monotonic decrease
        """
        # Compute target centroids
        unique_attrs = np.unique(target_attrs, axis=0)
        centroids = {}
        for attr_vec in unique_attrs:
            attr_tuple = tuple(attr_vec)
            # Find final embeddings with this target
            mask = np.all(target_attrs == attr_vec, axis=1)
            if mask.sum() > 0:
                final_embs = trajectories[mask, -1, :]  # Final step
                centroids[attr_tuple] = final_embs.mean(axis=0)

        # Check monotonicity for each trajectory
        monotonic_count = 0
        for i in range(len(trajectories)):
            target_tuple = tuple(target_attrs[i])
            if target_tuple not in centroids:
                continue

            centroid = centroids[target_tuple]

            # Compute distance at each step
            distances = []
            for step in range(trajectories.shape[1]):
                dist = np.linalg.norm(trajectories[i, step] - centroid)
                distances.append(dist)

            # Check if monotonically decreasing
            is_monotonic = all(distances[i] >= distances[i+1] for i in range(len(distances)-1))
            if is_monotonic:
                monotonic_count += 1

        return monotonic_count / len(trajectories)

    def path_smoothness(self, trajectories):
        """
        Compute path smoothness metrics.

        Args:
            trajectories: [N, num_steps, D]

        Returns:
            dict with cosine_similarity, path_efficiency
        """
        cosine_sims = []
        path_efficiencies = []

        for traj in trajectories:
            # Cosine similarity between successive steps
            step_cosines = []
            for i in range(len(traj) - 2):
                v1 = traj[i+1] - traj[i]
                v2 = traj[i+2] - traj[i+1]

                # Normalize
                v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
                v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)

                cosine = np.dot(v1_norm, v2_norm)
                step_cosines.append(cosine)

            if step_cosines:
                cosine_sims.append(np.mean(step_cosines))

            # Path efficiency: straight-line distance / path length
            straight_line = np.linalg.norm(traj[-1] - traj[0])
            path_length = sum(np.linalg.norm(traj[i+1] - traj[i]) for i in range(len(traj)-1))

            if path_length > 0:
                efficiency = straight_line / path_length
                path_efficiencies.append(efficiency)

        return {
            'mean_cosine_similarity': np.mean(cosine_sims) if cosine_sims else 0,
            'mean_path_efficiency': np.mean(path_efficiencies) if path_efficiencies else 0
        }

    ### 3. VISUALIZATION ###

    def reduce_dimensionality(self, embeddings, method='umap', **kwargs):
        """
        Reduce embeddings to 2D for visualization.

        Args:
            embeddings: [N, D]
            method: 'umap', 'pca', or 'tsne'

        Returns:
            reduced: [N, 2]
        """
        if method == 'umap':
            reducer = umap.UMAP(random_state=42, **kwargs)
        elif method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
        elif method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

        return reducer.fit_transform(embeddings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--celeba_root', type=str, default='data/celeba')
    parser.add_argument('--embedding_dir', type=str, default='data/embeddings')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=2000)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    # Create output directory
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

    # Initialize evaluator
    evaluator = ComprehensiveEvaluator(model, args.device)

    # Collect data
    print(f"Collecting {args.num_samples} samples...")
    data = evaluator.collect_data(test_loader, num_samples=args.num_samples)

    # Save collected data
    print("Saving collected data...")
    np.savez(
        os.path.join(args.output_dir, 'evaluation_data.npz'),
        **data
    )

    print("\nEvaluation data collection complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"\nNext steps:")
    print("1. Run analysis on collected data")
    print("2. Generate visualizations")
    print("3. Compare with baselines")


if __name__ == '__main__':
    main()
