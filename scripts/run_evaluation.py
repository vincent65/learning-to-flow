#!/usr/bin/env python3
"""
Comprehensive evaluation script for FCLF.
"""

import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.vector_field import VectorFieldNetwork
from src.data.celeba_dataset import get_dataloader
from src.evaluation.metrics import (
    compute_silhouette_score,
    compute_cluster_purity,
    compute_trajectory_smoothness
)
from src.evaluation.visualize import (
    plot_embedding_2d,
    plot_trajectory_2d,
    plot_metrics_comparison
)


def evaluate_fclf(
    checkpoint_path: str,
    celeba_root: str,
    embedding_dir: str,
    output_dir: str,
    num_samples: int = 1000,
    device: str = None
):
    """
    Run comprehensive evaluation on trained FCLF model.

    Args:
        checkpoint_path: Path to FCLF checkpoint
        celeba_root: Path to CelebA dataset
        embedding_dir: Path to embeddings
        output_dir: Output directory for results
        num_samples: Number of samples to evaluate
        device: Device to use
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    # Load model
    model = VectorFieldNetwork(
        embedding_dim=config['model']['embedding_dim'],
        num_attributes=config['model']['num_attributes'],
        hidden_dim=config['model']['hidden_dim']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Load data
    print("Loading data...")
    test_loader = get_dataloader(
        root_dir=celeba_root,
        split='test',
        batch_size=128,
        embedding_path=os.path.join(embedding_dir, 'test_embeddings.pt'),
        load_images=False,
        num_workers=4,
        shuffle=False
    )

    # Collect embeddings and attributes
    print("Collecting embeddings...")
    original_embeddings = []
    flowed_embeddings = []
    attributes_list = []
    trajectories_list = []

    count = 0
    with torch.no_grad():
        for batch in tqdm(test_loader):
            if count >= num_samples:
                break

            embeddings = batch['embedding'].to(device)
            attributes = batch['attributes'].to(device)

            # Flow embeddings
            trajectory = model.get_trajectory(
                embeddings,
                attributes,
                num_steps=10,
                step_size=0.1
            )

            original_embeddings.append(embeddings.cpu().numpy())
            flowed_embeddings.append(trajectory[:, -1, :].cpu().numpy())
            trajectories_list.append(trajectory.cpu().numpy())
            attributes_list.append(attributes.cpu().numpy())

            count += embeddings.size(0)

    # Concatenate
    original_embeddings = np.concatenate(original_embeddings, axis=0)[:num_samples]
    flowed_embeddings = np.concatenate(flowed_embeddings, axis=0)[:num_samples]
    trajectories = np.concatenate(trajectories_list, axis=0)[:num_samples]
    attributes = np.concatenate(attributes_list, axis=0)[:num_samples]

    print(f"Collected {len(original_embeddings)} samples")

    # Compute metrics
    print("\n" + "=" * 60)
    print("Computing Metrics")
    print("=" * 60)

    results = {}

    # Silhouette scores
    print("\nSilhouette Scores:")
    sil_original = compute_silhouette_score(original_embeddings, attributes)
    sil_flowed = compute_silhouette_score(flowed_embeddings, attributes)

    print(f"  Original: {sil_original['overall']:.4f}")
    print(f"  Flowed: {sil_flowed['overall']:.4f}")

    results['silhouette_original'] = sil_original
    results['silhouette_flowed'] = sil_flowed

    # Cluster purity
    print("\nCluster Purity:")
    purity_original = compute_cluster_purity(original_embeddings, attributes)
    purity_flowed = compute_cluster_purity(flowed_embeddings, attributes)

    print(f"  Original: {purity_original:.4f}")
    print(f"  Flowed: {purity_flowed:.4f}")

    results['purity_original'] = purity_original
    results['purity_flowed'] = purity_flowed

    # Trajectory smoothness
    print("\nTrajectory Smoothness:")
    smoothness = compute_trajectory_smoothness(trajectories)

    print(f"  Mean step distance: {smoothness['mean_step_distance']:.4f}")
    print(f"  Std step distance: {smoothness['std_step_distance']:.4f}")

    results['smoothness'] = smoothness

    # Save metrics
    metrics_file = os.path.join(output_dir, 'metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics to {metrics_file}")

    # Generate visualizations
    print("\n" + "=" * 60)
    print("Generating Visualizations")
    print("=" * 60)

    attribute_names = ['Smiling', 'Young', 'Male', 'Eyeglasses', 'Mustache']

    # Embedding visualizations
    print("\nOriginal embeddings (UMAP)...")
    plot_embedding_2d(
        original_embeddings[:500],
        attributes[:500],
        method='umap',
        attribute_names=attribute_names,
        save_path=os.path.join(figures_dir, 'embeddings_original_umap.png'),
        title='Original Embeddings (UMAP)'
    )

    print("Flowed embeddings (UMAP)...")
    plot_embedding_2d(
        flowed_embeddings[:500],
        attributes[:500],
        method='umap',
        attribute_names=attribute_names,
        save_path=os.path.join(figures_dir, 'embeddings_flowed_umap.png'),
        title='Flowed Embeddings (UMAP)'
    )

    # Trajectory visualization
    print("Flow trajectories...")
    plot_trajectory_2d(
        trajectories[:50],
        attributes[:50],
        method='umap',
        num_samples=20,
        save_path=os.path.join(figures_dir, 'trajectories_umap.png'),
        title='Flow Trajectories (UMAP)'
    )

    # Metrics comparison
    print("Metrics comparison...")
    plot_metrics_comparison(
        results,
        save_path=os.path.join(figures_dir, 'metrics_comparison.png')
    )

    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate FCLF model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to FCLF checkpoint")
    parser.add_argument("--celeba_root", type=str, default="data/celeba",
                        help="Path to CelebA root")
    parser.add_argument("--embedding_dir", type=str, default="data/embeddings",
                        help="Path to embeddings")
    parser.add_argument("--output_dir", type=str, default="results/evaluation",
                        help="Output directory")
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Number of samples to evaluate")
    parser.add_argument("--device", type=str, default=None,
                        help="Device")

    args = parser.parse_args()

    evaluate_fclf(
        checkpoint_path=args.checkpoint,
        celeba_root=args.celeba_root,
        embedding_dir=args.embedding_dir,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        device=args.device
    )
