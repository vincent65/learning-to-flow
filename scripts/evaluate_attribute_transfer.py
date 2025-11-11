#!/usr/bin/env python3
"""
Attribute Transfer Evaluation for FCLF.

Tests the model's ability to CHANGE attributes, not just flow to existing ones.
For each sample, we flip one or more attributes and see if the flowed embedding
matches the new target attributes.
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


def flip_attributes(attributes: np.ndarray, attr_indices: list) -> np.ndarray:
    """
    Flip specified attributes.

    Args:
        attributes: [N, num_attrs] binary attribute array
        attr_indices: List of attribute indices to flip

    Returns:
        flipped: [N, num_attrs] with specified attributes flipped
    """
    flipped = attributes.copy()
    for idx in attr_indices:
        flipped[:, idx] = 1 - flipped[:, idx]
    return flipped


def evaluate_attribute_transfer(
    checkpoint_path: str,
    celeba_root: str,
    embedding_dir: str,
    output_dir: str,
    num_samples: int = 1000,
    split: str = 'test',
    transfer_mode: str = 'single',
    device: str = None
):
    """
    Evaluate FCLF model on attribute transfer task.

    Args:
        checkpoint_path: Path to FCLF checkpoint
        celeba_root: Path to CelebA dataset
        embedding_dir: Path to embeddings
        output_dir: Output directory for results
        num_samples: Number of samples to evaluate
        split: Data split to use ('train', 'val', or 'test')
        transfer_mode: 'single' (flip one attribute) or 'multi' (flip multiple)
        device: Device to use
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {device}")
    print(f"Transfer mode: {transfer_mode}")

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
    print(f"Loading data from {split} split...")
    data_loader = get_dataloader(
        root_dir=celeba_root,
        split=split,
        batch_size=128,
        embedding_path=os.path.join(embedding_dir, f'{split}_embeddings.pt'),
        load_images=False,
        num_workers=0,
        shuffle=False
    )

    # Collect embeddings and perform transfers
    print(f"Performing attribute transfer on {num_samples} samples...")

    original_embeddings = []
    original_attributes = []
    target_attributes = []
    flowed_embeddings = []
    trajectories_list = []

    attribute_names = ['Smiling', 'Young', 'Male', 'Eyeglasses', 'Mustache']

    count = 0
    with torch.no_grad():
        for batch in tqdm(data_loader):
            if count >= num_samples:
                break

            embeddings = batch['embedding'].to(device)
            attributes = batch['attributes'].to(device)
            batch_size = embeddings.size(0)

            # Determine which attributes to flip
            if transfer_mode == 'single':
                # Flip one random attribute per sample
                attr_to_flip = np.random.randint(0, 5, size=batch_size)
                target_attrs = attributes.clone()
                for i, attr_idx in enumerate(attr_to_flip):
                    target_attrs[i, attr_idx] = 1 - target_attrs[i, attr_idx]

            elif transfer_mode == 'multi':
                # Flip 2-3 random attributes per sample
                target_attrs = attributes.clone()
                for i in range(batch_size):
                    num_flips = np.random.randint(2, 4)
                    attrs_to_flip = np.random.choice(5, size=num_flips, replace=False)
                    for attr_idx in attrs_to_flip:
                        target_attrs[i, attr_idx] = 1 - target_attrs[i, attr_idx]

            elif transfer_mode == 'smiling':
                # Only flip smiling attribute
                target_attrs = attributes.clone()
                target_attrs[:, 0] = 1 - target_attrs[:, 0]

            else:
                raise ValueError(f"Unknown transfer mode: {transfer_mode}")

            # Flow embeddings toward target attributes
            trajectory = model.get_trajectory(
                embeddings,
                target_attrs,
                num_steps=10,
                step_size=0.1
            )

            original_embeddings.append(embeddings.cpu().numpy())
            original_attributes.append(attributes.cpu().numpy())
            target_attributes.append(target_attrs.cpu().numpy())
            flowed_embeddings.append(trajectory[:, -1, :].cpu().numpy())
            trajectories_list.append(trajectory.cpu().numpy())

            count += batch_size

    # Concatenate all results
    original_embeddings = np.concatenate(original_embeddings, axis=0)[:num_samples]
    original_attributes = np.concatenate(original_attributes, axis=0)[:num_samples]
    target_attributes = np.concatenate(target_attributes, axis=0)[:num_samples]
    flowed_embeddings = np.concatenate(flowed_embeddings, axis=0)[:num_samples]
    trajectories = np.concatenate(trajectories_list, axis=0)[:num_samples]

    print(f"\nCollected {len(original_embeddings)} samples")
    print(f"Attribute changes made: {(original_attributes != target_attributes).sum()} total flips")

    # Compute metrics
    print("\n" + "=" * 60)
    print("Computing Metrics")
    print("=" * 60)

    results = {}

    # Silhouette scores
    print("\nSilhouette Scores (clustering by attributes):")
    sil_original_by_original = compute_silhouette_score(original_embeddings, original_attributes)
    sil_original_by_target = compute_silhouette_score(original_embeddings, target_attributes)
    sil_flowed_by_original = compute_silhouette_score(flowed_embeddings, original_attributes)
    sil_flowed_by_target = compute_silhouette_score(flowed_embeddings, target_attributes)

    print(f"  Original embeddings clustered by original attrs: {sil_original_by_original['overall']:.4f}")
    print(f"  Original embeddings clustered by target attrs:   {sil_original_by_target['overall']:.4f}")
    print(f"  Flowed embeddings clustered by original attrs:   {sil_flowed_by_original['overall']:.4f}")
    print(f"  Flowed embeddings clustered by target attrs:     {sil_flowed_by_target['overall']:.4f}")

    print(f"\n  ✓ Success if flowed_by_target > flowed_by_original")
    print(f"    (Flowed embeddings should match TARGET attributes, not original)")

    results['silhouette_original_by_original'] = sil_original_by_original
    results['silhouette_original_by_target'] = sil_original_by_target
    results['silhouette_flowed_by_original'] = sil_flowed_by_original
    results['silhouette_flowed_by_target'] = sil_flowed_by_target

    # Cluster purity
    print("\nCluster Purity:")
    purity_original_by_original = compute_cluster_purity(original_embeddings, original_attributes)
    purity_flowed_by_target = compute_cluster_purity(flowed_embeddings, target_attributes)

    print(f"  Original embeddings (by original attrs): {purity_original_by_original:.4f}")
    print(f"  Flowed embeddings (by target attrs):     {purity_flowed_by_target:.4f}")
    print(f"\n  ✓ Success if flowed > original")

    results['purity_original'] = purity_original_by_original
    results['purity_flowed'] = purity_flowed_by_target

    # Trajectory smoothness
    print("\nTrajectory Smoothness:")
    smoothness = compute_trajectory_smoothness(trajectories)

    print(f"  Mean step distance: {smoothness['mean_step_distance']:.4f}")
    print(f"  Std step distance:  {smoothness['std_step_distance']:.4f}")
    print(f"\n  ✓ Good values: mean ~0.02-0.05, std <0.02")

    results['smoothness'] = smoothness

    # Movement magnitude (how much embeddings moved)
    movement = np.linalg.norm(flowed_embeddings - original_embeddings, axis=1)
    results['movement'] = {
        'mean': float(movement.mean()),
        'std': float(movement.std()),
        'max': float(movement.max()),
        'min': float(movement.min())
    }

    print("\nEmbedding Movement:")
    print(f"  Mean distance moved: {movement.mean():.4f}")
    print(f"  Std distance moved:  {movement.std():.4f}")

    # Save metrics
    metrics_file = os.path.join(output_dir, 'transfer_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics to {metrics_file}")

    # Generate visualizations
    print("\n" + "=" * 60)
    print("Generating Visualizations")
    print("=" * 60)

    # Original embeddings colored by original attributes
    print("\nOriginal embeddings (colored by original attributes)...")
    plot_embedding_2d(
        original_embeddings[:500],
        original_attributes[:500],
        method='umap',
        attribute_names=attribute_names,
        save_path=os.path.join(figures_dir, 'original_by_original_attrs.png'),
        title='Original Embeddings (by Original Attributes)'
    )

    # Original embeddings colored by TARGET attributes (should look random)
    print("Original embeddings (colored by TARGET attributes)...")
    plot_embedding_2d(
        original_embeddings[:500],
        target_attributes[:500],
        method='umap',
        attribute_names=attribute_names,
        save_path=os.path.join(figures_dir, 'original_by_target_attrs.png'),
        title='Original Embeddings (by Target Attributes - should be random)'
    )

    # Flowed embeddings colored by original attributes (should become less clustered)
    print("Flowed embeddings (colored by original attributes)...")
    plot_embedding_2d(
        flowed_embeddings[:500],
        original_attributes[:500],
        method='umap',
        attribute_names=attribute_names,
        save_path=os.path.join(figures_dir, 'flowed_by_original_attrs.png'),
        title='Flowed Embeddings (by Original Attributes - should be less clustered)'
    )

    # Flowed embeddings colored by TARGET attributes (should be well-clustered)
    print("Flowed embeddings (colored by target attributes)...")
    plot_embedding_2d(
        flowed_embeddings[:500],
        target_attributes[:500],
        method='umap',
        attribute_names=attribute_names,
        save_path=os.path.join(figures_dir, 'flowed_by_target_attrs.png'),
        title='Flowed Embeddings (by Target Attributes - should be well-clustered)'
    )

    # Trajectory visualization
    print("Flow trajectories...")
    plot_trajectory_2d(
        trajectories[:50],
        target_attributes[:50],
        method='umap',
        num_samples=20,
        save_path=os.path.join(figures_dir, 'transfer_trajectories.png'),
        title='Attribute Transfer Trajectories'
    )

    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")

    print("\n" + "=" * 60)
    print("Interpretation Guide:")
    print("=" * 60)
    print("✓ WORKING MODEL:")
    print("  - Flowed embeddings cluster well by TARGET attributes")
    print("  - Silhouette score: flowed_by_target > flowed_by_original")
    print("  - Trajectories are smooth and consistent")
    print()
    print("✗ NOT WORKING:")
    print("  - Flowed embeddings still cluster by ORIGINAL attributes")
    print("  - Mode collapse: all embeddings go to same points")
    print("  - Erratic, zigzagging trajectories")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate FCLF attribute transfer")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to FCLF checkpoint")
    parser.add_argument("--celeba_root", type=str, default="data/celeba",
                        help="Path to CelebA root")
    parser.add_argument("--embedding_dir", type=str, default="data/embeddings",
                        help="Path to embeddings")
    parser.add_argument("--output_dir", type=str, default="results/transfer_evaluation",
                        help="Output directory")
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Number of samples to evaluate")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"],
                        help="Data split to evaluate on")
    parser.add_argument("--transfer_mode", type=str, default="single",
                        choices=["single", "multi", "smiling"],
                        help="Attribute transfer mode: single (1 attr), multi (2-3 attrs), smiling (only smiling)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device")

    args = parser.parse_args()

    evaluate_attribute_transfer(
        checkpoint_path=args.checkpoint,
        celeba_root=args.celeba_root,
        embedding_dir=args.embedding_dir,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        split=args.split,
        transfer_mode=args.transfer_mode,
        device=args.device
    )
