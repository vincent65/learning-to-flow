"""
Visualization utilities for FCLF.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap
from typing import Optional, List
import os


def plot_embedding_2d(
    embeddings: np.ndarray,
    attributes: np.ndarray,
    method: str = 'umap',
    attribute_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = "Embedding Visualization"
):
    """
    Project embeddings to 2D and visualize colored by attributes.

    Args:
        embeddings: [N, dim] embeddings
        attributes: [N, num_attrs] binary attribute vectors
        method: 'umap' or 'tsne'
        attribute_names: Names of attributes
        save_path: Path to save figure
        title: Plot title
    """
    # Project to 2D
    if method == 'umap':
        reducer = umap.UMAP(random_state=42)
        coords_2d = reducer.fit_transform(embeddings)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
        coords_2d = reducer.fit_transform(embeddings)
    else:
        raise ValueError(f"Unknown method: {method}")

    num_attrs = attributes.shape[1]

    # Create subplots for each attribute
    fig, axes = plt.subplots(1, num_attrs, figsize=(5*num_attrs, 4))
    if num_attrs == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        # Color by this attribute
        colors = attributes[:, i]

        scatter = ax.scatter(
            coords_2d[:, 0],
            coords_2d[:, 1],
            c=colors,
            cmap='RdBu',
            alpha=0.5,
            s=10
        )

        if attribute_names:
            ax.set_title(attribute_names[i])
        else:
            ax.set_title(f'Attribute {i}')

        ax.set_xlabel(f'{method.upper()} 1')
        ax.set_ylabel(f'{method.upper()} 2')
        plt.colorbar(scatter, ax=ax)

    fig.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()


def plot_trajectory_2d(
    trajectories: np.ndarray,
    attributes: np.ndarray,
    method: str = 'umap',
    num_samples: int = 10,
    save_path: Optional[str] = None,
    title: str = "Flow Trajectories"
):
    """
    Visualize flow trajectories in 2D.

    Args:
        trajectories: [N, num_steps, dim] trajectories
        attributes: [N, num_attrs] target attributes
        method: Projection method
        num_samples: Number of trajectories to plot
        save_path: Save path
        title: Plot title
    """
    # Select random samples
    N = trajectories.shape[0]
    indices = np.random.choice(N, min(num_samples, N), replace=False)

    # Flatten trajectories for projection
    # Shape: [num_samples * num_steps, dim]
    num_steps = trajectories.shape[1]
    flat_trajectories = trajectories[indices].reshape(-1, trajectories.shape[2])

    # Project to 2D
    if method == 'umap':
        reducer = umap.UMAP(random_state=42)
        coords_2d = reducer.fit_transform(flat_trajectories)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
        coords_2d = reducer.fit_transform(flat_trajectories)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Reshape back
    coords_2d = coords_2d.reshape(len(indices), num_steps, 2)

    # Plot
    plt.figure(figsize=(10, 8))

    for i, traj in enumerate(coords_2d):
        # Plot trajectory
        plt.plot(traj[:, 0], traj[:, 1], 'o-', alpha=0.6, markersize=4)

        # Mark start and end
        plt.scatter(traj[0, 0], traj[0, 1], c='green', s=100, marker='o', edgecolors='black', zorder=10)
        plt.scatter(traj[-1, 0], traj[-1, 1], c='red', s=100, marker='s', edgecolors='black', zorder=10)

    plt.xlabel(f'{method.upper()} 1')
    plt.ylabel(f'{method.upper()} 2')
    plt.title(title)
    plt.legend(['Start', 'End'], loc='best')
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()


def plot_image_grid(
    images: torch.Tensor,
    nrow: int = 8,
    save_path: Optional[str] = None,
    title: str = "Images"
):
    """
    Plot grid of images.

    Args:
        images: [N, C, H, W] images in [0, 1] range
        nrow: Number of images per row
        save_path: Save path
        title: Title
    """
    from torchvision.utils import make_grid

    grid = make_grid(images, nrow=nrow, padding=2, normalize=False)
    grid_np = grid.permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=(12, 12))
    plt.imshow(grid_np)
    plt.axis('off')
    plt.title(title)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()


def plot_reconstruction_comparison(
    original_images: torch.Tensor,
    reconstructed_images: torch.Tensor,
    num_samples: int = 8,
    save_path: Optional[str] = None
):
    """
    Plot original vs reconstructed images side-by-side.

    Args:
        original_images: [N, C, H, W] original images
        reconstructed_images: [N, C, H, W] reconstructed images
        num_samples: Number of samples to show
        save_path: Save path
    """
    num_samples = min(num_samples, original_images.size(0))

    fig, axes = plt.subplots(2, num_samples, figsize=(2*num_samples, 4))

    for i in range(num_samples):
        # Original
        orig_img = original_images[i].permute(1, 2, 0).cpu().numpy()
        axes[0, i].imshow(orig_img)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=10)

        # Reconstructed
        recon_img = reconstructed_images[i].permute(1, 2, 0).cpu().numpy()
        axes[1, i].imshow(recon_img)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()


def plot_attribute_transfer(
    original_images: torch.Tensor,
    flowed_images: torch.Tensor,
    original_attrs: np.ndarray,
    target_attrs: np.ndarray,
    attribute_names: List[str],
    num_samples: int = 5,
    save_path: Optional[str] = None
):
    """
    Visualize attribute transfer: original -> flowed.

    Args:
        original_images: [N, C, H, W] original images
        flowed_images: [N, C, H, W] images after flow
        original_attrs: [N, num_attrs] original attributes
        target_attrs: [N, num_attrs] target attributes
        attribute_names: Names of attributes
        num_samples: Number of samples
        save_path: Save path
    """
    num_samples = min(num_samples, original_images.size(0))

    fig, axes = plt.subplots(num_samples, 2, figsize=(6, 3*num_samples))

    for i in range(num_samples):
        # Original
        orig_img = original_images[i].permute(1, 2, 0).cpu().numpy()
        axes[i, 0].imshow(orig_img)
        axes[i, 0].axis('off')

        # Format attribute labels
        orig_label = ', '.join([
            f"{name}:{int(val)}"
            for name, val in zip(attribute_names, original_attrs[i])
        ])
        axes[i, 0].set_title(f'Original\n{orig_label}', fontsize=8)

        # Flowed
        flowed_img = flowed_images[i].permute(1, 2, 0).cpu().numpy()
        axes[i, 1].imshow(flowed_img)
        axes[i, 1].axis('off')

        target_label = ', '.join([
            f"{name}:{int(val)}"
            for name, val in zip(attribute_names, target_attrs[i])
        ])
        axes[i, 1].set_title(f'Flowed\n{target_label}', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()


def plot_metrics_comparison(
    metrics_dict: dict,
    save_path: Optional[str] = None
):
    """
    Plot comparison of metrics (e.g., before/after flow).

    Args:
        metrics_dict: Dictionary of metrics
        save_path: Save path
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Silhouette scores
    if 'silhouette_original' in metrics_dict and 'silhouette_flowed' in metrics_dict:
        sil_orig = metrics_dict['silhouette_original']
        sil_flow = metrics_dict['silhouette_flowed']

        # Extract overall scores
        labels = ['Original', 'Flowed']
        overall_scores = [sil_orig.get('overall', 0), sil_flow.get('overall', 0)]

        axes[0].bar(labels, overall_scores, color=['blue', 'orange'])
        axes[0].set_ylabel('Silhouette Score')
        axes[0].set_title('Overall Silhouette Score')
        axes[0].set_ylim([0, 1])

    # Cluster purity
    if 'purity_original' in metrics_dict and 'purity_flowed' in metrics_dict:
        labels = ['Original', 'Flowed']
        purity_scores = [
            metrics_dict['purity_original'],
            metrics_dict['purity_flowed']
        ]

        axes[1].bar(labels, purity_scores, color=['blue', 'orange'])
        axes[1].set_ylabel('Cluster Purity')
        axes[1].set_title('Cluster Purity')
        axes[1].set_ylim([0, 1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()
