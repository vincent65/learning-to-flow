"""
Projection utilities for FCLF.

Key insight from cs229.ipynb: Projecting embeddings to a fixed-radius hypersphere
after every flow step prevents latent blow-up and mode collapse.

For CLIP embeddings, we use radius=1.0 (unit sphere) since CLIP embeddings
are already unit-normalized.
"""

import torch
import torch.nn.functional as F


def project_to_sphere(z: torch.Tensor, radius: float = 1.0) -> torch.Tensor:
    """
    Project embeddings to a hypersphere of given radius.

    This prevents latent blow-up during flow and provides implicit regularization.
    Inspired by cs229.ipynb implementation.

    Args:
        z: [batch, dim] embeddings
        radius: Radius of hypersphere (default 1.0 for CLIP)

    Returns:
        z_proj: [batch, dim] projected embeddings with ||z_proj|| = radius

    Example:
        >>> z = torch.randn(32, 512)
        >>> z_proj = project_to_sphere(z, radius=1.0)
        >>> assert torch.allclose(z_proj.norm(dim=1), torch.ones(32))
    """
    # Compute norms
    norms = z.norm(dim=1, keepdim=True) + 1e-8

    # Project to sphere
    z_proj = radius * z / norms

    return z_proj


def normalize_embeddings(z: torch.Tensor) -> torch.Tensor:
    """
    L2-normalize embeddings (equivalent to project_to_sphere with radius=1.0).

    This is an alias for clarity when we specifically want unit normalization.

    Args:
        z: [batch, dim] embeddings

    Returns:
        z_norm: [batch, dim] unit-normalized embeddings
    """
    return F.normalize(z, dim=1)
