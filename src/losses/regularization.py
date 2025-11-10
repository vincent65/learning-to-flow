"""
Regularization terms for vector field: curl and divergence penalties.
"""

import torch
import torch.nn as nn


class CurlRegularization(nn.Module):
    """
    Penalize curl (rotation) in the vector field.

    R_curl = ||∇ × v||²

    For high-dimensional spaces, we estimate using random 2D plane samples.
    """

    def __init__(self, epsilon: float = 1e-4, num_samples: int = 10):
        """
        Args:
            epsilon: Step size for finite differences
            num_samples: Number of random planes to sample
        """
        super().__init__()
        self.epsilon = epsilon
        self.num_samples = num_samples

    def forward(
        self,
        vector_field,
        z: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute curl regularization.

        Args:
            vector_field: VectorFieldNetwork instance
            z: [batch, dim] embeddings
            y: [batch, num_attrs] attributes

        Returns:
            curl_loss: scalar regularization term
        """
        curl_mag = vector_field.compute_curl_magnitude(
            z, y,
            epsilon=self.epsilon,
            num_samples=self.num_samples
        )

        # L2 penalty on curl magnitude
        return (curl_mag ** 2).mean()


class DivergenceRegularization(nn.Module):
    """
    Penalize divergence (expansion/contraction) in the vector field.

    R_div = (∇ · v)²

    Encourages volume-preserving (incompressible) flows.
    """

    def __init__(self, epsilon: float = 1e-4):
        """
        Args:
            epsilon: Step size for finite differences
        """
        super().__init__()
        self.epsilon = epsilon

    def forward(
        self,
        vector_field,
        z: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute divergence regularization.

        Args:
            vector_field: VectorFieldNetwork instance
            z: [batch, dim] embeddings
            y: [batch, num_attrs] attributes

        Returns:
            div_loss: scalar regularization term
        """
        divergence = vector_field.compute_divergence(z, y, epsilon=self.epsilon)

        # L2 penalty on divergence
        return (divergence ** 2).mean()


class MagnitudeRegularization(nn.Module):
    """
    Penalize large magnitude flows (optional).

    Prevents the vector field from producing very large velocities.
    """

    def __init__(self):
        super().__init__()

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """
        Compute magnitude regularization.

        Args:
            v: [batch, dim] velocity vectors

        Returns:
            mag_loss: scalar regularization term
        """
        # L2 norm of velocities
        magnitudes = torch.norm(v, dim=1)
        return (magnitudes ** 2).mean()


class SmoothnessRegularization(nn.Module):
    """
    Encourage smooth vector field (optional).

    Penalizes large changes in velocity for nearby embeddings.
    """

    def __init__(self, epsilon: float = 1e-3):
        """
        Args:
            epsilon: Perturbation size for smoothness check
        """
        super().__init__()
        self.epsilon = epsilon

    def forward(
        self,
        vector_field,
        z: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute smoothness regularization.

        Args:
            vector_field: VectorFieldNetwork instance
            z: [batch, dim] embeddings
            y: [batch, num_attrs] attributes

        Returns:
            smooth_loss: scalar regularization term
        """
        # Compute velocity at current point
        v_original = vector_field(z, y)

        # Random perturbation
        noise = torch.randn_like(z) * self.epsilon
        z_perturbed = z + noise

        # Compute velocity at perturbed point
        v_perturbed = vector_field(z_perturbed, y)

        # Penalize difference
        diff = v_perturbed - v_original
        return (diff ** 2).mean()
