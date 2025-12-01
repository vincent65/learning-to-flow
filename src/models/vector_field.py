"""
Vector Field Network for FCLF.
Learns function-conditioned vector field v_ω(z, y) in CLIP embedding space.

Key insight from cs229.ipynb: Project embeddings to unit sphere after every
flow step to prevent latent blow-up and mode collapse.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.projection import project_to_sphere


class VectorFieldNetwork(nn.Module):
    """
    Function-conditioned vector field network.

    Maps (embedding, attributes) -> velocity vector in embedding space.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        num_attributes: int = 5,
        hidden_dim: int = 256,
        projection_radius: float = 1.0
    ):
        """
        Args:
            embedding_dim: Dimension of CLIP embeddings
            num_attributes: Number of binary attributes
            hidden_dim: Hidden layer dimension
            projection_radius: Radius for sphere projection (1.0 for CLIP)
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_attributes = num_attributes
        self.hidden_dim = hidden_dim
        self.projection_radius = projection_radius

        # Network architecture
        input_dim = embedding_dim + num_attributes

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, embedding_dim)
        )

        # Initialize final layer with small weights for stability
        nn.init.normal_(self.network[-1].weight, std=0.01)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute velocity field at given embedding and attribute.

        Args:
            z: [batch, embedding_dim] CLIP embeddings
            y: [batch, num_attributes] attribute vectors

        Returns:
            v: [batch, embedding_dim] velocity vectors
        """
        # Concatenate embedding and attributes
        x = torch.cat([z, y], dim=-1)

        # Compute velocity
        v = self.network(x)

        return v

    def flow(
        self,
        z: torch.Tensor,
        y: torch.Tensor,
        num_steps: int = 10,
        step_size: float = 0.1,
        method: str = 'euler'
    ) -> torch.Tensor:
        """
        Integrate flow from initial embedding to final embedding.

        Args:
            z: [batch, embedding_dim] initial embeddings
            y: [batch, num_attributes] target attributes
            num_steps: Number of integration steps
            step_size: Step size for integration
            method: Integration method ('euler' or 'rk4')

        Returns:
            z_final: [batch, embedding_dim] final embeddings
        """
        z_current = z

        for _ in range(num_steps):
            if method == 'euler':
                z_current = self._euler_step(z_current, y, step_size)
            elif method == 'rk4':
                z_current = self._rk4_step(z_current, y, step_size)
            else:
                raise ValueError(f"Unknown integration method: {method}")

        return z_current

    def _euler_step(
        self,
        z: torch.Tensor,
        y: torch.Tensor,
        step_size: float
    ) -> torch.Tensor:
        """
        Single Euler integration step with projection.

        Following cs229.ipynb: project to sphere after each step to prevent blow-up.
        """
        v = self.forward(z, y)
        z_next = z + step_size * v
        # CRITICAL: Project to unit sphere (cs229.ipynb key insight)
        z_next = project_to_sphere(z_next, radius=self.projection_radius)
        return z_next

    def _rk4_step(
        self,
        z: torch.Tensor,
        y: torch.Tensor,
        step_size: float
    ) -> torch.Tensor:
        """
        Single RK4 integration step with projection.

        Following cs229.ipynb: project to sphere after each step to prevent blow-up.
        """
        k1 = self.forward(z, y)
        k2 = self.forward(z + 0.5 * step_size * k1, y)
        k3 = self.forward(z + 0.5 * step_size * k2, y)
        k4 = self.forward(z + step_size * k3, y)

        z_next = z + (step_size / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        # CRITICAL: Project to unit sphere (cs229.ipynb key insight)
        z_next = project_to_sphere(z_next, radius=self.projection_radius)
        return z_next

    def get_trajectory(
        self,
        z: torch.Tensor,
        y: torch.Tensor,
        num_steps: int = 10,
        step_size: float = 0.1,
        method: str = 'euler'
    ) -> torch.Tensor:
        """
        Get full trajectory of embeddings during flow.

        Args:
            z: [batch, embedding_dim] initial embeddings
            y: [batch, num_attributes] target attributes
            num_steps: Number of integration steps
            step_size: Step size
            method: Integration method

        Returns:
            trajectory: [batch, num_steps+1, embedding_dim] full trajectory
        """
        trajectory = [z]
        z_current = z

        for _ in range(num_steps):
            if method == 'euler':
                z_current = self._euler_step(z_current, y, step_size)
            elif method == 'rk4':
                z_current = self._rk4_step(z_current, y, step_size)
            else:
                raise ValueError(f"Unknown integration method: {method}")

            trajectory.append(z_current)

        return torch.stack(trajectory, dim=1)

    def compute_divergence(
        self,
        z: torch.Tensor,
        y: torch.Tensor,
        epsilon: float = 1e-4
    ) -> torch.Tensor:
        """
        Compute divergence of vector field using finite differences.

        div(v) = ∂v_1/∂z_1 + ∂v_2/∂z_2 + ... + ∂v_d/∂z_d

        Args:
            z: [batch, embedding_dim] embeddings
            y: [batch, num_attributes] attributes
            epsilon: Step size for finite differences

        Returns:
            div: [batch] divergence at each point
        """
        batch_size, dim = z.shape
        divergence = torch.zeros(batch_size, device=z.device)

        for i in range(dim):
            # Perturb along dimension i
            z_plus = z.clone()
            z_plus[:, i] += epsilon

            z_minus = z.clone()
            z_minus[:, i] -= epsilon

            # Compute finite difference
            v_plus = self.forward(z_plus, y)
            v_minus = self.forward(z_minus, y)

            # Partial derivative
            partial = (v_plus[:, i] - v_minus[:, i]) / (2 * epsilon)
            divergence += partial

        return divergence

    def compute_curl_magnitude(
        self,
        z: torch.Tensor,
        y: torch.Tensor,
        epsilon: float = 1e-4,
        num_samples: int = 10
    ) -> torch.Tensor:
        """
        Estimate curl magnitude using random projections (for high-dim spaces).

        For high dimensions, we sample random 2D planes and compute curl in those planes.

        Args:
            z: [batch, embedding_dim] embeddings
            y: [batch, num_attributes] attributes
            epsilon: Step size for finite differences
            num_samples: Number of random planes to sample

        Returns:
            curl_mag: [batch] average curl magnitude
        """
        batch_size, dim = z.shape
        curl_magnitudes = []

        for _ in range(num_samples):
            # Sample random orthogonal directions
            idx1 = torch.randint(0, dim, (1,)).item()
            idx2 = torch.randint(0, dim, (1,)).item()

            if idx1 == idx2:
                continue

            # Compute curl in this 2D plane: ∂v_j/∂z_i - ∂v_i/∂z_j
            z_plus_i = z.clone()
            z_plus_i[:, idx1] += epsilon

            z_minus_i = z.clone()
            z_minus_i[:, idx1] -= epsilon

            z_plus_j = z.clone()
            z_plus_j[:, idx2] += epsilon

            z_minus_j = z.clone()
            z_minus_j[:, idx2] -= epsilon

            v_plus_i = self.forward(z_plus_i, y)
            v_minus_i = self.forward(z_minus_i, y)
            v_plus_j = self.forward(z_plus_j, y)
            v_minus_j = self.forward(z_minus_j, y)

            # Partial derivatives
            dv_j_dz_i = (v_plus_i[:, idx2] - v_minus_i[:, idx2]) / (2 * epsilon)
            dv_i_dz_j = (v_plus_j[:, idx1] - v_minus_j[:, idx1]) / (2 * epsilon)

            # Curl component
            curl = dv_j_dz_i - dv_i_dz_j
            curl_magnitudes.append(curl.abs())

        # Average over samples
        curl_mag = torch.stack(curl_magnitudes).mean(dim=0)

        return curl_mag
