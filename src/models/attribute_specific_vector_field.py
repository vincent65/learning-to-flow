"""
Attribute-Specific Vector Field Network.

Instead of one vector field conditioned on all attributes,
learns separate vector fields for each attribute.

This avoids discrete clustering and enables smooth manifold flows.
"""

import torch
import torch.nn as nn
from typing import Optional
from src.utils.projection import project_to_sphere


class AttributeSpecificVectorField(nn.Module):
    """
    Learn separate vector field for each attribute.

    For an image with attributes [a1, a2, a3, a4, a5],
    to change it to [b1, b2, b3, b4, b5], we compose flows:

    z_final = flow_5(flow_4(flow_3(flow_2(flow_1(z, b1-a1), b2-a2), b3-a3), b4-a4), b5-a5)

    Each flow_i only affects attribute i.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        num_attributes: int = 5,
        hidden_dim: int = 256,
        projection_radius: float = 1.0
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_attributes = num_attributes
        self.projection_radius = projection_radius

        # Shared encoder (optional - can also make fully separate)
        self.shared_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Separate vector field head for each attribute
        self.attribute_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim + 1, hidden_dim),  # +1 for target direction
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, embedding_dim)
            )
            for _ in range(num_attributes)
        ])

    def forward_single_attribute(
        self,
        z: torch.Tensor,
        attr_idx: int,
        direction: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute vector field for a single attribute.

        Args:
            z: [batch, embedding_dim] embeddings
            attr_idx: Which attribute (0-4)
            direction: [batch, 1] target direction (-1, 0, or +1)
                      -1 = remove attribute
                       0 = no change
                      +1 = add attribute

        Returns:
            v: [batch, embedding_dim] velocity vector
        """
        # Encode
        h = self.shared_encoder(z)

        # Concatenate with direction
        h_with_dir = torch.cat([h, direction], dim=1)

        # Compute velocity for this attribute
        v = self.attribute_heads[attr_idx](h_with_dir)

        return v

    def forward(
        self,
        z: torch.Tensor,
        target_attrs: torch.Tensor,
        current_attrs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute combined velocity from all attribute changes.

        Args:
            z: [batch, embedding_dim] embeddings
            target_attrs: [batch, num_attributes] target attributes (0/1)
            current_attrs: [batch, num_attributes] current attributes (0/1)
                          If None, doesn't apply direction weighting

        Returns:
            v: [batch, embedding_dim] combined velocity
        """
        batch_size = z.size(0)

        # Compute direction for each attribute
        if current_attrs is not None:
            # direction = target - current ∈ {-1, 0, +1}
            directions = target_attrs - current_attrs
        else:
            # Just use targets directly (for training with augmentation)
            directions = target_attrs * 2 - 1  # Convert 0/1 to -1/+1

        # Compute velocity for each attribute and sum
        v_total = torch.zeros_like(z)

        for attr_idx in range(self.num_attributes):
            direction = directions[:, attr_idx:attr_idx+1]  # [batch, 1]

            # Only compute if direction is non-zero
            mask = (direction.abs() > 0.01).float()
            if mask.sum() > 0:
                v_attr = self.forward_single_attribute(z, attr_idx, direction)
                v_total = v_total + v_attr * mask

        return v_total

    def flow_single_attribute(
        self,
        z: torch.Tensor,
        attr_idx: int,
        target_value: float,  # 0 or 1
        num_steps: int = 10,
        step_size: float = 0.1
    ) -> torch.Tensor:
        """
        Flow embeddings to change a single attribute.

        Args:
            z: [batch, embedding_dim] starting embeddings
            attr_idx: Which attribute to change
            target_value: 0 or 1
            num_steps: Number of flow steps
            step_size: Step size

        Returns:
            z_flowed: [batch, embedding_dim] final embeddings
        """
        z_current = z
        direction = torch.ones(z.size(0), 1, device=z.device) * (target_value * 2 - 1)

        for _ in range(num_steps):
            v = self.forward_single_attribute(z_current, attr_idx, direction)
            z_current = z_current + step_size * v
            z_current = project_to_sphere(z_current, radius=self.projection_radius)

        return z_current

    def flow_multiple_attributes(
        self,
        z: torch.Tensor,
        current_attrs: torch.Tensor,
        target_attrs: torch.Tensor,
        num_steps: int = 10,
        step_size: float = 0.1,
        method: str = 'sequential'
    ) -> torch.Tensor:
        """
        Flow embeddings to change multiple attributes.

        Args:
            z: [batch, embedding_dim] starting embeddings
            current_attrs: [batch, num_attributes] current attributes
            target_attrs: [batch, num_attributes] target attributes
            method: 'sequential' or 'parallel'
                   sequential: flow one attribute at a time
                   parallel: flow all attributes simultaneously

        Returns:
            z_flowed: [batch, embedding_dim] final embeddings
        """
        if method == 'sequential':
            # Flow each attribute one at a time (more stable)
            z_current = z
            for attr_idx in range(self.num_attributes):
                if (current_attrs[:, attr_idx] != target_attrs[:, attr_idx]).any():
                    z_current = self.flow_single_attribute(
                        z_current,
                        attr_idx,
                        target_attrs[0, attr_idx].item(),  # Assumes batch has same target
                        num_steps=num_steps,
                        step_size=step_size
                    )
            return z_current

        else:  # parallel
            # Flow all attributes at once (faster but less stable)
            z_current = z
            for _ in range(num_steps):
                v = self.forward(z_current, target_attrs, current_attrs)
                z_current = z_current + step_size * v
                z_current = project_to_sphere(z_current, radius=self.projection_radius)
            return z_current

    def get_trajectory(
        self,
        z: torch.Tensor,
        target_attrs: torch.Tensor,
        current_attrs: Optional[torch.Tensor] = None,
        num_steps: int = 10,
        step_size: float = 0.1,
        method: str = 'parallel'
    ) -> torch.Tensor:
        """
        Get full trajectory of flow.

        Returns:
            trajectory: [batch, num_steps+1, embedding_dim]
        """
        trajectory = [z]
        z_current = z

        for _ in range(num_steps):
            v = self.forward(z_current, target_attrs, current_attrs)
            z_current = z_current + step_size * v
            z_current = project_to_sphere(z_current, radius=self.projection_radius)
            trajectory.append(z_current)

        return torch.stack(trajectory, dim=1)
