"""
Attribute-Specific Vector Field Network (v2 - Production Ready)

Key design principles:
1. Separate vector field per attribute (avoids 2^N discrete clustering)
2. Shared encoder for efficiency (learns general embedding structure)
3. Parallel and sequential flow modes
4. Clean interface matching original VectorFieldNetwork
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple
from src.utils.projection import project_to_sphere


class AttributeSpecificVectorField(nn.Module):
    """
    Learn separate vector field for each binary attribute.

    Architecture:
        z -> [SharedEncoder] -> h (shared representation)
        h + direction_i -> [AttributeHead_i] -> v_i (velocity for attribute i)

    To change multiple attributes, we can:
    - Sequential: Apply each attribute flow one after another
    - Parallel: Sum all velocity vectors and flow together
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        num_attributes: int = 5,
        hidden_dim: int = 256,
        projection_radius: float = 1.0,
        use_layer_norm: bool = True
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_attributes = num_attributes
        self.hidden_dim = hidden_dim
        self.projection_radius = projection_radius

        # Shared encoder: Learns general structure of embedding space
        encoder_layers = [
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
        ]
        self.shared_encoder = nn.Sequential(*encoder_layers)

        # Attribute-specific heads: One per attribute
        self.attribute_heads = nn.ModuleList([
            self._build_attribute_head(hidden_dim, embedding_dim, use_layer_norm)
            for _ in range(num_attributes)
        ])

        # Initialize weights properly
        self.apply(self._init_weights)

    def _build_attribute_head(
        self,
        hidden_dim: int,
        embedding_dim: int,
        use_layer_norm: bool
    ) -> nn.Module:
        """Build a single attribute-specific head."""
        return nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),  # +1 for direction signal
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def _init_weights(self, module):
        """Initialize weights with Xavier/Kaiming."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward_single_attribute(
        self,
        z: torch.Tensor,
        attr_idx: int,
        direction: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute velocity for a SINGLE attribute.

        Args:
            z: [batch, embedding_dim] embeddings
            attr_idx: Which attribute (0 to num_attributes-1)
            direction: [batch, 1] direction signal:
                      +1.0 = add attribute (0 -> 1)
                      -1.0 = remove attribute (1 -> 0)
                       0.0 = no change

        Returns:
            v: [batch, embedding_dim] velocity vector
        """
        # Encode to shared representation
        h = self.shared_encoder(z)

        # Concatenate with direction signal
        h_with_dir = torch.cat([h, direction], dim=1)

        # Compute attribute-specific velocity
        v = self.attribute_heads[attr_idx](h_with_dir)

        return v

    def forward(
        self,
        z: torch.Tensor,
        y: torch.Tensor,
        current_attrs: Optional[torch.Tensor] = None,
        mode: str = 'parallel'
    ) -> torch.Tensor:
        """
        Compute velocity for attribute changes.

        Args:
            z: [batch, embedding_dim] embeddings
            y: [batch, num_attributes] target attributes (0 or 1)
            current_attrs: [batch, num_attributes] current attributes (0 or 1)
                          If None, assumes we want to move toward y
            mode: 'parallel' (sum all velocities) or 'sequential' (not used here)

        Returns:
            v: [batch, embedding_dim] combined velocity
        """
        batch_size = z.size(0)

        # Compute direction signals
        if current_attrs is not None:
            # Direction = target - current ∈ {-1, 0, +1}
            directions = y - current_attrs  # [batch, num_attributes]
        else:
            # If no current attributes given, use binary target
            # Convert 0/1 to -1/+1 (assume we're moving from opposite state)
            directions = y * 2 - 1

        # Compute velocity from each attribute head
        v_total = torch.zeros_like(z)

        for attr_idx in range(self.num_attributes):
            direction = directions[:, attr_idx:attr_idx+1]  # [batch, 1]

            # Compute velocity for this attribute
            v_attr = self.forward_single_attribute(z, attr_idx, direction)

            # Weight by direction magnitude (allows 0 to mean "no change")
            v_total = v_total + v_attr * direction.abs()

        return v_total

    def _euler_step(
        self,
        z: torch.Tensor,
        y: torch.Tensor,
        step_size: float,
        current_attrs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Single Euler step with projection."""
        v = self.forward(z, y, current_attrs)
        z_next = z + step_size * v
        z_next = project_to_sphere(z_next, radius=self.projection_radius)
        return z_next

    def flow(
        self,
        z: torch.Tensor,
        y: torch.Tensor,
        num_steps: int = 10,
        step_size: float = 0.1,
        method: str = 'euler'
    ) -> torch.Tensor:
        """
        Flow embeddings to target attributes.

        Args:
            z: [batch, embedding_dim] starting embeddings
            y: [batch, num_attributes] target attributes
            num_steps: Number of flow steps
            step_size: Step size for integration
            method: Integration method ('euler' only for now)

        Returns:
            z_final: [batch, embedding_dim] final embeddings
        """
        z_current = z

        for _ in range(num_steps):
            if method == 'euler':
                z_current = self._euler_step(z_current, y, step_size)
            else:
                raise ValueError(f"Unknown integration method: {method}")

        return z_current

    def get_trajectory(
        self,
        z: torch.Tensor,
        y: torch.Tensor,
        num_steps: int = 10,
        step_size: float = 0.1,
        method: str = 'euler',
        current_attrs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get full trajectory during flow.

        Args:
            z: [batch, embedding_dim] starting embeddings
            y: [batch, num_attributes] target attributes
            num_steps: Number of flow steps
            step_size: Step size
            method: Integration method
            current_attrs: [batch, num_attributes] current attributes (optional)
                          If provided, computes direction as (target - current)
                          If None, assumes moving from opposite state

        Returns:
            trajectory: [batch, num_steps+1, embedding_dim]
                       Includes starting point at index 0
        """
        trajectory = [z]
        z_current = z

        for _ in range(num_steps):
            if method == 'euler':
                z_current = self._euler_step(z_current, y, step_size, current_attrs)
            else:
                raise ValueError(f"Unknown integration method: {method}")

            trajectory.append(z_current)

        return torch.stack(trajectory, dim=1)

    def flow_single_attribute_trajectory(
        self,
        z: torch.Tensor,
        attr_idx: int,
        target_value: float,
        num_steps: int = 10,
        step_size: float = 0.1
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Flow a single attribute and return full trajectory.

        Useful for visualization and debugging.

        Args:
            z: [batch, embedding_dim] starting embeddings
            attr_idx: Which attribute to change (0 to num_attributes-1)
            target_value: Target value (0.0 or 1.0)
            num_steps: Number of steps
            step_size: Step size

        Returns:
            z_final: [batch, embedding_dim] final embeddings
            trajectory: List of [batch, embedding_dim] at each step
        """
        trajectory = [z]
        z_current = z

        # Direction: -1 if removing (1->0), +1 if adding (0->1)
        direction = torch.ones(z.size(0), 1, device=z.device) * (target_value * 2 - 1)

        for _ in range(num_steps):
            v = self.forward_single_attribute(z_current, attr_idx, direction)
            z_current = z_current + step_size * v
            z_current = project_to_sphere(z_current, radius=self.projection_radius)
            trajectory.append(z_current)

        return z_current, trajectory

    def compute_attribute_velocities(
        self,
        z: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Compute velocity vector for each attribute independently.

        Useful for analyzing attribute separability and orthogonality.

        Args:
            z: [batch, embedding_dim] embeddings

        Returns:
            velocities: List of [batch, embedding_dim], one per attribute
        """
        velocities = []
        direction = torch.ones(z.size(0), 1, device=z.device)

        for attr_idx in range(self.num_attributes):
            v = self.forward_single_attribute(z, attr_idx, direction)
            velocities.append(v)

        return velocities


def load_attr_specific_model(
    checkpoint_path: str,
    device: str = 'cuda'
) -> AttributeSpecificVectorField:
    """
    Load a trained attribute-specific model from checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint file
        device: 'cuda' or 'cpu'

    Returns:
        model: Loaded model in eval mode
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    model = AttributeSpecificVectorField(
        embedding_dim=config['model']['embedding_dim'],
        num_attributes=config['model']['num_attributes'],
        hidden_dim=config['model']['hidden_dim'],
        projection_radius=config['model'].get('projection_radius', 1.0)
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model
