"""
Loss functions for attribute-specific vector field training.

Key innovations:
1. Per-attribute contrastive loss (avoids 2^N discrete clustering)
2. Orthogonality loss (encourages attribute independence)
3. Smoothness loss (prevents sudden jumps)
4. Optional cycle consistency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class PerAttributeContrastiveLoss(nn.Module):
    """
    Contrastive loss applied PER ATTRIBUTE instead of all attributes together.

    For each attribute independently:
    - Positive pairs: Samples with SAME value for this attribute
    - Negative pairs: Samples with DIFFERENT values for this attribute

    This creates smooth manifolds instead of discrete 2^N clusters!
    """

    def __init__(
        self,
        temperature: float = 0.2,
        allow_partial_negatives: bool = True
    ):
        super().__init__()
        self.temperature = temperature
        self.allow_partial_negatives = allow_partial_negatives

    def forward(
        self,
        z_flowed: torch.Tensor,
        target_attrs: torch.Tensor,
        attr_idx: int
    ) -> torch.Tensor:
        """
        Compute contrastive loss for ONE attribute.

        Args:
            z_flowed: [batch, dim] embeddings after flow
            target_attrs: [batch, num_attrs] target attributes
            attr_idx: Which attribute to compute loss for

        Returns:
            loss: scalar
        """
        batch_size = z_flowed.size(0)

        # Normalize embeddings
        z_norm = F.normalize(z_flowed, dim=1)

        # Compute similarity matrix
        sim_matrix = torch.matmul(z_norm, z_norm.T) / self.temperature

        # Create positive/negative masks for this attribute
        attr_values = target_attrs[:, attr_idx]  # [batch]
        positive_mask = (attr_values.unsqueeze(0) == attr_values.unsqueeze(1)).float()
        negative_mask = 1.0 - positive_mask

        # Remove self-similarity
        eye_mask = torch.eye(batch_size, device=z_flowed.device)
        positive_mask = positive_mask * (1 - eye_mask)

        # Compute InfoNCE loss
        exp_sim = torch.exp(sim_matrix)

        # Numerator: similarity to positives
        pos_sim = (exp_sim * positive_mask).sum(dim=1)

        # Denominator: similarity to all (positives + negatives)
        if self.allow_partial_negatives:
            # Only use samples with different attribute value as negatives
            all_sim = (exp_sim * (positive_mask + negative_mask)).sum(dim=1)
        else:
            # Use all samples (standard InfoNCE)
            all_sim = exp_sim.sum(dim=1) - torch.diag(exp_sim)

        # Avoid log(0)
        loss = -torch.log((pos_sim + 1e-8) / (all_sim + 1e-8))

        # Only compute loss for samples that have positives
        has_positives = (positive_mask.sum(dim=1) > 0).float()
        loss = (loss * has_positives).sum() / (has_positives.sum() + 1e-8)

        return loss


class OrthogonalityLoss(nn.Module):
    """
    Encourages different attribute vector fields to be orthogonal.

    This ensures attributes are independent and don't interfere with each other.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        velocities: list
    ) -> torch.Tensor:
        """
        Compute orthogonality loss given velocity vectors for each attribute.

        Args:
            velocities: List of [batch, dim] tensors, one per attribute

        Returns:
            loss: scalar
        """
        num_attrs = len(velocities)

        if num_attrs < 2:
            return torch.tensor(0.0, device=velocities[0].device)

        # Normalize velocities
        velocities_norm = [F.normalize(v, dim=1) for v in velocities]

        # Compute pairwise cosine similarities
        total_loss = 0.0
        count = 0

        for i in range(num_attrs):
            for j in range(i+1, num_attrs):
                # Cosine similarity between velocity_i and velocity_j
                cos_sim = (velocities_norm[i] * velocities_norm[j]).sum(dim=1)

                # We want orthogonality, so cosine sim should be 0
                # Penalize absolute value (don't care about sign)
                loss = cos_sim.abs().mean()

                total_loss = total_loss + loss
                count += 1

        return total_loss / count if count > 0 else torch.tensor(0.0)


class SmoothnessLoss(nn.Module):
    """
    Encourages smooth flow trajectories.

    Measures second derivative (acceleration) along trajectory.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        trajectory: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute smoothness loss from trajectory.

        Args:
            trajectory: [batch, num_steps, dim] trajectory

        Returns:
            loss: scalar
        """
        # Compute velocities (first derivative)
        velocities = trajectory[:, 1:, :] - trajectory[:, :-1, :]  # [batch, num_steps-1, dim]

        # Compute accelerations (second derivative)
        accelerations = velocities[:, 1:, :] - velocities[:, :-1, :]  # [batch, num_steps-2, dim]

        # Penalize large accelerations
        loss = torch.norm(accelerations, dim=2).mean()

        return loss


class AttributeSpecificCombinedLoss(nn.Module):
    """
    Combined loss for attribute-specific vector field training.

    Combines:
    1. Per-attribute contrastive losses (one for each attribute)
    2. Orthogonality loss (attribute independence)
    3. Identity loss (stay near manifold)
    4. Smoothness loss (smooth trajectories)
    5. Optional regularization (curl, divergence)
    """

    def __init__(
        self,
        num_attributes: int = 5,
        temperature: float = 0.2,
        lambda_contrastive: float = 0.5,
        lambda_orthogonal: float = 0.1,
        lambda_identity: float = 0.2,
        lambda_smoothness: float = 0.1,
        lambda_curl: float = 0.0,
        lambda_div: float = 0.0
    ):
        super().__init__()
        self.num_attributes = num_attributes
        self.lambda_contrastive = lambda_contrastive
        self.lambda_orthogonal = lambda_orthogonal
        self.lambda_identity = lambda_identity
        self.lambda_smoothness = lambda_smoothness
        self.lambda_curl = lambda_curl
        self.lambda_div = lambda_div

        # Initialize component losses
        self.per_attr_contrastive = PerAttributeContrastiveLoss(
            temperature=temperature
        )
        self.orthogonality = OrthogonalityLoss()
        self.smoothness = SmoothnessLoss()

    def forward(
        self,
        model,
        z_start: torch.Tensor,
        z_end: torch.Tensor,
        target_attrs: torch.Tensor,
        trajectory: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all loss components.

        Args:
            model: The vector field model
            z_start: [batch, dim] starting embeddings
            z_end: [batch, dim] embeddings after flow
            target_attrs: [batch, num_attrs] target attributes
            trajectory: [batch, num_steps, dim] optional full trajectory

        Returns:
            losses: Dict with all loss components
        """
        losses = {}

        # 1. Per-attribute contrastive losses
        contrastive_total = 0.0
        for attr_idx in range(self.num_attributes):
            loss = self.per_attr_contrastive(z_end, target_attrs, attr_idx)
            contrastive_total = contrastive_total + loss
            losses[f'contrastive_attr{attr_idx}'] = loss

        contrastive_total = contrastive_total / self.num_attributes
        losses['contrastive'] = contrastive_total

        # 2. Orthogonality loss (attribute independence)
        if self.lambda_orthogonal > 0:
            # Compute velocities for all attributes
            velocities = model.compute_attribute_velocities(z_start)
            ortho_loss = self.orthogonality(velocities)
            losses['orthogonal'] = ortho_loss
        else:
            losses['orthogonal'] = torch.tensor(0.0, device=z_start.device)

        # 3. Identity loss (stay near manifold)
        if self.lambda_identity > 0:
            # Use cosine distance on unit sphere
            identity_loss = 1 - F.cosine_similarity(z_start, z_end, dim=1).mean()
            losses['identity'] = identity_loss
        else:
            losses['identity'] = torch.tensor(0.0, device=z_start.device)

        # 4. Smoothness loss (smooth trajectories)
        if self.lambda_smoothness > 0 and trajectory is not None:
            smooth_loss = self.smoothness(trajectory)
            losses['smoothness'] = smooth_loss
        else:
            losses['smoothness'] = torch.tensor(0.0, device=z_start.device)

        # 5. Regularization (curl, divergence) - placeholders for now
        losses['curl'] = torch.tensor(0.0, device=z_start.device)
        losses['div'] = torch.tensor(0.0, device=z_start.device)

        # Total loss
        total = (
            self.lambda_contrastive * losses['contrastive'] +
            self.lambda_orthogonal * losses['orthogonal'] +
            self.lambda_identity * losses['identity'] +
            self.lambda_smoothness * losses['smoothness'] +
            self.lambda_curl * losses['curl'] +
            self.lambda_div * losses['div']
        )
        losses['total'] = total

        return losses


def create_attr_specific_loss(config: dict) -> AttributeSpecificCombinedLoss:
    """
    Create loss from config dict.

    Args:
        config: Config dictionary with 'loss' section

    Returns:
        loss: AttributeSpecificCombinedLoss
    """
    loss_config = config['loss']

    return AttributeSpecificCombinedLoss(
        num_attributes=config['model']['num_attributes'],
        temperature=loss_config.get('temperature', 0.2),
        lambda_contrastive=loss_config.get('lambda_contrastive', 0.5),
        lambda_orthogonal=loss_config.get('lambda_orthogonal', 0.1),
        lambda_identity=loss_config.get('lambda_identity', 0.2),
        lambda_smoothness=loss_config.get('lambda_smoothness', 0.1),
        lambda_curl=loss_config.get('lambda_curl', 0.0),
        lambda_div=loss_config.get('lambda_div', 0.0)
    )
