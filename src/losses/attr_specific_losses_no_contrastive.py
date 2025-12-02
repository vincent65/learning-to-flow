"""
Loss functions WITHOUT contrastive loss to prevent mode collapse.

Key idea: Don't pull embeddings into clusters! Instead:
1. Classifier loss: Ensure flowed embeddings are classified correctly
2. Identity loss: Stay near original CLIP manifold
3. Smoothness loss: Smooth trajectories
4. Orthogonality: Attribute independence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class AttributeClassifierLoss(nn.Module):
    """
    Train simple classifiers to verify flowed embeddings have target attributes.

    Instead of pulling embeddings into clusters (contrastive),
    we just verify that a classifier can predict the target attribute.
    """

    def __init__(self, embedding_dim=512, num_attributes=5):
        super().__init__()

        # Simple linear classifiers per attribute
        self.classifiers = nn.ModuleList([
            nn.Linear(embedding_dim, 1) for _ in range(num_attributes)
        ])

    def forward(self, z_flowed: torch.Tensor, target_attrs: torch.Tensor) -> torch.Tensor:
        """
        Compute classifier loss for all attributes.

        Args:
            z_flowed: [batch, embedding_dim] flowed embeddings
            target_attrs: [batch, num_attributes] target attributes (0 or 1)

        Returns:
            loss: scalar
        """
        total_loss = 0.0

        for attr_idx in range(len(self.classifiers)):
            logits = self.classifiers[attr_idx](z_flowed).squeeze(1)  # [batch]
            targets = target_attrs[:, attr_idx].float()
            loss = F.binary_cross_entropy_with_logits(logits, targets)
            total_loss += loss

        return total_loss / len(self.classifiers)


class OrthogonalityLoss(nn.Module):
    """Encourages different attribute vector fields to be orthogonal."""

    def __init__(self):
        super().__init__()

    def forward(self, velocities: Dict[int, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            velocities: Dict mapping attr_idx -> velocity tensor [batch, embedding_dim]
        """
        if len(velocities) < 2:
            return torch.tensor(0.0, device=list(velocities.values())[0].device)

        # Normalize velocities
        velocities_norm = {k: F.normalize(v, dim=1) for k, v in velocities.items()}

        # Compute pairwise cosine similarities
        loss = 0.0
        count = 0

        attr_indices = list(velocities.keys())
        for i in range(len(attr_indices)):
            for j in range(i + 1, len(attr_indices)):
                idx_i, idx_j = attr_indices[i], attr_indices[j]
                cos_sim = (velocities_norm[idx_i] * velocities_norm[idx_j]).sum(dim=1)
                loss += cos_sim.abs().mean()
                count += 1

        return loss / count if count > 0 else torch.tensor(0.0)


class NoContrastiveCombinedLoss(nn.Module):
    """
    Combined loss WITHOUT contrastive clustering.

    Components:
    1. Classifier loss: Verify target attributes
    2. Identity loss: Stay near manifold
    3. Smoothness loss: Smooth trajectories
    4. Orthogonality: Attribute independence
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        num_attributes: int = 5,
        lambda_classifier: float = 1.0,
        lambda_orthogonal: float = 0.1,
        lambda_identity: float = 0.5,
        lambda_smoothness: float = 0.1,
        lambda_curl: float = 0.0,
        lambda_div: float = 0.0
    ):
        super().__init__()

        self.classifier_loss = AttributeClassifierLoss(embedding_dim, num_attributes)
        self.orthogonality_loss = OrthogonalityLoss()

        self.lambda_classifier = lambda_classifier
        self.lambda_orthogonal = lambda_orthogonal
        self.lambda_identity = lambda_identity
        self.lambda_smoothness = lambda_smoothness
        self.lambda_curl = lambda_curl
        self.lambda_div = lambda_div

    def forward(
        self,
        model,
        z_start: torch.Tensor,
        z_end: torch.Tensor,
        target_attrs: torch.Tensor,
        trajectory: Optional[torch.Tensor] = None,
        current_attrs: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            model: The vector field model
            z_start: [batch, dim] starting embeddings
            z_end: [batch, dim] ending embeddings
            target_attrs: [batch, num_attrs] target attributes
            trajectory: [batch, num_steps, dim] full trajectory (optional, for smoothness)
            current_attrs: [batch, num_attrs] current attributes (optional)

        Returns:
            Dict with 'total' and individual loss components
        """
        losses = {}

        # 1. Classifier loss - verify target attributes
        classifier_loss = self.classifier_loss(z_end, target_attrs)
        losses['classifier'] = self.lambda_classifier * classifier_loss

        # 2. Identity loss - stay near original embeddings
        identity_loss = F.mse_loss(z_end, z_start)
        losses['identity'] = self.lambda_identity * identity_loss

        # 3. Orthogonality loss - attribute independence
        if current_attrs is not None:
            # Compute velocities for each attribute change
            directions = target_attrs - current_attrs  # [batch, num_attrs]
            velocities = {}

            for attr_idx in range(target_attrs.size(1)):
                # Get samples where this attribute changes
                mask = directions[:, attr_idx] != 0
                if mask.sum() > 0:
                    z_samples = z_start[mask]
                    target_samples = target_attrs[mask]
                    current_samples = current_attrs[mask]

                    # Compute velocity for this attribute
                    with torch.no_grad():
                        v = model.forward_single_attribute(
                            z_samples,
                            attr_idx,
                            directions[mask, attr_idx:attr_idx+1]
                        )
                    velocities[attr_idx] = v

            if len(velocities) > 1:
                orth_loss = self.orthogonality_loss(velocities)
                losses['orthogonal'] = self.lambda_orthogonal * orth_loss
            else:
                losses['orthogonal'] = torch.tensor(0.0, device=z_start.device)
        else:
            losses['orthogonal'] = torch.tensor(0.0, device=z_start.device)

        # 4. Smoothness loss - smooth trajectories
        if trajectory is not None and self.lambda_smoothness > 0:
            # Second-order finite differences (acceleration)
            num_steps = trajectory.size(1)
            if num_steps >= 3:
                accel = trajectory[:, 2:, :] - 2 * trajectory[:, 1:-1, :] + trajectory[:, :-2, :]
                smoothness_loss = (accel ** 2).mean()
                losses['smoothness'] = self.lambda_smoothness * smoothness_loss
            else:
                losses['smoothness'] = torch.tensor(0.0, device=z_start.device)
        else:
            losses['smoothness'] = torch.tensor(0.0, device=z_start.device)

        # 5. Optional: Curl/divergence regularization
        losses['curl'] = torch.tensor(0.0, device=z_start.device)
        losses['div'] = torch.tensor(0.0, device=z_start.device)

        # Total loss
        losses['total'] = sum(losses.values())

        return losses


def create_no_contrastive_loss(config: dict) -> NoContrastiveCombinedLoss:
    """Factory function to create loss from config."""
    loss_config = config.get('loss', {})

    return NoContrastiveCombinedLoss(
        embedding_dim=config['model']['embedding_dim'],
        num_attributes=config['model']['num_attributes'],
        lambda_classifier=loss_config.get('lambda_classifier', 1.0),
        lambda_orthogonal=loss_config.get('lambda_orthogonal', 0.1),
        lambda_identity=loss_config.get('lambda_identity', 0.5),
        lambda_smoothness=loss_config.get('lambda_smoothness', 0.1),
        lambda_curl=loss_config.get('lambda_curl', 0.0),
        lambda_div=loss_config.get('lambda_div', 0.0)
    )
