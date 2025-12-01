"""
Combined loss function for FCLF training.

Following cs229.ipynb: projection to unit sphere provides implicit regularization,
so identity loss is optional (default 0.0).

L_total = λ_c * L_contrastive + λ_curl * R_curl + λ_div * R_div + [λ_id * L_identity]
"""

import torch
import torch.nn as nn
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from losses.contrastive_flow_loss import ContrastiveFlowLoss, SimpleContrastiveLoss, AttributeContrastiveLoss
from losses.regularization import CurlRegularization, DivergenceRegularization
from utils.projection import project_to_sphere


class FCLFLoss(nn.Module):
    """
    Complete FCLF loss with contrastive term and regularization.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        alpha: float = 0.1,
        lambda_contrastive: float = 1.0,
        lambda_curl: float = 0.01,
        lambda_div: float = 0.01,
        lambda_identity: float = 0.0,
        curl_epsilon: float = 1e-4,
        curl_samples: int = 10,
        div_epsilon: float = 1e-4
    ):
        """
        Args:
            temperature: Temperature for contrastive loss
            alpha: Flow step size
            lambda_contrastive: Weight for contrastive loss
            lambda_curl: Weight for curl regularization
            lambda_div: Weight for divergence regularization
            lambda_identity: Weight for identity preservation loss
            curl_epsilon: Epsilon for curl finite differences
            curl_samples: Number of random planes for curl estimation
            div_epsilon: Epsilon for divergence finite differences
        """
        super().__init__()

        self.alpha = alpha
        self.lambda_contrastive = lambda_contrastive
        self.lambda_curl = lambda_curl
        self.lambda_div = lambda_div
        self.lambda_identity = lambda_identity

        # Loss components
        # Use AttributeContrastiveLoss for soft similarity matching
        # This never returns zero even with unique attribute combinations
        self.contrastive_loss = AttributeContrastiveLoss(
            temperature=temperature
        )

        self.curl_reg = CurlRegularization(
            epsilon=curl_epsilon,
            num_samples=3  # Reduced from 10 to 3 for speed (2-3x faster!)
        )

        self.div_reg = DivergenceRegularization(
            epsilon=div_epsilon
        )

    def forward(
        self,
        vector_field,
        z: torch.Tensor,
        y: torch.Tensor,
        return_components: bool = False
    ):
        """
        Compute total FCLF loss.

        Args:
            vector_field: VectorFieldNetwork instance
            z: [batch, dim] CLIP embeddings
            y: [batch, num_attrs] target attributes
            return_components: Whether to return loss components separately

        Returns:
            If return_components=False:
                total_loss: scalar
            If return_components=True:
                (total_loss, contrastive_loss, curl_loss, div_loss)
        """
        # Compute one-step flow
        v = vector_field(z, y)
        z_flowed = z + self.alpha * v

        # CRITICAL: Project to unit sphere after flow step (cs229.ipynb key insight)
        # This prevents latent blow-up and provides implicit regularization
        z_flowed = project_to_sphere(z_flowed, radius=1.0)

        # Contrastive loss (AttributeWeightedContrastiveLoss only needs z_flowed and y)
        contrastive_loss = self.contrastive_loss(z_flowed, y)

        # Regularization terms
        curl_loss = self.curl_reg(vector_field, z, y)
        div_loss = self.div_reg(vector_field, z, y)

        # Identity preservation loss (optional - cs229.ipynb uses projection instead)
        # Following cs229: projection provides implicit regularization, so identity loss
        # is optional (default 0.0). Only compute if lambda_identity > 0.
        if self.lambda_identity > 0:
            identity_loss = torch.mean((z_flowed - z) ** 2)
        else:
            identity_loss = torch.tensor(0.0, device=z.device)

        # Total loss
        total_loss = (
            self.lambda_contrastive * contrastive_loss +
            self.lambda_curl * curl_loss +
            self.lambda_div * div_loss +
            self.lambda_identity * identity_loss
        )

        if return_components:
            return total_loss, contrastive_loss, curl_loss, div_loss, identity_loss
        else:
            return total_loss


class FCLFLossSimple(nn.Module):
    """
    Simplified FCLF loss without expensive regularization terms.
    Useful for faster training/debugging.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        alpha: float = 0.1
    ):
        """
        Args:
            temperature: Temperature for contrastive loss
            alpha: Flow step size
        """
        super().__init__()

        self.alpha = alpha
        self.contrastive_loss = SimpleContrastiveLoss(
            temperature=temperature,
            alpha=alpha
        )

    def forward(
        self,
        vector_field,
        z: torch.Tensor,
        y: torch.Tensor
    ):
        """
        Compute contrastive loss only (no regularization).

        Args:
            vector_field: VectorFieldNetwork instance
            z: [batch, dim] embeddings
            y: [batch, num_attrs] attributes

        Returns:
            loss: scalar
        """
        # One-step flow
        v = vector_field(z, y)
        z_flowed = z + self.alpha * v

        # Contrastive loss
        loss = self.contrastive_loss(z, z_flowed, y)

        return loss
