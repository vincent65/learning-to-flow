"""
CLIP Decoder Network.
Maps CLIP embeddings back to images for visualization.
"""

import torch
import torch.nn as nn


class CLIPDecoder(nn.Module):
    """
    Simple decoder network: CLIP embedding -> image.

    Uses transposed convolutions to upsample from embedding to image.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        img_size: int = 128,
        img_channels: int = 3
    ):
        """
        Args:
            embedding_dim: Dimension of CLIP embeddings
            img_size: Output image size (assumes square images)
            img_channels: Number of image channels (3 for RGB)
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.img_size = img_size
        self.img_channels = img_channels

        # Initial projection to 4x4 feature map
        self.fc = nn.Linear(embedding_dim, 512 * 4 * 4)

        # Decoder network (4 -> 8 -> 16 -> 32 -> 64 -> 128)
        self.decoder = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 8x8 -> 16x16
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 16x16 -> 32x32
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 32x32 -> 64x64
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 64x64 -> 128x128
            nn.ConvTranspose2d(32, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output in [-1, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode CLIP embedding to image.

        Args:
            z: [batch, embedding_dim] CLIP embeddings

        Returns:
            img: [batch, channels, height, width] reconstructed images
        """
        # Project to feature map
        x = self.fc(z)
        x = x.view(-1, 512, 4, 4)

        # Decode to image
        img = self.decoder(x)

        return img


class CLIPDecoderVAE(nn.Module):
    """
    VAE-based decoder with KL divergence regularization.
    Better quality but more complex to train.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        latent_dim: int = 256,
        img_size: int = 128,
        img_channels: int = 3
    ):
        """
        Args:
            embedding_dim: Dimension of CLIP embeddings
            latent_dim: Dimension of VAE latent space
            img_size: Output image size
            img_channels: Number of image channels
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.img_channels = img_channels

        # Encoder: CLIP embedding -> latent distribution
        self.fc_mu = nn.Linear(embedding_dim, latent_dim)
        self.fc_logvar = nn.Linear(embedding_dim, latent_dim)

        # Decoder: latent -> image
        self.fc_decode = nn.Linear(latent_dim, 512 * 4 * 4)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def encode(self, z: torch.Tensor):
        """Encode to latent distribution parameters."""
        mu = self.fc_mu(z)
        logvar = self.fc_logvar(z)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode from latent space to image."""
        x = self.fc_decode(latent)
        x = x.view(-1, 512, 4, 4)
        img = self.decoder(x)
        return img

    def forward(self, z: torch.Tensor):
        """
        Forward pass with reparameterization.

        Args:
            z: [batch, embedding_dim] CLIP embeddings

        Returns:
            img: [batch, channels, height, width] reconstructed images
            mu: [batch, latent_dim] latent mean
            logvar: [batch, latent_dim] latent log-variance
        """
        mu, logvar = self.encode(z)
        latent = self.reparameterize(mu, logvar)
        img = self.decode(latent)
        return img, mu, logvar


def vae_loss(
    recon_img: torch.Tensor,
    target_img: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float = 0.001
) -> tuple:
    """
    Compute VAE loss: reconstruction + KL divergence.

    Args:
        recon_img: Reconstructed images
        target_img: Target images
        mu: Latent mean
        logvar: Latent log-variance
        kl_weight: Weight for KL term

    Returns:
        total_loss, recon_loss, kl_loss
    """
    # Reconstruction loss (MSE)
    recon_loss = nn.functional.mse_loss(recon_img, target_img, reduction='mean')

    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = kl_loss / mu.size(0)  # Average over batch

    # Total loss
    total_loss = recon_loss + kl_weight * kl_loss

    return total_loss, recon_loss, kl_loss
