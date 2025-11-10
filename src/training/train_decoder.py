"""
Training script for CLIP decoder.

Pretrain decoder to map CLIP embeddings -> images before FCLF training.
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.clip_decoder import CLIPDecoder, CLIPDecoderVAE, vae_loss
from data.celeba_dataset import get_dataloader
from data.data_utils import get_image_transforms, denormalize_image


def train_decoder(
    config_path: str,
    celeba_root: str,
    embedding_dir: str,
    output_dir: str,
    use_vae: bool = False,
    device: str = None
):
    """
    Train CLIP decoder.

    Args:
        config_path: Path to config YAML file
        celeba_root: Path to CelebA dataset
        embedding_dir: Path to precomputed embeddings
        output_dir: Directory to save checkpoints and logs
        use_vae: Whether to use VAE decoder
        device: Device to use (defaults to cuda if available)
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # TensorBoard
    writer = SummaryWriter(os.path.join(output_dir, 'logs'))

    # Data transforms
    img_size = config['model']['img_size']
    train_transform = get_image_transforms(img_size, is_train=True)
    val_transform = get_image_transforms(img_size, is_train=False)

    # Data loaders
    train_loader = get_dataloader(
        root_dir=celeba_root,
        split='train',
        batch_size=config['training']['batch_size'],
        embedding_path=os.path.join(embedding_dir, 'train_embeddings.pt'),
        load_images=True,
        transform=train_transform,
        num_workers=config['data']['num_workers'],
        shuffle=True
    )

    val_loader = get_dataloader(
        root_dir=celeba_root,
        split='val',
        batch_size=config['training']['batch_size'],
        embedding_path=os.path.join(embedding_dir, 'val_embeddings.pt'),
        load_images=True,
        transform=val_transform,
        num_workers=config['data']['num_workers'],
        shuffle=False
    )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    # Model
    if use_vae:
        model = CLIPDecoderVAE(
            embedding_dim=config['model']['embedding_dim'],
            img_size=img_size
        )
    else:
        model = CLIPDecoder(
            embedding_dim=config['model']['embedding_dim'],
            img_size=img_size
        )

    model = model.to(device)
    print(f"Model: {model.__class__.__name__}")

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True
    )

    # Training loop
    best_val_loss = float('inf')
    num_epochs = config['training']['num_epochs']

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, device, use_vae, writer, epoch
        )

        # Validate
        val_loss = validate(
            model, val_loader, device, use_vae, writer, epoch
        )

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': config
        }

        # Save latest
        torch.save(
            checkpoint,
            os.path.join(checkpoint_dir, 'decoder_latest.pt')
        )

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                checkpoint,
                os.path.join(checkpoint_dir, 'decoder_best.pt')
            )
            print(f"Saved best model (val_loss: {val_loss:.4f})")

    writer.close()
    print("Training complete!")


def train_epoch(model, dataloader, optimizer, device, use_vae, writer, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0

    pbar = tqdm(dataloader, desc="Training")

    for batch_idx, batch in enumerate(pbar):
        embeddings = batch['embedding'].to(device)
        images = batch['image'].to(device)

        optimizer.zero_grad()

        if use_vae:
            recon_images, mu, logvar = model(embeddings)
            loss, recon_loss, kl_loss = vae_loss(recon_images, images, mu, logvar)
            total_kl_loss += kl_loss.item()
        else:
            recon_images = model(embeddings)
            recon_loss = nn.functional.mse_loss(recon_images, images)
            loss = recon_loss
            kl_loss = torch.tensor(0.0)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon_loss += recon_loss.item()

        # Update progress bar
        pbar.set_postfix({
            'loss': loss.item(),
            'recon': recon_loss.item()
        })

        # Log to TensorBoard
        global_step = epoch * len(dataloader) + batch_idx
        writer.add_scalar('Train/Loss', loss.item(), global_step)
        writer.add_scalar('Train/ReconLoss', recon_loss.item(), global_step)
        if use_vae:
            writer.add_scalar('Train/KLLoss', kl_loss.item(), global_step)

    avg_loss = total_loss / len(dataloader)
    print(f"Train Loss: {avg_loss:.4f}")

    return avg_loss


def validate(model, dataloader, device, use_vae, writer, epoch):
    """Validate model."""
    model.eval()
    total_loss = 0
    total_recon_loss = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validation")):
            embeddings = batch['embedding'].to(device)
            images = batch['image'].to(device)

            if use_vae:
                recon_images, mu, logvar = model(embeddings)
                loss, recon_loss, kl_loss = vae_loss(recon_images, images, mu, logvar)
            else:
                recon_images = model(embeddings)
                recon_loss = nn.functional.mse_loss(recon_images, images)
                loss = recon_loss

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()

            # Log sample reconstructions
            if batch_idx == 0:
                # Denormalize for visualization
                images_vis = denormalize_image(images[:8])
                recon_vis = denormalize_image(recon_images[:8])

                writer.add_images('Val/Original', images_vis, epoch)
                writer.add_images('Val/Reconstructed', recon_vis, epoch)

    avg_loss = total_loss / len(dataloader)
    print(f"Val Loss: {avg_loss:.4f}")

    writer.add_scalar('Val/Loss', avg_loss, epoch)
    writer.add_scalar('Val/ReconLoss', total_recon_loss / len(dataloader), epoch)

    return avg_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CLIP decoder")
    parser.add_argument("--config", type=str, default="configs/decoder_config.yaml",
                        help="Path to config file")
    parser.add_argument("--celeba_root", type=str, required=True,
                        help="Path to CelebA root directory")
    parser.add_argument("--embedding_dir", type=str, default="data/embeddings",
                        help="Path to precomputed embeddings")
    parser.add_argument("--output_dir", type=str, default="outputs/decoder",
                        help="Output directory")
    parser.add_argument("--use_vae", action="store_true",
                        help="Use VAE decoder instead of simple decoder")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu)")

    args = parser.parse_args()

    train_decoder(
        config_path=args.config,
        celeba_root=args.celeba_root,
        embedding_dir=args.embedding_dir,
        output_dir=args.output_dir,
        use_vae=args.use_vae,
        device=args.device
    )
