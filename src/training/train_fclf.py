"""
Training script for FCLF vector field network.
"""

import os
import yaml
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vector_field import VectorFieldNetwork
from data.celeba_dataset import get_dataloader
from losses.combined_loss import FCLFLoss, FCLFLossSimple


def train_fclf(
    config_path: str,
    celeba_root: str,
    embedding_dir: str,
    output_dir: str,
    use_regularization: bool = True,
    device: str = None,
    resume_from: str = None
):
    """
    Train FCLF vector field network.

    Args:
        config_path: Path to config YAML
        celeba_root: Path to CelebA dataset
        embedding_dir: Path to precomputed embeddings
        output_dir: Output directory for checkpoints and logs
        use_regularization: Whether to use curl/divergence regularization
        device: Device to use
        resume_from: Path to checkpoint to resume from (optional)
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Ensure numeric types are correct (some YAML parsers return strings)
    config['training']['learning_rate'] = float(config['training']['learning_rate'])
    config['training']['batch_size'] = int(config['training']['batch_size'])
    config['training']['num_epochs'] = int(config['training']['num_epochs'])
    config['training']['alpha'] = float(config['training']['alpha'])
    config['loss']['temperature'] = float(config['loss']['temperature'])
    config['loss']['lambda_curl'] = float(config['loss']['lambda_curl'])
    config['loss']['lambda_div'] = float(config['loss']['lambda_div'])

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

    # Data loaders (no images needed, just embeddings and attributes)
    # Use num_workers=0 to avoid multiprocessing issues on VMs
    train_loader = get_dataloader(
        root_dir=celeba_root,
        split='train',
        batch_size=config['training']['batch_size'],
        embedding_path=os.path.join(embedding_dir, 'train_embeddings.pt'),
        load_images=False,
        num_workers=0,
        shuffle=True
    )

    val_loader = get_dataloader(
        root_dir=celeba_root,
        split='val',
        batch_size=config['training']['batch_size'],
        embedding_path=os.path.join(embedding_dir, 'val_embeddings.pt'),
        load_images=False,
        num_workers=0,
        shuffle=False
    )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    # Model
    model = VectorFieldNetwork(
        embedding_dim=config['model']['embedding_dim'],
        num_attributes=config['model']['num_attributes'],
        hidden_dim=config['model']['hidden_dim']
    )
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss function
    if use_regularization:
        criterion = FCLFLoss(
            temperature=config['loss']['temperature'],
            alpha=config['training']['alpha'],
            lambda_curl=config['loss']['lambda_curl'],
            lambda_div=config['loss']['lambda_div']
        )
    else:
        criterion = FCLFLossSimple(
            temperature=config['loss']['temperature'],
            alpha=config['training']['alpha']
        )

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate']
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['num_epochs'],
        eta_min=1e-6
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')

    if resume_from is not None:
        if os.path.exists(resume_from):
            print(f"\nResuming from checkpoint: {resume_from}")
            checkpoint = torch.load(resume_from, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('val_loss', float('inf'))
            print(f"Resuming from epoch {start_epoch} (best val loss: {best_val_loss:.4f})")
        else:
            print(f"Warning: Checkpoint not found at {resume_from}, starting from scratch")

    # Training loop
    num_epochs = config['training']['num_epochs']

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device,
            use_regularization, writer, epoch
        )

        # Validate
        val_loss = validate(
            model, val_loader, criterion, device,
            use_regularization, writer, epoch
        )

        # Learning rate scheduling
        scheduler.step()

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
            os.path.join(checkpoint_dir, 'fclf_latest.pt')
        )

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                checkpoint,
                os.path.join(checkpoint_dir, 'fclf_best.pt')
            )
            print(f"Saved best model (val_loss: {val_loss:.4f})")

        # Save periodic checkpoints
        if (epoch + 1) % 5 == 0:
            torch.save(
                checkpoint,
                os.path.join(checkpoint_dir, f'fclf_epoch_{epoch+1}.pt')
            )

    writer.close()
    print("Training complete!")


def train_epoch(model, dataloader, criterion, optimizer, device,
                use_regularization, writer, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_contrastive = 0
    total_curl = 0
    total_div = 0

    pbar = tqdm(dataloader, desc="Training")

    for batch_idx, batch in enumerate(pbar):
        embeddings = batch['embedding'].to(device)
        attributes = batch['attributes'].to(device)

        optimizer.zero_grad()

        if use_regularization:
            loss, contrastive, curl, div = criterion(
                model, embeddings, attributes, return_components=True
            )
            total_contrastive += contrastive.item()
            total_curl += curl.item()
            total_div += div.item()
        else:
            loss = criterion(model, embeddings, attributes)
            contrastive = loss

        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()

        # Update progress bar
        if use_regularization:
            pbar.set_postfix({
                'loss': loss.item(),
                'contr': contrastive.item(),
                'curl': curl.item(),
                'div': div.item()
            })
        else:
            pbar.set_postfix({'loss': loss.item()})

        # Log to TensorBoard
        global_step = epoch * len(dataloader) + batch_idx
        writer.add_scalar('Train/Loss', loss.item(), global_step)
        writer.add_scalar('Train/Contrastive', contrastive.item(), global_step)
        if use_regularization:
            writer.add_scalar('Train/Curl', curl.item(), global_step)
            writer.add_scalar('Train/Div', div.item(), global_step)

    avg_loss = total_loss / len(dataloader)
    print(f"Train Loss: {avg_loss:.4f}")

    return avg_loss


def validate(model, dataloader, criterion, device, use_regularization, writer, epoch):
    """Validate model."""
    model.eval()
    total_loss = 0
    total_contrastive = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            embeddings = batch['embedding'].to(device)
            attributes = batch['attributes'].to(device)

            if use_regularization:
                loss, contrastive, curl, div = criterion(
                    model, embeddings, attributes, return_components=True
                )
                total_contrastive += contrastive.item()
            else:
                loss = criterion(model, embeddings, attributes)

            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Val Loss: {avg_loss:.4f}")

    writer.add_scalar('Val/Loss', avg_loss, epoch)

    return avg_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FCLF vector field")
    parser.add_argument("--config", type=str, default="configs/fclf_config.yaml",
                        help="Path to config file")
    parser.add_argument("--celeba_root", type=str, required=True,
                        help="Path to CelebA root directory")
    parser.add_argument("--embedding_dir", type=str, default="data/embeddings",
                        help="Path to precomputed embeddings")
    parser.add_argument("--output_dir", type=str, default="outputs/fclf",
                        help="Output directory")
    parser.add_argument("--no_regularization", action="store_true",
                        help="Disable curl/divergence regularization")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from (e.g., outputs/fclf/checkpoints/fclf_latest.pt)")

    args = parser.parse_args()

    train_fclf(
        config_path=args.config,
        celeba_root=args.celeba_root,
        embedding_dir=args.embedding_dir,
        output_dir=args.output_dir,
        use_regularization=not args.no_regularization,
        device=args.device,
        resume_from=args.resume
    )
