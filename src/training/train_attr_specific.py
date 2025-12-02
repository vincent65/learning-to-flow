"""
Training script for attribute-specific vector field model.

Usage:
    python src/training/train_attr_specific.py \
        --config configs/attr_specific_config.yaml \
        --output_dir outputs/attr_specific \
        --device cuda
"""

import os
import argparse
import yaml
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from models.attr_specific_vector_field import AttributeSpecificVectorField
from losses.attr_specific_losses import create_attr_specific_loss
from losses.attr_specific_losses_no_contrastive import create_no_contrastive_loss
from data.celeba_dataset import get_dataloader


def train_epoch(
    model,
    criterion,
    train_loader,
    optimizer,
    device,
    config,
    epoch
):
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    loss_components = {}  # Will be populated dynamically based on loss type

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        embeddings = batch['embedding'].to(device).float()  # Convert to float32
        attributes = batch['attributes'].to(device)
        batch_size = embeddings.size(0)

        # Create target attributes by flipping random attributes
        target_attrs = attributes.clone()
        num_flips = torch.randint(1, 3, (batch_size,))  # Flip 1 or 2 attributes

        for i in range(batch_size):
            attrs_to_flip = torch.randperm(config['model']['num_attributes'])[:num_flips[i]]
            for attr_idx in attrs_to_flip:
                target_attrs[i, attr_idx] = 1 - target_attrs[i, attr_idx]

        # Forward pass - get trajectory
        num_steps = config['training'].get('flow_steps', 10)
        step_size = config['training'].get('alpha', 0.12)

        trajectory = model.get_trajectory(
            embeddings,
            target_attrs,
            num_steps=num_steps,
            step_size=step_size
        )

        z_start = trajectory[:, 0, :]
        z_end = trajectory[:, -1, :]

        # Compute loss
        losses = criterion(
            model=model,
            z_start=z_start,
            z_end=z_end,
            target_attrs=target_attrs,
            trajectory=trajectory
        )

        loss = losses['total']

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Update metrics
        total_loss += loss.item()
        for key in losses.keys():
            if key != 'total':  # Skip the total key
                if key not in loss_components:
                    loss_components[key] = 0.0
                loss_components[key] += losses[key].item()

        # Update progress bar
        postfix = {'loss': f"{loss.item():.4f}"}

        # Add loss components that exist
        if 'contrastive' in losses:
            postfix['contr'] = f"{losses['contrastive'].item():.3f}"
        if 'classifier' in losses:
            postfix['class'] = f"{losses['classifier'].item():.3f}"
        if 'orthogonal' in losses:
            postfix['orth'] = f"{losses['orthogonal'].item():.3f}"
        if 'identity' in losses:
            postfix['ident'] = f"{losses['identity'].item():.4f}"

        pbar.set_postfix(postfix)

    # Average losses
    num_batches = len(train_loader)
    avg_loss = total_loss / num_batches
    for key in loss_components.keys():
        loss_components[key] /= num_batches

    return avg_loss, loss_components


def validate(
    model,
    criterion,
    val_loader,
    device,
    config
):
    """Validate model."""
    model.eval()

    total_loss = 0.0
    loss_components = {}  # Will be populated dynamically

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            embeddings = batch['embedding'].to(device).float()  # Convert to float32
            attributes = batch['attributes'].to(device)
            batch_size = embeddings.size(0)

            # Create target attributes
            target_attrs = attributes.clone()
            num_flips = torch.randint(1, 3, (batch_size,))

            for i in range(batch_size):
                attrs_to_flip = torch.randperm(config['model']['num_attributes'])[:num_flips[i]]
                for attr_idx in attrs_to_flip:
                    target_attrs[i, attr_idx] = 1 - target_attrs[i, attr_idx]

            # Forward pass
            num_steps = config['training'].get('flow_steps', 10)
            step_size = config['training'].get('alpha', 0.12)

            trajectory = model.get_trajectory(
                embeddings,
                target_attrs,
                num_steps=num_steps,
                step_size=step_size
            )

            z_start = trajectory[:, 0, :]
            z_end = trajectory[:, -1, :]

            # Compute loss
            losses = criterion(
                model=model,
                z_start=z_start,
                z_end=z_end,
                target_attrs=target_attrs,
                trajectory=trajectory
            )

            total_loss += losses['total'].item()
            for key in losses.keys():
                if key != 'total':
                    if key not in loss_components:
                        loss_components[key] = 0.0
                    loss_components[key] += losses[key].item()

    # Average losses
    num_batches = len(val_loader)
    avg_loss = total_loss / num_batches
    for key in loss_components.keys():
        loss_components[key] /= num_batches

    return avg_loss, loss_components


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='outputs/attr_specific', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # Save config
    config_path = os.path.join(args.output_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    # Initialize TensorBoard
    writer = SummaryWriter(log_dir)

    # Create model
    print("Creating model...")
    model = AttributeSpecificVectorField(
        embedding_dim=config['model']['embedding_dim'],
        num_attributes=config['model']['num_attributes'],
        hidden_dim=config['model']['hidden_dim'],
        projection_radius=config['model'].get('projection_radius', 1.0)
    )
    model = model.to(args.device)

    # Create loss (choose based on config)
    if config.get('training', {}).get('use_no_contrastive', False):
        print("Using NO CONTRASTIVE loss (classifier-based)")
        criterion = create_no_contrastive_loss(config)
        criterion = criterion.to(args.device)  # Move loss to device (has classifiers)
    else:
        print("Using standard contrastive loss")
        criterion = create_attr_specific_loss(config)

    # Create optimizer
    # Include criterion parameters if it has trainable components (classifiers)
    params_to_optimize = list(model.parameters())
    if config.get('training', {}).get('use_no_contrastive', False):
        params_to_optimize += list(criterion.parameters())

    optimizer = optim.Adam(
        params_to_optimize,
        lr=config['training']['learning_rate']
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10
    )

    # Load dataloaders
    print("Loading data...")
    train_loader = get_dataloader(
        root_dir=config['data']['celeba_root'],
        split='train',
        batch_size=config['training']['batch_size'],
        embedding_path=os.path.join(config['data']['embedding_dir'], 'train_embeddings.pt'),
        load_images=False,
        shuffle=True,
        num_workers=4
    )

    val_loader = get_dataloader(
        root_dir=config['data']['celeba_root'],
        split='val',
        batch_size=config['training']['batch_size'],
        embedding_path=os.path.join(config['data']['embedding_dir'], 'val_embeddings.pt'),
        load_images=False,
        shuffle=False,
        num_workers=4
    )

    # Resume from checkpoint if specified
    start_epoch = 1
    best_val_loss = float('inf')

    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    # Training loop
    print(f"\nStarting training for {config['training']['num_epochs']} epochs...")
    print(f"Output directory: {args.output_dir}\n")

    for epoch in range(start_epoch, config['training']['num_epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['training']['num_epochs']}")
        print("="*60)

        # Train
        train_loss, train_components = train_epoch(
            model, criterion, train_loader, optimizer, args.device, config, epoch
        )

        print(f"Train Loss: {train_loss:.4f}")
        if 'contrastive' in train_components:
            print(f"  Contrastive: {train_components['contrastive']:.4f}")
        if 'classifier' in train_components:
            print(f"  Classifier:  {train_components['classifier']:.4f}")
        if 'orthogonal' in train_components:
            print(f"  Orthogonal:  {train_components['orthogonal']:.4f}")
        if 'identity' in train_components:
            print(f"  Identity:    {train_components['identity']:.4f}")
        if 'smoothness' in train_components:
            print(f"  Smoothness:  {train_components['smoothness']:.4f}")

        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        for key, value in train_components.items():
            writer.add_scalar(f'Loss/train_{key}', value, epoch)

        # Validate
        val_loss, val_components = validate(
            model, criterion, val_loader, args.device, config
        )

        print(f"Val Loss: {val_loss:.4f}")

        writer.add_scalar('Loss/val', val_loss, epoch)
        for key, value in val_components.items():
            writer.add_scalar(f'Loss/val_{key}', value, epoch)

        # Learning rate schedule
        scheduler.step(val_loss)

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'config': config
        }

        # Save last checkpoint
        last_path = os.path.join(checkpoint_dir, 'last.pt')
        torch.save(checkpoint, last_path)

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(checkpoint_dir, 'best.pt')
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model (val_loss: {val_loss:.4f})")

        # Save periodic checkpoints
        if epoch % 20 == 0:
            periodic_path = os.path.join(checkpoint_dir, f'epoch_{epoch}.pt')
            torch.save(checkpoint, periodic_path)

    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {os.path.join(checkpoint_dir, 'best.pt')}")
    writer.close()


if __name__ == '__main__':
    main()
