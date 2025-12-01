"""
Plot training curves from TensorBoard logs.

Generates publication-ready figures showing loss components over training.
Inspired by cs229.ipynb visualization approach.

Usage:
    python scripts/plot_training_curves.py \
        --logdir outputs/v4_projection/logs \
        --output_dir outputs/v4_projection/plots
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from utils.tensorboard_utils import get_training_stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9


def plot_training_curves(stats, output_path, steps_per_epoch=None):
    """
    Create 2x3 subplot figure with all training curves.

    Args:
        stats: Dict from get_training_stats()
        output_path: Where to save figure
        steps_per_epoch: If provided, use epoch-based x-axis
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Define which metrics to plot in each subplot
    metrics = [
        ('Train/Loss', 'Val/Loss', 'Total Loss'),
        ('Train/Contrastive', 'Val/Contrastive', 'Contrastive Loss'),
        ('Train/Curl', None, 'Curl Regularization'),
        ('Train/Div', None, 'Divergence Regularization'),
        ('Train/Identity', None, 'Identity Loss'),
        (None, None, 'Learning Rate')  # Placeholder for LR if available
    ]

    for idx, (train_tag, val_tag, title) in enumerate(metrics):
        ax = axes[idx]

        has_data = False

        # Plot training curve
        if train_tag and train_tag in stats:
            if steps_per_epoch and 'epoch_values' in stats[train_tag]:
                x = stats[train_tag]['epochs']
                y = stats[train_tag]['epoch_values']
                xlabel = 'Epoch'
            else:
                x = stats[train_tag]['steps']
                y = stats[train_tag]['smoothed']
                xlabel = 'Step'

            ax.plot(x, y, label='Train', linewidth=2, alpha=0.8, color='tab:blue')
            has_data = True

        # Plot validation curve
        if val_tag and val_tag in stats:
            if steps_per_epoch and 'epoch_values' in stats[val_tag]:
                x = stats[val_tag]['epochs']
                y = stats[val_tag]['epoch_values']
            else:
                x = stats[val_tag]['steps']
                y = stats[val_tag]['smoothed']

            ax.plot(x, y, label='Val', linewidth=2, alpha=0.8, color='tab:orange')
            has_data = True

        if has_data:
            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel('Loss', fontsize=11)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.grid(alpha=0.3)
            ax.legend(loc='best')
        else:
            # No data for this subplot
            ax.text(0.5, 0.5, 'No data available',
                   ha='center', va='center', fontsize=12, color='gray')
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved training curves to: {output_path}")


def plot_loss_breakdown(stats, output_path, steps_per_epoch=None):
    """
    Create stacked area chart showing loss component breakdown.

    Args:
        stats: Dict from get_training_stats()
        output_path: Where to save figure
        steps_per_epoch: If provided, use epoch-based x-axis
    """
    # Components to include
    components = [
        ('Train/Contrastive', 'Contrastive'),
        ('Train/Curl', 'Curl'),
        ('Train/Div', 'Divergence'),
        ('Train/Identity', 'Identity')
    ]

    # Check which components are available
    available = []
    x_data = None

    for tag, label in components:
        if tag in stats:
            if steps_per_epoch and 'epoch_values' in stats[tag]:
                x = np.array(stats[tag]['epochs'])
                y = np.array(stats[tag]['epoch_values'])
                xlabel = 'Epoch'
            else:
                x = np.array(stats[tag]['steps'])
                y = np.array(stats[tag]['smoothed'])
                xlabel = 'Step'

            available.append((x, y, label))

            if x_data is None:
                x_data = x

    if not available:
        print("No loss components available for breakdown plot")
        return

    # Create stacked area chart
    fig, ax = plt.subplots(figsize=(10, 6))

    # Stack the components
    y_stack = np.zeros(len(x_data))
    colors = plt.cm.Set3(np.linspace(0, 1, len(available)))

    for idx, (x, y, label) in enumerate(available):
        # Interpolate to common x-axis if needed
        if not np.array_equal(x, x_data):
            y = np.interp(x_data, x, y)

        ax.fill_between(x_data, y_stack, y_stack + y,
                        label=label, alpha=0.7, color=colors[idx])
        y_stack += y

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Loss Magnitude', fontsize=12)
    ax.set_title('Loss Component Breakdown Over Training', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved loss breakdown to: {output_path}")


def plot_loss_ratios(stats, output_path, steps_per_epoch=None):
    """
    Plot the ratio of loss components to total loss over time.

    Shows how the contribution of each component changes during training.
    """
    components = [
        ('Train/Contrastive', 'Contrastive'),
        ('Train/Curl', 'Curl'),
        ('Train/Div', 'Divergence'),
        ('Train/Identity', 'Identity')
    ]

    # Get total loss
    if 'Train/Loss' not in stats:
        print("Total loss not available for ratio plot")
        return

    if steps_per_epoch and 'epoch_values' in stats['Train/Loss']:
        x_total = np.array(stats['Train/Loss']['epochs'])
        y_total = np.array(stats['Train/Loss']['epoch_values'])
        xlabel = 'Epoch'
    else:
        x_total = np.array(stats['Train/Loss']['steps'])
        y_total = np.array(stats['Train/Loss']['smoothed'])
        xlabel = 'Step'

    fig, ax = plt.subplots(figsize=(10, 6))

    for tag, label in components:
        if tag in stats:
            if steps_per_epoch and 'epoch_values' in stats[tag]:
                x = np.array(stats[tag]['epochs'])
                y = np.array(stats[tag]['epoch_values'])
            else:
                x = np.array(stats[tag]['steps'])
                y = np.array(stats[tag]['smoothed'])

            # Interpolate to match total loss x-axis
            if not np.array_equal(x, x_total):
                y = np.interp(x_total, x, y)

            # Compute ratio
            ratio = y / (y_total + 1e-8)
            ax.plot(x_total, ratio, label=label, linewidth=2, alpha=0.8)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Fraction of Total Loss', fontsize=12)
    ax.set_title('Loss Component Ratios Over Training', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved loss ratios to: {output_path}")


def print_summary(stats):
    """Print summary statistics from training."""
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)

    # Final values
    print("\nFinal Loss Values:")
    for tag in ['Train/Loss', 'Val/Loss']:
        if tag in stats and stats[tag]['values']:
            final_val = stats[tag]['values'][-1]
            print(f"  {tag:<20} {final_val:.4f}")

    # Loss components
    print("\nFinal Loss Components:")
    for tag in ['Train/Contrastive', 'Train/Curl', 'Train/Div', 'Train/Identity']:
        if tag in stats and stats[tag]['values']:
            final_val = stats[tag]['values'][-1]
            print(f"  {tag:<20} {final_val:.4f}")

    # Training progress
    if 'Train/Loss' in stats:
        values = stats['Train/Loss']['values']
        if len(values) > 10:
            initial = np.mean(values[:10])
            final = np.mean(values[-10:])
            improvement = (initial - final) / initial * 100
            print(f"\nTraining Loss Improvement: {improvement:.1f}%")
            print(f"  Initial (first 10 steps): {initial:.4f}")
            print(f"  Final (last 10 steps):    {final:.4f}")

    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Plot training curves from TensorBoard logs')
    parser.add_argument('--logdir', type=str, required=True, help='Path to TensorBoard log directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for plots')
    parser.add_argument('--steps_per_epoch', type=int, default=None,
                       help='Number of steps per epoch (for epoch-based x-axis)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Check if logdir exists
    if not os.path.exists(args.logdir):
        print(f"Error: Log directory not found: {args.logdir}")
        return

    print(f"Parsing TensorBoard logs from: {args.logdir}")

    # Parse logs
    stats = get_training_stats(args.logdir, args.steps_per_epoch)

    if stats is None or not stats:
        print("No training data found in TensorBoard logs")
        return

    print(f"Found {len(stats)} metrics in logs")

    # Generate plots
    print("\nGenerating plots...")

    # 1. Main training curves (2x3 grid)
    plot_training_curves(
        stats,
        os.path.join(args.output_dir, 'training_curves.png'),
        args.steps_per_epoch
    )

    # 2. Loss breakdown (stacked area)
    plot_loss_breakdown(
        stats,
        os.path.join(args.output_dir, 'loss_breakdown.png'),
        args.steps_per_epoch
    )

    # 3. Loss ratios
    plot_loss_ratios(
        stats,
        os.path.join(args.output_dir, 'loss_ratios.png'),
        args.steps_per_epoch
    )

    # Print summary
    print_summary(stats)

    print(f"\n✅ All plots saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
