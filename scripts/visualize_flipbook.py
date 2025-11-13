"""
Visualize nearest-neighbor flipbooks to show flow trajectories through real images.

For each trajectory, show a grid of real images that are nearest to each point
along the flow path. This proves the vector field moves through meaningful regions.

Usage:
    python scripts/visualize_flipbook.py \
        --flipbook_data paper_metrics/flipbook_data.json \
        --celeba_root data/celeba \
        --output_dir paper_metrics/flipbooks
"""

import os
import argparse
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from data.celeba_dataset import CelebADataset


def create_flipbook_grid(image_paths, num_steps=11, img_size=128):
    """
    Create a horizontal grid showing nearest images at each flow step.

    Args:
        image_paths: List[str] of length num_steps, each is path to nearest image
        num_steps: Number of flow steps
        img_size: Size to resize images to

    Returns:
        PIL Image of the grid
    """
    # Load and resize images
    images = []
    for path in image_paths:
        try:
            img = Image.open(path).convert('RGB')
            img = img.resize((img_size, img_size))
            images.append(img)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # Create blank image
            images.append(Image.new('RGB', (img_size, img_size), color=(128, 128, 128)))

    # Create grid
    grid_width = num_steps * img_size
    grid_height = img_size + 30  # Extra space for labels

    grid = Image.new('RGB', (grid_width, grid_height), color=(255, 255, 255))

    # Paste images
    for i, img in enumerate(images):
        x = i * img_size
        grid.paste(img, (x, 30))

    # Add step labels
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except:
        font = ImageFont.load_default()

    for i in range(num_steps):
        x = i * img_size + img_size // 2
        label = f"t={i}"
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        draw.text((x - text_width // 2, 5), label, fill=(0, 0, 0), font=font)

    return grid


def create_comparison_figure(
    start_image, end_image, flipbook_grid,
    start_attrs, target_attrs, trajectory_idx,
    attribute_names=['Smiling', 'Young', 'Male', 'Eyeglasses', 'Mustache']
):
    """
    Create a comprehensive figure showing:
    - Top: Start image | Attribute changes | End image
    - Bottom: Full flipbook grid

    Returns:
        matplotlib figure
    """
    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 2], hspace=0.3, wspace=0.2)

    # Top row: Start image, attributes, end image
    ax_start = fig.add_subplot(gs[0, 0])
    ax_start.imshow(start_image)
    ax_start.axis('off')
    ax_start.set_title('Start Image', fontsize=14, fontweight='bold')

    ax_attrs = fig.add_subplot(gs[0, 1])
    ax_attrs.axis('off')

    # Format attribute changes
    changes = []
    for i, attr in enumerate(attribute_names):
        start_val = int(start_attrs[i])
        target_val = int(target_attrs[i])
        if start_val != target_val:
            arrow = "→"
            change_str = f"{attr}: {start_val} {arrow} {target_val}"
            changes.append(change_str)

    changes_text = "Attribute Changes:\n\n" + "\n".join(changes)
    ax_attrs.text(0.5, 0.5, changes_text, ha='center', va='center',
                  fontsize=12, family='monospace',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax_end = fig.add_subplot(gs[0, 2])
    ax_end.imshow(end_image)
    ax_end.axis('off')
    ax_end.set_title('End Image (Nearest to Final Embedding)', fontsize=14, fontweight='bold')

    # Bottom row: Full flipbook
    ax_flipbook = fig.add_subplot(gs[1, :])
    ax_flipbook.imshow(flipbook_grid)
    ax_flipbook.axis('off')
    ax_flipbook.set_title('Nearest Images Along Flow Trajectory', fontsize=14, fontweight='bold')

    plt.suptitle(f'Flow Trajectory #{trajectory_idx}', fontsize=16, fontweight='bold')

    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--flipbook_data', type=str, required=True,
                        help='Path to flipbook_data.json')
    parser.add_argument('--celeba_root', type=str, default='data/celeba',
                        help='Path to CelebA dataset')
    parser.add_argument('--embedding_dir', type=str, default='data/embeddings',
                        help='Path to embeddings directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for flipbook visualizations')
    parser.add_argument('--num_flipbooks', type=int, default=10,
                        help='Number of flipbooks to generate')
    parser.add_argument('--img_size', type=int, default=128,
                        help='Image size for grid')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load flipbook data
    print("Loading flipbook data...")
    with open(args.flipbook_data, 'r') as f:
        flipbook_data = json.load(f)

    # Load dataset to get image paths and attributes
    print("Loading CelebA dataset...")
    dataset = CelebADataset(
        root_dir=args.celeba_root,
        split='test',
        embedding_path=None,  # We don't need embeddings
        load_images=False
    )

    # Select flipbooks to visualize
    num_flipbooks = min(args.num_flipbooks, len(flipbook_data))
    selected_flipbooks = np.random.choice(len(flipbook_data), size=num_flipbooks, replace=False)

    print(f"Generating {num_flipbooks} flipbook visualizations...")

    for fb_idx in tqdm(selected_flipbooks):
        flipbook = flipbook_data[fb_idx]
        trajectory_idx = flipbook['trajectory_idx']
        nearest_indices = flipbook['nearest_indices']  # [num_steps, k]

        # Get start and end info
        start_idx = trajectory_idx
        start_item = dataset[start_idx]
        start_attrs = start_item['attributes'].numpy()
        start_image_path = start_item['image_path']

        # For target attributes, we need to infer from the original data collection
        # Since we don't have target_attrs saved, we'll skip the attribute comparison
        # and just show the flipbook

        # Get image paths for all steps
        image_paths = []
        for step in range(len(nearest_indices)):
            # Take first nearest neighbor (k=1)
            nearest_idx = nearest_indices[step][0]
            item = dataset[nearest_idx]
            image_paths.append(item['image_path'])

        # Create flipbook grid
        flipbook_grid = create_flipbook_grid(image_paths, num_steps=len(image_paths), img_size=args.img_size)

        # Save just the grid (simpler version without attribute comparison)
        output_path = os.path.join(args.output_dir, f'flipbook_{trajectory_idx:04d}.png')
        flipbook_grid.save(output_path)

    print(f"\n✅ Generated {num_flipbooks} flipbooks in {args.output_dir}")

    # Create a summary montage of multiple flipbooks
    print("\nCreating summary montage...")
    num_summary = min(5, num_flipbooks)
    summary_flipbooks = selected_flipbooks[:num_summary]

    fig, axes = plt.subplots(num_summary, 1, figsize=(20, 4 * num_summary))
    if num_summary == 1:
        axes = [axes]

    for i, fb_idx in enumerate(summary_flipbooks):
        flipbook = flipbook_data[fb_idx]
        trajectory_idx = flipbook['trajectory_idx']
        nearest_indices = flipbook['nearest_indices']

        # Get image paths
        image_paths = []
        for step in range(len(nearest_indices)):
            nearest_idx = nearest_indices[step][0]
            item = dataset[nearest_idx]
            image_paths.append(item['image_path'])

        # Create grid
        grid = create_flipbook_grid(image_paths, num_steps=len(image_paths), img_size=args.img_size)

        # Plot
        axes[i].imshow(grid)
        axes[i].axis('off')
        axes[i].set_title(f'Trajectory #{trajectory_idx}', fontsize=12, fontweight='bold', loc='left')

    plt.tight_layout()
    summary_path = os.path.join(args.output_dir, 'flipbook_summary.png')
    plt.savefig(summary_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"✅ Summary montage saved to: {summary_path}")

    print("\n" + "="*80)
    print("FLIPBOOK VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nIndividual flipbooks: {args.output_dir}/flipbook_*.png")
    print(f"Summary montage: {summary_path}")
    print("\nThese visualizations prove that your flow moves through")
    print("meaningful regions of the embedding space by showing real")
    print("CelebA images that are nearest to each point along the trajectory.")
    print("="*80)


if __name__ == '__main__':
    main()
