"""
Evaluation script for attribute-specific vector field model.

Computes same metrics as compute_paper_metrics.py but for attr-specific model.

Usage:
    python scripts/evaluate_attr_specific.py \
        --checkpoint outputs/attr_specific/checkpoints/best.pt \
        --embedding_dir data/embeddings \
        --celeba_root data/celeba \
        --output_dir outputs/attr_specific/evaluation \
        --num_samples 2000 \
        --device cuda
"""

import os
import argparse
import json
import numpy as np
import torch
from tqdm import tqdm
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from models.attr_specific_vector_field import load_attr_specific_model
from data.celeba_dataset import get_dataloader

# Import evaluation functions from original script
# (These work the same for both models)
sys.path.append(os.path.join(project_root, 'scripts'))
from compute_paper_metrics import (
    compute_attribute_leakage,
    compute_auc_along_path,
    compute_field_diagnostics,
    compute_nearest_neighbor_flipbook,
    evaluate_method,
    plot_auc_curves,
    ATTRIBUTE_NAMES
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--embedding_dir', type=str, required=True)
    parser.add_argument('--celeba_root', type=str, default='data/celeba')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=2000)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print("Loading model...")
    model = load_attr_specific_model(args.checkpoint, device=args.device)
    model.eval()

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    config = checkpoint['config']

    # Load test data
    print("Loading test data...")
    test_loader = get_dataloader(
        root_dir=args.celeba_root,
        split='test',
        batch_size=128,
        embedding_path=os.path.join(args.embedding_dir, 'test_embeddings.pt'),
        load_images=False,
        num_workers=0,
        shuffle=False
    )

    # Load training embeddings for flipbook search
    print("Loading training embeddings for flipbook search...")
    train_embeddings = torch.load(os.path.join(args.embedding_dir, 'train_embeddings.pt'))
    print(f"  Loaded {len(train_embeddings)} training embeddings")

    # Collect data
    print(f"\nCollecting {args.num_samples} samples...")
    original_embeddings = []
    original_attributes = []
    target_attributes = []
    trajectories = []

    count = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Collecting data"):
            if count >= args.num_samples:
                break

            embeddings = batch['embedding'].to(args.device).float()  # Convert to float32
            attributes = batch['attributes'].to(args.device)
            batch_size = embeddings.size(0)

            # Create target: flip exactly 1 random attribute
            target_attrs = attributes.clone()
            for i in range(batch_size):
                num_flips = 1
                attrs_to_flip = np.random.choice(5, size=num_flips, replace=False)
                for attr_idx in attrs_to_flip:
                    target_attrs[i, attr_idx] = 1 - target_attrs[i, attr_idx]

            # Get trajectory
            num_steps = config.get('inference', {}).get('num_flow_steps', 10)
            step_size = config.get('inference', {}).get('step_size', 0.1)

            trajectory = model.get_trajectory(
                embeddings,
                target_attrs,
                num_steps=num_steps,
                step_size=step_size
            )

            # Store
            original_embeddings.append(embeddings.cpu())
            original_attributes.append(attributes.cpu())
            target_attributes.append(target_attrs.cpu())
            trajectories.append(trajectory.cpu())

            count += batch_size

    # Concatenate
    original_embeddings = torch.cat(original_embeddings)[:args.num_samples]
    original_attributes = torch.cat(original_attributes)[:args.num_samples]
    target_attributes = torch.cat(target_attributes)[:args.num_samples]
    trajectories = torch.cat(trajectories)[:args.num_samples]

    print("\n" + "="*80)
    print("COMPUTING METRICS")
    print("="*80)

    # 1. Attribute Leakage
    print("\n[1/6] Computing attribute leakage...")
    leakage = compute_attribute_leakage(trajectories, original_attributes, target_attributes)

    # 2. Linear Steering Baseline
    print("[2/6] Computing linear steering baseline...")
    # Compute simple linear direction for each attribute
    # (same as original)
    z_start = trajectories[:, 0, :]
    z_end = trajectories[:, -1, :]
    directions = target_attributes - original_attributes
    linear_step_size = 0.1
    z_linear = z_start + 10 * linear_step_size * directions.unsqueeze(1).expand_as(z_start)

    # Normalize to sphere
    from src.utils.projection import project_to_sphere
    z_linear_steered = project_to_sphere(z_linear, radius=1.0)

    # 3. AUC Along Path
    print("[3/6] Computing AUC along path...")
    auc_curves, monotonic_frac = compute_auc_along_path(
        trajectories, original_attributes, target_attributes
    )

    # 4. Field Diagnostics
    print("[4/6] Computing field diagnostics...")
    field_stats = compute_field_diagnostics(model, original_embeddings, original_attributes, args.device)

    # 5. Nearest Neighbor Flipbook
    print("[5/6] Computing nearest-neighbor flipbook data...")
    flipbook = compute_nearest_neighbor_flipbook(
        trajectories, train_embeddings, original_attributes, target_attributes, num_paths=50, k=1
    )

    # 6. Method Comparison
    print("[6/6] Evaluating methods...")
    attr_specific_metrics = evaluate_method(
        original_embeddings, trajectories, original_attributes, target_attributes, "AttrSpecific"
    )
    linear_metrics = evaluate_method(
        original_embeddings, z_linear_steered.unsqueeze(1), original_attributes, target_attributes, "Linear"
    )

    # Compile results
    results = {
        'attribute_leakage': leakage,
        'auc_curves': auc_curves,
        'monotonic_auc_fraction': monotonic_frac,
        'field_diagnostics': field_stats,
        'comparison': {
            'attr_specific': attr_specific_metrics,
            'linear_steering': linear_metrics
        }
    }

    # Save results
    results_file = os.path.join(args.output_dir, 'metrics.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    flipbook_file = os.path.join(args.output_dir, 'flipbook_data.json')
    with open(flipbook_file, 'w') as f:
        json.dump(flipbook, f, indent=2)

    # Plot AUC curves
    plot_auc_curves(auc_curves, os.path.join(args.output_dir, 'auc_curves.png'))

    # Print summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    print("\n1. ATTRIBUTE LEAKAGE (lower is better):")
    for attr, score in leakage.items():
        if isinstance(score, dict):
            print(f"   {attr}: {score['leakage']:.4f} (acc: {score['acc_start']:.3f} → {score['acc_end']:.3f})")

    print("\n2. MONOTONIC AUC PROGRESS:")
    for attr, frac in monotonic_frac.items():
        print(f"   {attr}: {frac*100:.1f}%")

    print("\n3. FIELD DIAGNOSTICS:")
    print(f"   Divergence: mean={field_stats['divergence']['mean']:.4f}, "
          f"std={field_stats['divergence']['std']:.4f}, "
          f"max={field_stats['divergence']['max']:.4f}")
    print(f"   Curl:       mean={field_stats['curl']['mean']:.4f}, "
          f"std={field_stats['curl']['std']:.4f}, "
          f"max={field_stats['curl']['max']:.4f}")

    print("\n4. METHOD COMPARISON:")
    print("\n   Linear Probe Accuracy:")
    print("   " + "-"*60)
    print(f"   {'Attribute':<15} {'AttrSpec':<10} {'Linear':<10} {'Difference'}")
    print("   " + "-"*60)
    for attr in ATTRIBUTE_NAMES:
        as_acc = attr_specific_metrics['linear_probe_accuracy'].get(attr, 0)
        lin_acc = linear_metrics['linear_probe_accuracy'].get(attr, 0)
        diff = as_acc - lin_acc
        print(f"   {attr:<15} {as_acc:.4f}     {lin_acc:.4f}     {diff:+.4f}")

    print(f"\n✅ Results saved to: {results_file}")
    print(f"✅ Flipbook data saved to: {flipbook_file}")
    print(f"✅ AUC curves saved to: {os.path.join(args.output_dir, 'auc_curves.png')}")


if __name__ == '__main__':
    main()
