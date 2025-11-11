#!/usr/bin/env python3
"""
Debug script to understand what's happening with metrics.
"""

import json
import numpy as np

# Load the metrics
with open('results/transfer_eval_fixed/transfer_metrics.json', 'r') as f:
    metrics = json.load(f)

print("=" * 60)
print("METRICS DEBUG ANALYSIS")
print("=" * 60)

print("\nSilhouette Scores:")
print(f"  Original by original: {metrics['silhouette_original_by_original']['overall']:.4f}")
print(f"  Original by target:   {metrics['silhouette_original_by_target']['overall']:.4f}")
print(f"  Flowed by original:   {metrics['silhouette_flowed_by_original']['overall']:.4f}")
print(f"  Flowed by target:     {metrics['silhouette_flowed_by_target']['overall']:.4f}")

print("\nPer-Attribute Silhouette (Flowed by Target):")
attrs = ['Smiling', 'Young', 'Male', 'Eyeglasses', 'Mustache']
for i, attr in enumerate(attrs):
    score = metrics['silhouette_flowed_by_target'].get(f'attr_{i}', 0.0)
    print(f"  {attr:12s}: {score:.4f}")

print("\nCluster Purity:")
print(f"  Original: {metrics['purity_original']:.4f}")
print(f"  Flowed:   {metrics['purity_flowed']:.4f}")

print("\nTrajectory Smoothness:")
print(f"  Mean step: {metrics['smoothness']['mean_step_distance']:.4f}")
print(f"  Std step:  {metrics['smoothness']['std_step_distance']:.4f}")
print(f"  Max step:  {metrics['smoothness']['max_step_distance']:.4f}")
print(f"  Min step:  {metrics['smoothness']['min_step_distance']:.4f}")

print("\nEmbedding Movement:")
print(f"  Mean: {metrics['movement']['mean']:.4f}")
print(f"  Std:  {metrics['movement']['std']:.4f}")
print(f"  Max:  {metrics['movement']['max']:.4f}")
print(f"  Min:  {metrics['movement']['min']:.4f}")

print("\n" + "=" * 60)
print("INTERPRETATION:")
print("=" * 60)

# Analyze the results
sil_by_target = metrics['silhouette_flowed_by_target']['overall']
sil_by_original = metrics['silhouette_flowed_by_original']['overall']
movement = metrics['movement']['mean']
step_size = metrics['smoothness']['mean_step_distance']

print("\n1. Mode Collapse Check:")
if sil_by_target > 0.95:
    print("   ❌ COLLAPSED: Silhouette = 1.0 indicates discrete points")
else:
    print(f"   ✓ OK: Silhouette = {sil_by_target:.2f} (not perfect)")

print("\n2. Attribute Transfer Check:")
if sil_by_target > sil_by_original:
    print(f"   ✓ SUCCESS: Flowed embeddings match target attrs better")
    print(f"     Target: {sil_by_target:.2f} > Original: {sil_by_original:.2f}")
else:
    print(f"   ❌ FAILURE: Model not learning transfer")

print("\n3. Movement Magnitude:")
if movement > 5.0:
    print(f"   ❌ TOO LARGE: Mean movement = {movement:.2f}")
    print(f"     (Should be <1.5 for normalized embeddings)")
elif movement > 1.5:
    print(f"   ⚠️  HIGH: Mean movement = {movement:.2f}")
elif movement < 0.1:
    print(f"   ⚠️  TOO SMALL: Mean movement = {movement:.2f}")
    print(f"     (Model barely moving embeddings)")
else:
    print(f"   ✓ GOOD: Mean movement = {movement:.2f}")

print("\n4. Trajectory Smoothness:")
if step_size > 0.2:
    print(f"   ❌ POOR: Mean step = {step_size:.2f} (should be 0.02-0.05)")
elif step_size > 0.1:
    print(f"   ⚠️  ROUGH: Mean step = {step_size:.2f}")
else:
    print(f"   ✓ SMOOTH: Mean step = {step_size:.2f}")

print("\n" + "=" * 60)
print("RECOMMENDATION:")
print("=" * 60)

if movement > 3.0 and sil_by_target > 0.95:
    print("\n⚠️  Your model has EXCESSIVE MOVEMENT + MODE COLLAPSE")
    print("\nRoot cause: lambda_identity is too weak")
    print("\nAction needed:")
    print("  1. Increase lambda_identity from 0.1 to 1.0")
    print("  2. Retrain from scratch")
    print("  3. Target movement: <1.5, step size: <0.1")
elif movement > 3.0:
    print("\n⚠️  Your model is OVER-CORRECTING")
    print("  Increase lambda_identity to constrain movement")
elif sil_by_target > 0.95:
    print("\n⚠️  MODE COLLAPSE detected")
    print("  Despite visual appearance, embeddings are collapsing")
else:
    print("\n✓ Model appears to be working!")
    print("  Continue training to improve metrics")
