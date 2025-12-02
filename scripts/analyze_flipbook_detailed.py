"""
Detailed flipbook analysis to understand mode collapse patterns.
"""
import json
import numpy as np
import sys

if len(sys.argv) < 2:
    print("Usage: python scripts/analyze_flipbook_detailed.py <flipbook_json>")
    sys.exit(1)

flipbook_path = sys.argv[1]

with open(flipbook_path) as f:
    data = json.load(f)

print(f"\n{'='*80}")
print("DETAILED FLIPBOOK ANALYSIS")
print(f"{'='*80}\n")

print(f"Total flipbooks: {len(data)}\n")

# Analyze uniqueness statistics
uniqueness_ratios = []
repeated_indices_count = []

for fb in data:
    indices = [idx[0] for idx in fb['nearest_indices']]
    unique = len(set(indices))
    total = len(indices)
    ratio = unique / total
    uniqueness_ratios.append(ratio)

    # Count how many times the most common index appears
    from collections import Counter
    counts = Counter(indices)
    most_common_count = counts.most_common(1)[0][1] if counts else 0
    repeated_indices_count.append(most_common_count)

print("📊 UNIQUENESS STATISTICS:")
print(f"  Mean uniqueness: {np.mean(uniqueness_ratios)*100:.1f}%")
print(f"  Median uniqueness: {np.median(uniqueness_ratios)*100:.1f}%")
print(f"  Min uniqueness: {np.min(uniqueness_ratios)*100:.1f}%")
print(f"  Max uniqueness: {np.max(uniqueness_ratios)*100:.1f}%")
print(f"  Std dev: {np.std(uniqueness_ratios)*100:.1f}%\n")

print("🔄 REPETITION PATTERNS:")
print(f"  Mean repetitions (most common index): {np.mean(repeated_indices_count):.1f}")
print(f"  Max repetitions: {np.max(repeated_indices_count)}")
print(f"  Trajectories with >5 repeats: {sum(1 for x in repeated_indices_count if x > 5)}/{len(data)}\n")

# Check if indices are actually changing or stuck
print("🎯 TRAJECTORY MOVEMENT ANALYSIS:")
trajectories_with_no_movement = 0
trajectories_with_some_movement = 0

for fb in data:
    indices = [idx[0] for idx in fb['nearest_indices']]
    first_idx = indices[0]
    last_idx = indices[-1]

    if first_idx == last_idx:
        trajectories_with_no_movement += 1
    else:
        trajectories_with_some_movement += 1

print(f"  Start == End (no movement): {trajectories_with_no_movement}/{len(data)}")
print(f"  Start != End (some movement): {trajectories_with_some_movement}/{len(data)}\n")

# Detailed breakdown of first 10
print("📋 DETAILED BREAKDOWN (first 10 flipbooks):")
print("-" * 80)

for i in range(min(10, len(data))):
    fb = data[i]
    indices = [idx[0] for idx in fb['nearest_indices']]
    unique = len(set(indices))
    total = len(indices)

    # Count transitions
    transitions = sum(1 for j in range(len(indices)-1) if indices[j] != indices[j+1])

    from collections import Counter
    counts = Counter(indices)
    most_common_idx, most_common_count = counts.most_common(1)[0]

    print(f"\n{i}: {fb['change_string']}")
    print(f"   Indices: {indices}")
    print(f"   Unique: {unique}/{total} ({unique/total*100:.0f}%)")
    print(f"   Transitions: {transitions}/{total-1}")
    print(f"   Most repeated: index {most_common_idx} appears {most_common_count} times")

print("\n" + "="*80)

# Determine verdict with more nuanced thresholds
mean_uniqueness = np.mean(uniqueness_ratios)
mean_transitions = np.mean([sum(1 for j in range(len([idx[0] for idx in fb['nearest_indices']])-1)
                                 if [idx[0] for idx in fb['nearest_indices']][j] != [idx[0] for idx in fb['nearest_indices']][j+1])
                             for fb in data])

print("\n🎯 VERDICT:")
if mean_uniqueness >= 0.7:
    print("✅ EXCELLENT - No mode collapse")
elif mean_uniqueness >= 0.5 and mean_transitions >= 5:
    print("⚠️  MODERATE - Some repetition but trajectories are moving")
    print("   This is acceptable for a 2-epoch model. Likely to improve with more training.")
elif mean_uniqueness >= 0.4:
    print("⚠️  CONCERNING - Significant mode collapse")
    print("   Model may need hyperparameter tuning.")
else:
    print("❌ SEVERE - Critical mode collapse")
    print("   Model is collapsing to discrete attractors.")

print(f"\n   For comparison:")
print(f"   - Original FCLF: ~10-20% unique (severe collapse)")
print(f"   - This model: {mean_uniqueness*100:.1f}% unique")
print(f"   - Target: >70% unique")
print("="*80 + "\n")
