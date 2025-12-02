"""
Check if flipbook trajectories are going in the correct direction.
"""
import json
import sys

if len(sys.argv) < 2:
    print("Usage: python scripts/check_flipbook_directions.py <flipbook_json>")
    sys.exit(1)

flipbook_path = sys.argv[1]

with open(flipbook_path) as f:
    data = json.load(f)

ATTRIBUTE_NAMES = ['Smiling', 'Young', 'Male', 'Eyeglasses', 'Mustache']

print(f"\n{'='*80}")
print("FLIPBOOK DIRECTION CHECK")
print(f"{'='*80}\n")

print("Checking if model moves in correct direction for each attribute...\n")

# For each attribute, track if transitions go in the right direction
attribute_stats = {attr: {'correct': 0, 'wrong': 0, 'total': 0} for attr in ATTRIBUTE_NAMES}

for fb_idx, fb in enumerate(data[:20]):  # Check first 20
    orig_attrs = fb['original_attributes']
    targ_attrs = fb['target_attributes']

    print(f"\nFlipbook {fb_idx}: {fb['change_string']}")
    print(f"  Original attrs: {orig_attrs}")
    print(f"  Target attrs:   {targ_attrs}")

    # For each changed attribute, check if nearest neighbors reflect the change
    for attr_idx, attr_name in enumerate(ATTRIBUTE_NAMES):
        if orig_attrs[attr_idx] != targ_attrs[attr_idx]:
            expected_direction = "add" if targ_attrs[attr_idx] == 1 else "remove"
            print(f"  {attr_name}: {orig_attrs[attr_idx]} → {targ_attrs[attr_idx]} (expect to {expected_direction})")

            # We can't directly check nearest neighbor attributes without loading the dataset
            # But we can check if the trajectory is moving (not stuck)
            indices = [idx[0] for idx in fb['nearest_indices']]
            start_idx = indices[0]
            end_idx = indices[-1]

            if start_idx != end_idx:
                print(f"    ✓ Trajectory moves (NN changes: {start_idx} → {end_idx})")
                attribute_stats[attr_name]['correct'] += 1
            else:
                print(f"    ✗ Trajectory stuck (NN: {start_idx})")
                attribute_stats[attr_name]['wrong'] += 1

            attribute_stats[attr_name]['total'] += 1

print(f"\n{'='*80}")
print("SUMMARY BY ATTRIBUTE")
print(f"{'='*80}\n")

for attr_name in ATTRIBUTE_NAMES:
    stats = attribute_stats[attr_name]
    if stats['total'] > 0:
        correct_pct = stats['correct'] / stats['total'] * 100
        print(f"{attr_name:15} {stats['correct']}/{stats['total']} moving ({correct_pct:.0f}%)")
    else:
        print(f"{attr_name:15} No samples")

print(f"\n{'='*80}")
print("\nNOTE: To fully verify directions, we need to:")
print("1. Load the CelebA dataset")
print("2. Check actual attributes of nearest neighbor images")
print("3. Verify they match the intended direction")
print("\nThis script only checks if trajectories are moving, not if they move")
print("in the CORRECT direction. The mode collapse issue (47% uniqueness)")
print("suggests trajectories converge to attractors rather than moving smoothly.")
print(f"{'='*80}\n")
