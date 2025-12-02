#!/bin/bash
# Quick test for BALANCED no-contrastive config
# Goal: Fix low linear probe accuracy and monotonic AUC issues

set -e

echo "========================================="
echo "Testing BALANCED No-Contrastive Config"
echo "========================================="
echo ""
echo "FIXES:"
echo "  1. Stronger classifier loss (1.0 → 2.0)"
echo "  2. Weaker identity loss (0.5 → 0.2)"
echo "  3. Smaller training steps (0.15 → 0.1)"
echo ""
echo "GOALS:"
echo "  - Linear probe accuracy >70% for all attributes"
echo "  - Monotonic AUC >50% for at least 3 attributes"
echo "  - Maintain uniqueness >45%"
echo ""

# Create test config
echo "[1/4] Creating balanced test config..."
cat > configs/attr_specific_test_balanced.yaml << EOF
model:
  embedding_dim: 512
  num_attributes: 5
  hidden_dim: 256
  projection_radius: 1.0

data:
  celeba_root: "data/celeba"
  embedding_dir: "data/embeddings"

training:
  num_epochs: 5
  batch_size: 128
  learning_rate: 0.0001
  alpha: 0.1
  flow_steps: 10
  use_no_contrastive: true

loss:
  lambda_classifier: 2.0  # Stronger!
  lambda_orthogonal: 0.1
  lambda_identity: 0.2   # Weaker!
  lambda_smoothness: 0.1
  lambda_curl: 0.0
  lambda_div: 0.0

inference:
  num_flow_steps: 10
  step_size: 0.1
EOF

echo "✓ Config created: configs/attr_specific_test_balanced.yaml"
echo ""

# Run training
echo "[2/4] Running test training (5 epochs)..."
python src/training/train_attr_specific.py \
    --config configs/attr_specific_test_balanced.yaml \
    --output_dir outputs/attr_specific_test_balanced \
    --device cuda

echo ""
echo "✓ Training complete!"
echo ""

# Run evaluation
echo "[3/4] Running evaluation..."
python scripts/evaluate_attr_specific.py \
    --checkpoint outputs/attr_specific_test_balanced/checkpoints/best.pt \
    --embedding_dir data/embeddings \
    --celeba_root data/celeba \
    --output_dir outputs/attr_specific_test_balanced/evaluation \
    --num_samples 200 \
    --device cuda

echo ""
echo "✓ Evaluation complete!"
echo ""

# Analyze results
echo "[4/4] Analyzing results..."
python << EOF
import json

with open('outputs/attr_specific_test_balanced/evaluation/metrics.json') as f:
    data = json.load(f)

print("\n" + "="*80)
print("RESULTS ANALYSIS")
print("="*80)

# Linear probe accuracy
print("\n1. LINEAR PROBE ACCURACY (target: >70% for all)")
print("-"*80)
comparison = data['comparison']
for attr in ['Smiling', 'Young', 'Male', 'Eyeglasses', 'Mustache']:
    acc = comparison['attr_specific']['linear_probe'].get(attr, 0)
    status = "✓" if acc > 0.7 else "✗"
    print(f"  {attr:15} {acc:.1%} {status}")

# Monotonic AUC
print("\n2. MONOTONIC AUC (target: >50% for 3+ attributes)")
print("-"*80)
monotonic = data['monotonic_auc_fraction']
count_good = sum(1 for v in monotonic.values() if v > 0.5)
for attr, frac in monotonic.items():
    status = "✓" if frac > 0.5 else "✗"
    print(f"  {attr:15} {frac:.1%} {status}")
print(f"\n  {count_good}/5 attributes have >50% monotonic AUC")

# Show AUC curves
print("\n3. AUC PROGRESSION")
print("-"*80)
auc_curves = data['auc_curves']
for attr in ['Smiling', 'Male']:  # Show problem attributes
    print(f"\n  {attr}:")
    curves = auc_curves[attr]
    for step in [0, 5, 10]:
        if step < len(curves):
            print(f"    Step {step:2d}: {curves[step]:.3f}")

print("\n" + "="*80)
EOF

# Analyze flipbook
python scripts/analyze_flipbook_detailed.py outputs/attr_specific_test_balanced/evaluation/flipbook_data.json

echo ""
echo "========================================="
echo "COMPARISON TO PREVIOUS TESTS"
echo "========================================="
echo ""
echo "Uniqueness:"
echo "  Original (5 epochs):   49.5%"
echo "  Balanced (5 epochs):   [see above]"
echo ""
echo "Linear Probe Accuracy (average):"
echo "  Original: ~72% (Smiling 48%, Male 58% dragged it down)"
echo "  Balanced: [see above]"
echo ""
echo "Monotonic AUC:"
echo "  Original: 0/5 attributes (all oscillating)"
echo "  Balanced: [see above]"
echo ""
echo "✅ SUCCESS CRITERIA:"
echo "   - Uniqueness >40%"
echo "   - Linear probe >70% for all attributes"
echo "   - Monotonic AUC >50% for 3+ attributes"
echo ""
