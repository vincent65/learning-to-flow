#!/bin/bash
# Quick test script for attribute-specific model
# Tests training for 2 epochs to verify everything works

set -e  # Exit on error

echo "========================================="
echo "Testing Attribute-Specific Model"
echo "========================================="
echo ""

# Create test config
echo "[1/4] Creating test config..."
cat > configs/attr_specific_test.yaml << EOF
model:
  embedding_dim: 512
  num_attributes: 5
  hidden_dim: 256
  projection_radius: 1.0

data:
  celeba_root: "data/celeba"
  embedding_dir: "data/embeddings"

training:
  num_epochs: 2
  batch_size: 128
  learning_rate: 0.0001
  alpha: 0.12
  flow_steps: 10

loss:
  temperature: 0.2
  lambda_contrastive: 0.5
  lambda_orthogonal: 0.1
  lambda_identity: 0.2
  lambda_smoothness: 0.1
  lambda_curl: 0.0
  lambda_div: 0.0

inference:
  num_flow_steps: 10
  step_size: 0.1
EOF

echo "✓ Config created: configs/attr_specific_test.yaml"
echo ""

# Run training
echo "[2/4] Running test training (2 epochs)..."
python src/training/train_attr_specific.py \
    --config configs/attr_specific_test.yaml \
    --output_dir outputs/attr_specific_test \
    --device cuda

echo ""
echo "✓ Training complete!"
echo ""

# Run evaluation
echo "[3/4] Running evaluation..."
python scripts/evaluate_attr_specific.py \
    --checkpoint outputs/attr_specific_test/checkpoints/best.pt \
    --embedding_dir data/embeddings \
    --celeba_root data/celeba \
    --output_dir outputs/attr_specific_test/evaluation \
    --num_samples 200 \
    --device cuda \
    --skip-flipbook

echo ""
echo "✓ Evaluation complete!"
echo ""

# Check for mode collapse
echo "[4/4] Checking for mode collapse..."
python << EOF
import json

with open('outputs/attr_specific_test/evaluation/flipbook_data.json') as f:
    data = json.load(f)

print("\nFlipbook Analysis:")
print("-" * 60)

total_collapsed = 0
total_good = 0

for i in range(min(10, len(data))):
    fb = data[i]
    indices = [idx[0] for idx in fb['nearest_indices']]
    unique = len(set(indices))
    total = len(indices)

    if unique >= total * 0.7:
        status = "✓ GOOD"
        total_good += 1
    else:
        status = "✗ COLLAPSED"
        total_collapsed += 1

    print(f"{i}: {fb['change_string']:<25} Unique: {unique}/{total} {status}")

print("-" * 60)
print(f"\nResults: {total_good} good, {total_collapsed} collapsed")

if total_good >= 7:
    print("\n✓✓✓ SUCCESS! Model is working well!")
    print("    Ready for full training.")
elif total_good >= 4:
    print("\n⚠ PARTIAL SUCCESS. Some mode collapse present.")
    print("   Consider adjusting hyperparameters.")
else:
    print("\n✗✗✗ FAILURE. Severe mode collapse.")
    print("   Check training logs and configuration.")
EOF

echo ""
echo "========================================="
echo "Test Complete!"
echo "========================================="
echo ""
echo "If successful, run full training with:"
echo "  python src/training/train_attr_specific.py \\"
echo "      --config configs/attr_specific_config.yaml \\"
echo "      --output_dir outputs/attr_specific \\"
echo "      --device cuda"
echo ""
