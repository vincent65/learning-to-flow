#!/bin/bash
# Quick test for softer clustering config
# Tests training for 5 epochs to verify it improves over original

set -e

echo "========================================="
echo "Testing SOFTER Clustering Config"
echo "========================================="
echo ""

# Create test config with softer clustering
echo "[1/4] Creating softer test config..."
cat > configs/attr_specific_test_softer.yaml << EOF
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
  alpha: 0.12
  flow_steps: 10

loss:
  # SOFTER CLUSTERING
  temperature: 0.3  # Increased from 0.2
  lambda_contrastive: 0.3  # Reduced from 0.5
  lambda_orthogonal: 0.1
  lambda_identity: 0.3  # Increased from 0.2
  lambda_smoothness: 0.1
  lambda_curl: 0.0
  lambda_div: 0.0

inference:
  num_flow_steps: 10
  step_size: 0.1
EOF

echo "✓ Config created: configs/attr_specific_test_softer.yaml"
echo ""

# Run training
echo "[2/4] Running test training (5 epochs)..."
python src/training/train_attr_specific.py \
    --config configs/attr_specific_test_softer.yaml \
    --output_dir outputs/attr_specific_test_softer \
    --device cuda

echo ""
echo "✓ Training complete!"
echo ""

# Run evaluation
echo "[3/4] Running evaluation..."
python scripts/evaluate_attr_specific.py \
    --checkpoint outputs/attr_specific_test_softer/checkpoints/best.pt \
    --embedding_dir data/embeddings \
    --celeba_root data/celeba \
    --output_dir outputs/attr_specific_test_softer/evaluation \
    --num_samples 200 \
    --device cuda

echo ""
echo "✓ Evaluation complete!"
echo ""

# Analyze flipbook
echo "[4/4] Analyzing flipbook..."
python scripts/analyze_flipbook_detailed.py outputs/attr_specific_test_softer/evaluation/flipbook_data.json

echo ""
echo "========================================="
echo "Test Complete!"
echo "========================================="
echo ""
echo "Compare uniqueness:"
echo "  Original (2 epochs):  47.5% unique"
echo "  Original (100 epochs): 20.4% unique (got worse!)"
echo "  Softer (5 epochs):     [see above]"
echo ""
echo "If softer config shows >40% unique after 5 epochs,"
echo "proceed with full 100-epoch training using:"
echo "  python src/training/train_attr_specific.py \\"
echo "      --config configs/attr_specific_config_softer.yaml \\"
echo "      --output_dir outputs/attr_specific_softer \\"
echo "      --device cuda"
echo ""
