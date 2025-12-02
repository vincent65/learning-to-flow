#!/bin/bash
# Quick test for NO CONTRASTIVE loss approach
# Tests training for 5 epochs to verify it maintains high uniqueness

set -e

echo "========================================="
echo "Testing NO CONTRASTIVE Loss"
echo "========================================="
echo ""
echo "HYPOTHESIS: Removing contrastive loss will prevent mode collapse"
echo "  - Contrastive loss creates discrete clusters → collapse"
echo "  - Classifier loss just verifies correctness → no clusters!"
echo ""

# Create test config
echo "[1/4] Creating no-contrastive test config..."
cat > configs/attr_specific_test_no_contrastive.yaml << EOF
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
  alpha: 0.15
  flow_steps: 10
  use_no_contrastive: true  # KEY: Use new loss!

loss:
  lambda_classifier: 1.0
  lambda_orthogonal: 0.1
  lambda_identity: 0.5
  lambda_smoothness: 0.1
  lambda_curl: 0.0
  lambda_div: 0.0

inference:
  num_flow_steps: 10
  step_size: 0.1
EOF

echo "✓ Config created: configs/attr_specific_test_no_contrastive.yaml"
echo ""

# Run training
echo "[2/4] Running test training (5 epochs)..."
python src/training/train_attr_specific.py \
    --config configs/attr_specific_test_no_contrastive.yaml \
    --output_dir outputs/attr_specific_test_no_contrastive \
    --device cuda

echo ""
echo "✓ Training complete!"
echo ""

# Run evaluation
echo "[3/4] Running evaluation..."
python scripts/evaluate_attr_specific.py \
    --checkpoint outputs/attr_specific_test_no_contrastive/checkpoints/best.pt \
    --embedding_dir data/embeddings \
    --celeba_root data/celeba \
    --output_dir outputs/attr_specific_test_no_contrastive/evaluation \
    --num_samples 200 \
    --device cuda

echo ""
echo "✓ Evaluation complete!"
echo ""

# Analyze flipbook
echo "[4/4] Analyzing flipbook..."
python scripts/analyze_flipbook_detailed.py outputs/attr_specific_test_no_contrastive/evaluation/flipbook_data.json

echo ""
echo "========================================="
echo "COMPARISON"
echo "========================================="
echo ""
echo "Uniqueness across different approaches:"
echo "  Original (2 epochs):     47.5% ← Best so far"
echo "  Original (100 epochs):   20.4% ← Got worse!"
echo "  Softer (5 epochs):       14.0% ← Even worse!"
echo "  No Contrastive (5 epochs): [see above] ← New approach"
echo ""
echo "✅ SUCCESS CRITERIA: >35% uniqueness after 5 epochs"
echo "   If achieved, proceed with full 100-epoch training:"
echo ""
echo "   python src/training/train_attr_specific.py \\"
echo "       --config configs/attr_specific_no_contrastive.yaml \\"
echo "       --output_dir outputs/attr_specific_no_contrastive \\"
echo "       --device cuda"
echo ""
