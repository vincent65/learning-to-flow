#!/bin/bash
# Train FCLF vector field

# Default paths
CELEBA_ROOT="data/celeba"
EMBEDDING_DIR="data/embeddings"
CONFIG="configs/fclf_config.yaml"
OUTPUT_DIR="outputs/fclf"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --celeba_root)
            CELEBA_ROOT="$2"
            shift 2
            ;;
        --embedding_dir)
            EMBEDDING_DIR="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --no_regularization)
            NO_REG="--no_regularization"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "======================================"
echo "Training FCLF Vector Field"
echo "======================================"
echo "CelebA root: $CELEBA_ROOT"
echo "Embeddings: $EMBEDDING_DIR"
echo "Config: $CONFIG"
echo "Output: $OUTPUT_DIR"
echo "======================================"

python -m src.training.train_fclf \
    --config "$CONFIG" \
    --celeba_root "$CELEBA_ROOT" \
    --embedding_dir "$EMBEDDING_DIR" \
    --output_dir "$OUTPUT_DIR" \
    $NO_REG

echo "======================================"
echo "Training complete!"
echo "======================================"
