#!/bin/bash
# Train CLIP decoder

# Default paths
CELEBA_ROOT="data/celeba"
EMBEDDING_DIR="data/embeddings"
CONFIG="configs/decoder_config.yaml"
OUTPUT_DIR="outputs/decoder"

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
        --use_vae)
            USE_VAE="--use_vae"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "======================================"
echo "Training CLIP Decoder"
echo "======================================"
echo "CelebA root: $CELEBA_ROOT"
echo "Embeddings: $EMBEDDING_DIR"
echo "Config: $CONFIG"
echo "Output: $OUTPUT_DIR"
echo "======================================"

python -m src.training.train_decoder \
    --config "$CONFIG" \
    --celeba_root "$CELEBA_ROOT" \
    --embedding_dir "$EMBEDDING_DIR" \
    --output_dir "$OUTPUT_DIR" \
    $USE_VAE

echo "======================================"
echo "Training complete!"
echo "======================================"
