# Quick Fix for Current Issues

## Fix NumPy Version Issue

```bash
# 1. Stop the current script (Ctrl+C if running)

# 2. Downgrade NumPy
pip install "numpy<2.0"

# 3. Verify it worked
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
# Should show: NumPy: 1.26.4 (or similar 1.x version)

# 4. Verify PyTorch works
python -c "import torch; print('PyTorch OK')"

# 5. Run the embedding script with fixed settings
python scripts/precompute_embeddings.py \
    --celeba_root data/celeba \
    --output_dir data/embeddings \
    --batch_size 128 \
    --num_workers 0
```

## What These Commands Do

- `pip install "numpy<2.0"` - Installs NumPy 1.x (compatible with PyTorch)
- `--num_workers 0` - Disables multiprocessing (fixes "Numpy not available" errors)
- `--batch_size 128` - Processes 128 images at a time

## Expected Output

You should see:
```
======================================================================
CLIP Embedding Precomputation Started at 2025-11-10 ...
======================================================================

[1/6] Configuration:
  Device: mps (or cuda/cpu)
  CLIP Model: ViT-B/32
  ...

[5/6] Computing embeddings...
  Encoding images: 100%|████████████████| 1583/1583 [20:15<00:00, 1.30it/s]
  
  ✓ Encoding complete in 20.25 minutes
  ...

✓ EMBEDDING PRECOMPUTATION COMPLETE!
```

## How Long Will It Take?

- **With GPU**: 15-25 minutes
- **With CPU**: 60-90 minutes
- **With MPS (Mac)**: 20-35 minutes

## What to Do While Waiting

- The progress bar will update continuously
- Every 100 batches you'll see detailed progress
- You can safely leave it running

## After It Completes

You'll have three files:
- `data/embeddings/train_embeddings.pt` (~316 MB)
- `data/embeddings/val_embeddings.pt` (~39 MB)  
- `data/embeddings/test_embeddings.pt` (~40 MB)

Then you can proceed to training the decoder!

## Still Having Issues?

Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for more solutions.
