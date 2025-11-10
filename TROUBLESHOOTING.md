# Troubleshooting Guide

## Common Issues and Solutions

### 0. NumPy 2.x Compatibility Warning

**Symptom:**
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.3.1
UserWarning: Failed to initialize NumPy: _ARRAY_API not found
```

**Cause:**
PyTorch was compiled against NumPy 1.x but you have NumPy 2.x installed.

**Solution:**
Downgrade to NumPy 1.x (most reliable):
```bash
pip install "numpy<2.0"
```

This installs NumPy 1.26.4 which is fully compatible with PyTorch. Verify:
```bash
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"  # Should show 1.x
python -c "import torch; print('PyTorch imports successfully')"
```

**Note:** This warning usually doesn't prevent the code from running, but it's best to fix it to avoid potential issues.

---

### 1. "Numpy is not available" Error During Embedding Precomputation

**Symptom:**
```
Error loading data/celeba/img_align_celeba/001645.jpg: Numpy is not available
Error loading data/celeba/img_align_celeba/001646.jpg: Numpy is not available
...
```

**Cause:**
This is a known issue with PyTorch DataLoader multiprocessing on some systems, particularly macOS. The worker processes can't properly access numpy/PIL.

**Solution:**
Run the embedding precomputation with `--num_workers 0` to disable multiprocessing:

```bash
python scripts/precompute_embeddings.py \
    --celeba_root data/celeba \
    --output_dir data/embeddings \
    --batch_size 128 \
    --num_workers 0
```

This will be slower but more reliable. You can also try reducing workers to 1 or 2:

```bash
python scripts/precompute_embeddings.py \
    --celeba_root data/celeba \
    --num_workers 1
```

**Note:** The script now has improved error handling, so even if some images fail, it will:
- Continue processing
- Replace failed images with blank embeddings
- Report the total number of errors at the end
- Still produce valid embedding files

---

### 2. CUDA Out of Memory

**Symptom:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

**Option A: Reduce batch size**
```bash
# For embedding precomputation
python scripts/precompute_embeddings.py \
    --celeba_root data/celeba \
    --batch_size 64  # Reduced from 128

# For decoder training
# Edit configs/decoder_config.yaml
training:
  batch_size: 32  # Reduced from 64

# For FCLF training
# Edit configs/fclf_config.yaml
training:
  batch_size: 64  # Reduced from 128
```

**Option B: Use CPU**
```bash
python scripts/precompute_embeddings.py \
    --celeba_root data/celeba \
    --device cpu
```

---

### 3. CelebA Dataset Not Found

**Symptom:**
```
FileNotFoundError: Image directory not found: data/celeba/img_align_celeba
```

**Solution:**
Make sure you've downloaded and extracted CelebA correctly:

```bash
# Check directory structure
ls -la data/celeba/

# Should show:
# data/celeba/
#   ├── img_align_celeba/    (directory with 202,599 .jpg files)
#   └── list_attr_celeba.txt (text file)

# Verify images exist
ls data/celeba/img_align_celeba/ | head -10
```

If missing, follow the download instructions in `scripts/download_data.sh`.

---

### 4. CLIP Installation Issues

**Symptom:**
```
ModuleNotFoundError: No module named 'clip'
```

**Solution:**
```bash
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

If that fails, try:
```bash
# Clone and install manually
git clone https://github.com/openai/CLIP.git /tmp/CLIP
pip install /tmp/CLIP
```

---

### 5. Training Loss Not Decreasing

**Symptom:**
Loss stays constant or increases during FCLF training.

**Solutions:**

1. **Start without regularization:**
```bash
./scripts/train_fclf.sh --no_regularization
```

2. **Reduce learning rate:**
Edit `configs/fclf_config.yaml`:
```yaml
training:
  learning_rate: 5e-5  # Reduced from 1e-4
```

3. **Verify embeddings are loaded correctly:**
```python
import torch
embeddings = torch.load('data/embeddings/train_embeddings.pt')
print(embeddings.shape)  # Should be [162000, 512]
print(embeddings.mean(), embeddings.std())  # Should be reasonable values
```

---

### 6. Decoder Produces Poor Quality Images

**Symptom:**
Reconstructed images are blurry or distorted.

**Solutions:**

1. **Train for more epochs:**
Edit `configs/decoder_config.yaml`:
```yaml
training:
  num_epochs: 30  # Increased from 20
```

2. **Try VAE decoder:**
```bash
./scripts/train_decoder.sh --use_vae
```

3. **Check reconstruction loss:**
- Should decrease steadily
- Target: < 0.05 for acceptable quality
- Use TensorBoard to monitor: `tensorboard --logdir outputs/decoder/logs`

---

### 7. Slow Training on CPU

**Symptom:**
Training takes many hours on CPU.

**Solutions:**

1. **Use a smaller subset:**
Modify the dataset split in `src/data/celeba_dataset.py`:
```python
# Use first 50k images instead of 162k
if self.split == 'train':
    return range(0, 50000)  # Changed from 162000
```

2. **Use Google Colab (free GPU):**
- Upload your code to Google Drive
- Open a Colab notebook
- Mount Drive and run training there

3. **Reduce model size:**
Edit `configs/fclf_config.yaml`:
```yaml
model:
  hidden_dim: 128  # Reduced from 256
```

---

### 8. Permission Denied on Scripts

**Symptom:**
```
bash: ./scripts/train_decoder.sh: Permission denied
```

**Solution:**
```bash
chmod +x scripts/*.sh
```

Or run with bash explicitly:
```bash
bash scripts/train_decoder.sh
```

---

### 9. Import Errors in Scripts

**Symptom:**
```
ModuleNotFoundError: No module named 'src'
```

**Solution:**
Install the package in development mode:
```bash
pip install -e .
```

Or run scripts from the project root:
```bash
cd /path/to/learning-to-flow
python scripts/precompute_embeddings.py --celeba_root data/celeba
```

---

### 10. TensorBoard Not Showing Logs

**Symptom:**
TensorBoard shows "No dashboards are active"

**Solution:**
Make sure training has started and created log files:
```bash
# Check if logs exist
ls outputs/decoder/logs/
ls outputs/fclf/logs/

# Start TensorBoard pointing to the correct directory
tensorboard --logdir outputs/decoder/logs

# Or for FCLF
tensorboard --logdir outputs/fclf/logs
```

---

## Quick Diagnostic Commands

**Check environment:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import clip; print('CLIP installed')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
```

**Check data:**
```bash
ls -lh data/celeba/img_align_celeba/ | head
cat data/celeba/list_attr_celeba.txt | head
ls -lh data/embeddings/
```

**Check disk space:**
```bash
df -h .
du -sh data/
```

**Monitor GPU usage (if using CUDA):**
```bash
watch -n 1 nvidia-smi
```

---

## Getting Help

If you encounter an issue not listed here:

1. Check the error message carefully
2. Look at the relevant code file
3. Check TensorBoard logs if training-related
4. Try the simplified version first (e.g., --no_regularization)
5. Verify your data is correct
6. Check system resources (RAM, disk, GPU)

**For "Numpy is not available" specifically:**
- Always start with `--num_workers 0`
- This is the most reliable solution
- Training will still work, just slightly slower
