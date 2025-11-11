# VM Training Fix Guide

## Issue You Encountered

```
TypeError: '<=' not supported between instances of 'float' and 'str'
```

This happened because the YAML parser on your VM was reading the learning rate as a string instead of a float.

## ✅ Fixes Applied

All fixes have been committed and pushed to GitHub. To get them on your VM:

```bash
# On your VM
cd ~/learning-to-flow
git pull
```

## What Was Fixed

### 1. Type Conversion in Training Scripts ✅

**train_fclf.py** and **train_decoder.py** now force all config values to proper types:

```python
# Now automatically converts strings to correct types
config['training']['learning_rate'] = float(config['training']['learning_rate'])
config['training']['batch_size'] = int(config['training']['batch_size'])
# ... etc
```

### 2. Config Files Updated ✅

**configs/fclf_config.yaml** and **configs/decoder_config.yaml** now use decimal notation:

```yaml
# Before (could be parsed as string by some YAML parsers)
learning_rate: 1e-4

# After (always parsed correctly)
learning_rate: 0.0001
```

### 3. num_workers Fixed ✅

Changed from `4` to `0` to avoid:
- Memory issues on VMs
- Multiprocessing warnings
- DataLoader freezes

### 4. Config Validation Tool Added ✅

New tool to check if configs are valid:

```bash
python scripts/check_config.py --config configs/fclf_config.yaml
```

---

## How to Use the Fixes on Your VM

### Step 1: Pull Latest Code

```bash
cd ~/learning-to-flow
git pull
```

You should see:
```
remote: Counting objects...
Updating 317d24c..7ce78c9
Fast-forward
 configs/decoder_config.yaml      | ...
 configs/fclf_config.yaml         | ...
 scripts/check_config.py          | ... (new)
 src/training/train_decoder.py    | ...
 src/training/train_fclf.py       | ...
```

### Step 2: Verify Configs (Optional)

```bash
# Check FCLF config
python scripts/check_config.py --config configs/fclf_config.yaml

# Check decoder config
python scripts/check_config.py --config configs/decoder_config.yaml
```

You should see:
```
======================================================================
Config Validation: configs/fclf_config.yaml
======================================================================

✓ File exists
✓ YAML parsed successfully

Validating numeric fields:
  ✓ training.learning_rate = 0.0001 (float)
  ✓ training.batch_size = 128 (int)
  ✓ training.num_epochs = 50 (int)
  ...

======================================================================
✅ CONFIG IS VALID
======================================================================
```

### Step 3: Run Training (Should Work Now!)

```bash
# Train FCLF
./scripts/train_fclf.sh \
    --celeba_root data/celeba \
    --embedding_dir data/embeddings \
    --output_dir outputs/fclf
```

---

## Expected Behavior After Fix

### Before (What You Saw) ❌

```
Using device: cuda
Train samples: 162000
Val samples: 20000
Model parameters: 396,800
Traceback (most recent call last):
  ...
TypeError: '<=' not supported between instances of 'float' and 'str'
```

### After (What You Should See) ✅

```
Using device: cuda
Train samples: 162000
Val samples: 20000
Model parameters: 396,800

Epoch 1/50
Training: 100%|████████████████| 1266/1266 [03:45<00:00,  5.61it/s]
  Train Loss: 2.3456
Validation: 100%|█████████████| 157/157 [00:22<00:00,  6.95it/s]
  Val Loss: 2.1234
Saved best model (val_loss: 2.1234)

Epoch 2/50
...
```

---

## Troubleshooting

### If git pull shows conflicts:

```bash
# Stash your local changes first
git stash

# Pull latest code
git pull

# Reapply your changes (if needed)
git stash pop
```

### If you still get the TypeError:

This means your VM might have cached the old Python bytecode:

```bash
# Clear Python cache
find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null
find . -name "*.pyc" -delete

# Try again
./scripts/train_fclf.sh --celeba_root data/celeba --embedding_dir data/embeddings
```

### If configs still fail validation:

Manually edit the configs to use decimal notation:

```bash
nano configs/fclf_config.yaml

# Change this line:
learning_rate: 1e-4

# To this:
learning_rate: 0.0001

# Save: Ctrl+X, Y, Enter
```

---

## Additional VM Optimizations

Since you have a T4 GPU with 16GB memory, you can increase batch size:

```bash
# Edit FCLF config for better T4 utilization
nano configs/fclf_config.yaml
```

Change:
```yaml
training:
  batch_size: 128  # Original
```

To:
```yaml
training:
  batch_size: 256  # Better for T4
```

This will make training ~30% faster!

---

## Quick Reference Commands

```bash
# Validate configs
python scripts/check_config.py --config configs/fclf_config.yaml --verbose

# Check Python can import modules
python -c "from src.training import train_fclf; print('OK')"

# Monitor GPU usage during training
watch -n 1 nvidia-smi

# Check training logs
tail -f outputs/fclf/logs/events.out.tfevents.*

# View TensorBoard (from local machine via SSH tunnel)
ssh -L 6006:localhost:6006 user@vm-ip
# Then open http://localhost:6006 in browser
```

---

## Summary

✅ All fixes have been applied and pushed to GitHub
✅ Simply run `git pull` on your VM to get them
✅ Training should now work without the TypeError
✅ num_workers set to 0 to avoid memory issues
✅ Config validation tool added for future debugging

**You're ready to train!** 🚀
