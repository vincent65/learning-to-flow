# Dataset Setup Guide

## ✅ What's in Git (Already Uploaded)

Your repository includes everything needed EXCEPT the actual dataset:

- ✅ Download scripts (automated + manual instructions)
- ✅ Data loading code (CelebA dataset class)
- ✅ Embedding precomputation script
- ✅ Documentation (data/README.md)
- ✅ .gitignore (excludes large files)

## ❌ What's NOT in Git (Correct!)

- ❌ CelebA images (1.7 GB) - Too large for Git
- ❌ CLIP embeddings (400 MB) - Generated locally
- ❌ Model checkpoints - Created during training

**This is the correct and standard practice!**

## 🎯 Why This Approach is Better

### Problems with uploading dataset to Git:
1. **GitHub rejects files > 100 MB**
2. **Repository becomes huge** (slow cloning)
3. **Wastes bandwidth** (everyone downloads same files)
4. **Violates copyright** (CelebA has usage terms)
5. **Not reproducible** (can't verify data integrity)

### Benefits of current approach:
1. ✅ **Fast repository** (only ~6 MB of code)
2. ✅ **Easy setup** (automated download script)
3. ✅ **Reproducible** (everyone gets same dataset)
4. ✅ **Standard practice** (like PyTorch, TensorFlow, etc.)
5. ✅ **Version control friendly** (only code changes tracked)

## 📥 How Others Will Download the Dataset

When someone clones your repo, they follow these steps:

```bash
# 1. Clone your repository (fast - only code)
git clone https://github.com/vincent65/learning-to-flow.git
cd learning-to-flow

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download CelebA dataset (one of two ways)

# Option A: Automated (requires Kaggle setup)
pip install kaggle
python scripts/download_celeba_auto.py

# Option B: Manual (follow instructions)
./scripts/download_data.sh

# 4. Precompute embeddings
python scripts/precompute_embeddings.py \
    --celeba_root data/celeba \
    --num_workers 0

# 5. Start training!
./scripts/train_decoder.sh
```

## 📊 What's on GitHub vs Local

| Item | In Git? | Size | Location |
|------|---------|------|----------|
| Source code | ✅ Yes | ~100 KB | `src/` |
| Scripts | ✅ Yes | ~20 KB | `scripts/` |
| Documentation | ✅ Yes | ~50 KB | `*.md` |
| Configs | ✅ Yes | ~2 KB | `configs/` |
| CelebA images | ❌ No | 1.7 GB | `data/celeba/` (download) |
| CLIP embeddings | ❌ No | 400 MB | `data/embeddings/` (generate) |
| Checkpoints | ❌ No | ~35 MB | `checkpoints/` (train) |

## 🔍 How to Verify

Check what's being tracked:

```bash
# See what's ignored (should include data/)
git status --ignored

# See repository size (should be small)
du -sh .git/

# See what's in the repo
git ls-files

# See what's ignored
cat .gitignore
```

## 🌟 This is How Professional Repos Work

Examples of popular ML repos that DON'T include datasets:

- **PyTorch**: Provides dataset loaders, not datasets
- **TensorFlow**: Provides downloaders, not data
- **Hugging Face**: Links to datasets, doesn't include them
- **OpenAI CLIP**: Provides model, not training data

Your repo follows the same professional standard! ✅

## 📝 Current Repository Status

```
✅ Code committed and pushed
✅ Download scripts included
✅ Documentation complete
✅ .gitignore configured correctly
✅ Dataset properly excluded
✅ Ready for others to use!
```

## 🚀 For Your CS229 Submission

When submitting your project:

1. **Share the GitHub repo**: https://github.com/vincent65/learning-to-flow
2. **Include README**: Already has setup instructions
3. **Document dataset source**: Already in data/README.md
4. **Provide trained models** (optional): Can share via Google Drive/Dropbox
5. **Results and figures**: Will be in `results/` after evaluation

Your instructors will:
1. Clone your repo
2. Download CelebA using your script
3. Run your training code
4. Evaluate your results

Everything they need is in the repo! 🎉
