#!/bin/bash
# Quick fix for dtype issue in compute_paper_metrics.py

sed -i 's/attr_change = target_attrs\[i\] - original_attrs\[i\]  # \[5\]/attr_change = (target_attrs[i] - original_attrs[i]).to(original_emb.dtype)  # [5], match dtype/' scripts/compute_paper_metrics.py

echo "✅ Fixed! Now run your script again."
