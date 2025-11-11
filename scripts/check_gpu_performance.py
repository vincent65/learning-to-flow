"""
Quick GPU performance diagnostic script.
Run this on your VM to check if GPU is being utilized properly.
"""

import torch
import time
import subprocess

print("=" * 60)
print("GPU Performance Diagnostic")
print("=" * 60)

# 1. Check CUDA availability
print("\n[1] CUDA Check:")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  Number of GPUs: {torch.cuda.device_count()}")
    print(f"  Current device: {torch.cuda.current_device()}")
    print(f"  Device name: {torch.cuda.get_device_name(0)}")
else:
    print("  ❌ CUDA not available! Training will be slow!")
    exit(1)

# 2. Check GPU memory
print("\n[2] GPU Memory:")
mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
print(f"  Allocated: {mem_allocated:.2f} GB")
print(f"  Reserved: {mem_reserved:.2f} GB")

# 3. Check GPU utilization with nvidia-smi
print("\n[3] GPU Utilization (nvidia-smi):")
try:
    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit',
                           '--format=csv,noheader,nounits'],
                          capture_output=True, text=True)
    output = result.stdout.strip().split(',')
    print(f"  GPU Util: {output[0].strip()}%")
    print(f"  Memory Util: {output[1].strip()}%")
    print(f"  Temperature: {output[2].strip()}°C")
    print(f"  Power Draw: {output[3].strip()}W / {output[4].strip()}W")
except Exception as e:
    print(f"  ⚠ Could not run nvidia-smi: {e}")

# 4. Quick benchmark
print("\n[4] Quick Benchmark:")
print("  Running matrix multiplication benchmark...")

device = torch.device('cuda:0')
size = 4096

# Warmup
for _ in range(3):
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    c = torch.matmul(a, b)
    torch.cuda.synchronize()

# Benchmark
times = []
for _ in range(10):
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    torch.cuda.synchronize()
    start = time.time()
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    end = time.time()

    times.append(end - start)

avg_time = sum(times) / len(times)
tflops = (2 * size**3) / (avg_time * 1e12)

print(f"  Avg time: {avg_time*1000:.2f} ms")
print(f"  Performance: {tflops:.2f} TFLOPS")

# Expected performance:
# T4: ~8-10 TFLOPS (FP32)
# L4: ~30 TFLOPS (FP32), ~120 TFLOPS (TF32/Tensor cores)
print("\n  Expected performance:")
print("    T4:  8-10 TFLOPS (FP32)")
print("    L4:  30+ TFLOPS (FP32), 120+ TFLOPS with Tensor Cores")

if tflops < 15:
    print("\n  ⚠ Performance seems LOW for L4!")
    print("  Possible issues:")
    print("    - Not using Tensor Cores (mixed precision)")
    print("    - CPU bottleneck")
    print("    - Small batch size")
    print("    - Data loading bottleneck")
elif tflops < 25:
    print("\n  ✓ Performance is OK (but could be better with mixed precision)")
else:
    print("\n  ✓ Excellent performance! GPU is working well.")

# 5. Check PyTorch settings
print("\n[5] PyTorch Settings:")
print(f"  cuDNN enabled: {torch.backends.cudnn.enabled}")
print(f"  cuDNN benchmark: {torch.backends.cudnn.benchmark}")
print(f"  TF32 matmul: {torch.backends.cuda.matmul.allow_tf32}")
print(f"  TF32 cuDNN: {torch.backends.cudnn.allow_tf32}")

if not torch.backends.cudnn.benchmark:
    print("\n  💡 TIP: Enable cuDNN benchmark for faster training:")
    print("     torch.backends.cudnn.benchmark = True")

if not torch.backends.cuda.matmul.allow_tf32:
    print("\n  💡 TIP: Enable TF32 for L4 Tensor Cores (4x faster!):")
    print("     torch.backends.cuda.matmul.allow_tf32 = True")
    print("     torch.backends.cudnn.allow_tf32 = True")

print("\n" + "=" * 60)
print("Diagnostic complete!")
print("=" * 60)
