# TrustGraph Multi-Architecture Support Guide

## Overview
This guide explains the architecture-specific changes made for ARM64 support and how to migrate between ARM64 (current GB10) and x86_64 (AWS g6e.xlarge) instances.

---

## Architecture-Specific Changes Summary

### 1. **Containerfile.dev**
**Location**: `/home/cloudsineai/Desktop/trustgraph/Containerfile.dev`

**ARM64 Change (Current):**
- **REMOVED**: GPU-optimized ONNX Runtime installation
```dockerfile
# These lines were removed:
python -m pip uninstall -y onnxruntime || true && \
python -m pip install --no-cache-dir "onnxruntime-gpu>=1.17,<1.19"
```

**Reason**: `onnxruntime-gpu` only provides x86_64 wheels. Not available for ARM64.

---

### 2. **containers/Containerfile.hf**
**Location**: `/home/cloudsineai/Desktop/trustgraph/containers/Containerfile.hf`

**ARM64 Change (Current):**
```dockerfile
# Before (x86_64):
RUN pip3 install torch==2.5.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# After (ARM64):
RUN pip3 install torch==2.5.1
```

**Reason**: PyTorch x86_64 CPU-optimized wheels aren't compatible with ARM64. Generic version is cross-platform.

---

### 3. **Base Image Versions**
- Updated from `1.4.23` → `1.6.5` (likely includes ARM64 support in base images)

---

## Migration Strategies

### **Strategy 1: Runtime Architecture Detection (Recommended)**

✅ **Pros:**
- Single Dockerfile works on both architectures
- No manual intervention needed
- Automatically adapts to environment

❌ **Cons:**
- Slightly longer build time (conditional logic)

**Implementation:** Use the provided multi-arch Dockerfiles:
- `Containerfile.dev.multiarch`
- `containers/Containerfile.hf.multiarch`

These detect architecture at build time and install appropriate packages.

---

### **Strategy 2: Separate Architecture-Specific Files**

✅ **Pros:**
- Cleaner, faster builds
- Explicit control

❌ **Cons:**
- Must maintain two sets of files
- Manual switching required

**Structure:**
```
Containerfile.dev.x86_64    # For g6e.xlarge
Containerfile.dev.arm64     # For current GB10
Containerfile.hf.x86_64
Containerfile.hf.arm64
```

Use build script to select appropriate file based on target.

---

### **Strategy 3: Build Args (Most Flexible)**

Use Docker build arguments to control architecture-specific features:

```dockerfile
ARG TARGETARCH=amd64
ARG ENABLE_GPU=true

RUN if [ "$ENABLE_GPU" = "true" ] && [ "$TARGETARCH" = "amd64" ]; then \
        pip install onnxruntime-gpu; \
    else \
        pip install onnxruntime; \
    fi
```

---

## Migration to g6e.xlarge (x86_64 + GPU)

### **What Changes Are Needed:**

#### **1. Containerfile.dev**
**Option A - Use multiarch file (recommended):**
```bash
cp Containerfile.dev.multiarch Containerfile.dev
```

**Option B - Manual restoration:**
Restore the GPU-optimized ONNX Runtime installation:
```dockerfile
# Add back after line 53:
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir "unstructured[all-docs]" pillow pytesseract Spire.PDF Spire.Doc requests asyncio minio && \
    python -m pip uninstall -y onnxruntime || true && \
    python -m pip install --no-cache-dir "onnxruntime-gpu>=1.17,<1.19"
```

#### **2. containers/Containerfile.hf**
**Option A - Use multiarch file (recommended):**
```bash
cp containers/Containerfile.hf.multiarch containers/Containerfile.hf
```

**Option B - Manual restoration:**
```dockerfile
# Change line 18-19 from:
RUN pip3 install torch==2.5.1

# Back to:
RUN pip3 install torch==2.5.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu
```

#### **3. deployment/.env**
Update GPU-related settings for g6e.xlarge:
```bash
# GPU Configuration for g6e.xlarge (NVIDIA L40S)
VLLM_GPU_MEMORY_UTILIZATION=0.90  # Can increase on g6e (48GB GPU)
VLLM_DTYPE=auto  # Will auto-detect GPU capabilities
VLLM_TENSOR_PARALLEL_SIZE=1  # Single GPU on g6e.xlarge

# Optional: Enable GPU for CUDA workloads
CUDA_VISIBLE_DEVICES=0
```

#### **4. docker-compose.yaml (if needed)**
Add GPU reservations for services that need it:
```yaml
services:
  vllm-vision:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

---

## Quick Migration Commands

### **Migrate to g6e.xlarge (x86_64 + GPU):**

```bash
# Use multi-arch Dockerfiles (recommended)
cp Containerfile.dev.multiarch Containerfile.dev
cp containers/Containerfile.hf.multiarch containers/Containerfile.hf

# Rebuild containers
docker-compose -f deployment/docker-compose.yaml build

# Or revert to x86_64-optimized versions
git show origin/master:Containerfile.dev > Containerfile.dev
git show origin/master:containers/Containerfile.hf > containers/Containerfile.hf
```

### **Stay on ARM64 (current setup):**
```bash
# Keep current files (already ARM64-compatible)
# No changes needed
```

---

## Package Compatibility Matrix

| Package | x86_64 | ARM64 | Notes |
|---------|--------|-------|-------|
| `onnxruntime-gpu` | ✅ Yes | ❌ No | GPU-accelerated, x86_64 only |
| `onnxruntime` | ✅ Yes | ✅ Yes | CPU-only, cross-platform |
| `torch==2.5.1+cpu` | ✅ Yes | ❌ No | x86_64 AVX optimizations |
| `torch==2.5.1` | ✅ Yes | ✅ Yes | Generic, cross-platform |
| `unstructured[all-docs]` | ✅ Yes | ✅ Yes | Cross-platform |
| CUDA/cuDNN | ✅ Yes | ❌ No | NVIDIA GPUs (x86_64) |

---

## Testing After Migration

### **On g6e.xlarge (x86_64):**
```bash
# Verify architecture
uname -m  # Should output: x86_64

# Verify GPU
nvidia-smi  # Should show NVIDIA L40S

# Check ONNX Runtime
python -c "import onnxruntime; print(onnxruntime.get_device())"
# Should show: GPU

# Check PyTorch
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Should show: 2.5.1+cpu False (for CPU-only) or True (if CUDA enabled)
```

### **On ARM64 (current GB10):**
```bash
# Verify architecture
uname -m  # Should output: aarch64

# Check ONNX Runtime
python -c "import onnxruntime; print(onnxruntime.get_device())"
# Should show: CPU

# Check PyTorch
python -c "import torch; print(torch.__version__)"
# Should show: 2.5.1
```

---

## Performance Expectations

### **ARM64 (Current - GB10):**
- ✅ Lower power consumption
- ✅ Good CPU performance
- ❌ No GPU acceleration
- ❌ Slower ML inference

### **x86_64 + GPU (g6e.xlarge):**
- ✅ NVIDIA L40S GPU (48GB VRAM)
- ✅ Much faster ML inference (10-50x for LLMs)
- ✅ Native support for all ML libraries
- ❌ Higher cost
- ✅ Better for production workloads

---

## Recommended Approach

**Use the multi-arch Dockerfiles (`*.multiarch` files)** - they automatically detect and adapt to the architecture at build time. This gives you:

1. ✅ Single codebase for both environments
2. ✅ No manual intervention needed
3. ✅ Easy migration between architectures
4. ✅ Optimal performance on each platform

Simply:
```bash
mv Containerfile.dev.multiarch Containerfile.dev
mv containers/Containerfile.hf.multiarch containers/Containerfile.hf
```

Then rebuild and deploy on any architecture!

---

## Questions or Issues?

Check package compatibility:
- ONNX Runtime: https://onnxruntime.ai/
- PyTorch: https://pytorch.org/get-started/locally/
- Docker buildx (multi-arch): https://docs.docker.com/buildx/working-with-buildx/
