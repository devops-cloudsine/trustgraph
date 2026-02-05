#!/bin/bash
# Quick architecture detection and package compatibility check

echo "=========================================="
echo "TrustGraph Architecture Compatibility Check"
echo "=========================================="
echo ""

# Detect architecture
ARCH=$(uname -m)
echo "Current Architecture: $ARCH"
echo ""

# Check if on ARM64 or x86_64
if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
    echo "✓ Running on ARM64 (like current GB10 machine)"
    echo ""
    echo "Compatible packages:"
    echo "  ✓ onnxruntime (CPU-only)"
    echo "  ✓ torch==2.5.1 (generic)"
    echo "  ✗ onnxruntime-gpu (not available)"
    echo "  ✗ torch+cpu index (not available)"
    echo ""
    echo "Current setup: OPTIMIZED for this architecture"
    
elif [ "$ARCH" = "x86_64" ] || [ "$ARCH" = "amd64" ]; then
    echo "✓ Running on x86_64 (like AWS g6e.xlarge)"
    echo ""
    echo "Compatible packages:"
    echo "  ✓ onnxruntime (CPU)"
    echo "  ✓ onnxruntime-gpu (GPU-accelerated)"
    echo "  ✓ torch==2.5.1 (generic)"
    echo "  ✓ torch==2.5.1+cpu (AVX-optimized)"
    echo "  ✓ CUDA/cuDNN support"
    echo ""
    echo "Current setup: NEEDS UPDATE for GPU optimization"
    echo ""
    echo "To optimize for g6e.xlarge:"
    echo "  1. cp Containerfile.dev.multiarch Containerfile.dev"
    echo "  2. cp containers/Containerfile.hf.multiarch containers/Containerfile.hf"
    echo "  3. Rebuild containers"
else
    echo "⚠ Unknown architecture: $ARCH"
fi

echo ""
echo "=========================================="

# Check if GPU is available
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "GPU: Not detected (or nvidia-smi not installed)"
fi

echo "=========================================="
