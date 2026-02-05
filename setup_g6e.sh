#!/bin/bash
# =============================================================================
# TrustGraph g6e.xlarge Quick Setup Script
# =============================================================================
# This script automates the setup of TrustGraph on AWS g6e.xlarge instances
# Run this after cloning the repository on your g6e instance
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_ok() { echo -e "${GREEN}✅ $1${NC}"; }
log_warn() { echo -e "${YELLOW}⚠️  $1${NC}"; }
log_err() { echo -e "${RED}❌ $1${NC}"; }
log_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }

echo "=========================================="
echo "TrustGraph g6e.xlarge Setup Script"
echo "=========================================="
echo ""

# Detect architecture
ARCH=$(uname -m)
log_info "Detected architecture: $ARCH"

if [ "$ARCH" != "x86_64" ] && [ "$ARCH" != "amd64" ]; then
    log_warn "This script is designed for x86_64 architecture (g6e.xlarge)"
    log_warn "You're running on $ARCH - results may vary"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if running on AWS
if [ -f /sys/devices/virtual/dmi/id/product_uuid ]; then
    UUID=$(sudo cat /sys/devices/virtual/dmi/id/product_uuid 2>/dev/null || echo "")
    if [[ $UUID == EC2* ]] || [[ $UUID == ec2* ]]; then
        log_ok "Running on AWS EC2"
    fi
fi

echo ""
echo "This script will:"
echo "  1. Check prerequisites (Docker, GPU)"
echo "  2. Configure architecture-specific Dockerfiles"
echo "  3. Update environment configuration"
echo "  4. Fix file paths for this instance"
echo "  5. Prepare for deployment"
echo ""
read -p "Continue? (Y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Nn]$ ]]; then
    exit 0
fi

# =============================================================================
# 1. Check Prerequisites
# =============================================================================
echo ""
echo "=== Step 1: Checking Prerequisites ==="
echo ""

# Check Docker
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version)
    log_ok "Docker installed: $DOCKER_VERSION"
else
    log_err "Docker not found!"
    log_info "Install with: curl -fsSL https://get.docker.com | sh"
    exit 1
fi

# Check Docker Compose
if docker compose version &> /dev/null; then
    COMPOSE_VERSION=$(docker compose version)
    log_ok "Docker Compose installed: $COMPOSE_VERSION"
else
    log_err "Docker Compose V2 not found!"
    log_info "Install with: sudo apt-get install docker-compose-plugin -y"
    exit 1
fi

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "Unknown")
    log_ok "GPU detected: $GPU_INFO"
    
    # Check Docker GPU support
    if docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
        log_ok "Docker GPU support enabled"
    else
        log_warn "Docker cannot access GPU"
        log_info "Install nvidia-container-toolkit and restart Docker"
    fi
else
    log_warn "nvidia-smi not found - GPU acceleration disabled"
    log_info "For GPU support, ensure NVIDIA drivers are installed"
fi

# =============================================================================
# 2. Configure Architecture-Specific Files
# =============================================================================
echo ""
echo "=== Step 2: Configuring for x86_64 Architecture ==="
echo ""

cd "$(dirname "$0")"
REPO_ROOT=$(pwd)

# Option A: Use multi-arch files if available
if [ -f "Containerfile.dev.multiarch" ]; then
    log_info "Using multi-arch Dockerfiles (auto-detect architecture)..."
    cp Containerfile.dev.multiarch Containerfile.dev
    log_ok "Updated Containerfile.dev"
    
    if [ -f "containers/Containerfile.hf.multiarch" ]; then
        cp containers/Containerfile.hf.multiarch containers/Containerfile.hf
        log_ok "Updated containers/Containerfile.hf"
    fi
else
    # Option B: Revert to x86_64-optimized versions from git
    log_info "Reverting to x86_64-optimized Dockerfiles from git..."
    
    if git rev-parse --git-dir > /dev/null 2>&1; then
        git show origin/master:Containerfile.dev > Containerfile.dev 2>/dev/null || \
            log_warn "Could not restore Containerfile.dev from git"
        
        git show origin/master:containers/Containerfile.hf > containers/Containerfile.hf 2>/dev/null || \
            log_warn "Could not restore containers/Containerfile.hf from git"
        
        log_ok "Restored x86_64-optimized Dockerfiles"
    else
        log_warn "Not a git repository - keeping current Dockerfiles"
    fi
fi

# =============================================================================
# 3. Update Environment Configuration
# =============================================================================
echo ""
echo "=== Step 3: Updating Environment Configuration ==="
echo ""

ENV_FILE="deployment/.env"

if [ -f "$ENV_FILE" ]; then
    log_info "Backing up existing .env to .env.backup"
    cp "$ENV_FILE" "${ENV_FILE}.backup"
fi

# Update paths in .env
log_info "Updating file paths in $ENV_FILE..."
sed -i "s|FILES_TO_PARSE_HOST_PATH=.*|FILES_TO_PARSE_HOST_PATH=$REPO_ROOT/files_to_parse|g" "$ENV_FILE"
sed -i "s|TRUSTGRAPH_DEV_PATH=.*|TRUSTGRAPH_DEV_PATH=$REPO_ROOT|g" "$ENV_FILE"

# Increase GPU memory utilization for g6e.xlarge (48GB GPU)
sed -i "s|VLLM_GPU_MEMORY_UTILIZATION=.*|VLLM_GPU_MEMORY_UTILIZATION=0.90|g" "$ENV_FILE"

log_ok "Updated environment configuration"

# Prompt for secrets update
echo ""
log_warn "Security: Default secrets detected in .env"
echo "For production, update these values:"
echo "  - GATEWAY_SECRET"
echo "  - MCP_SERVER_SECRET"
echo "  - MINIO_ROOT_USER/PASSWORD"
echo "  - HUGGING_FACE_HUB_TOKEN"
echo ""
read -p "Generate new secrets now? (Y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    if command -v openssl &> /dev/null; then
        GATEWAY_SECRET=$(openssl rand -hex 32)
        MCP_SECRET=$(openssl rand -hex 32)
        sed -i "s|GATEWAY_SECRET=.*|GATEWAY_SECRET=$GATEWAY_SECRET|g" "$ENV_FILE"
        sed -i "s|MCP_SERVER_SECRET=.*|MCP_SERVER_SECRET=$MCP_SECRET|g" "$ENV_FILE"
        log_ok "Generated new secrets"
    else
        log_warn "openssl not found - skipping secret generation"
    fi
fi

# =============================================================================
# 4. Fix File Paths in Deploy Script
# =============================================================================
echo ""
echo "=== Step 4: Fixing Deploy Script Paths ==="
echo ""

DEPLOY_SCRIPT="deployment/deploy-trustgraph.sh"
if [ -f "$DEPLOY_SCRIPT" ]; then
    log_info "Updating paths in $DEPLOY_SCRIPT..."
    sed -i "s|/home/cloudsineai/Desktop/trustgraph|$REPO_ROOT|g" "$DEPLOY_SCRIPT"
    log_ok "Updated deploy script paths"
else
    log_warn "$DEPLOY_SCRIPT not found"
fi

# =============================================================================
# 5. Create Required Directories
# =============================================================================
echo ""
echo "=== Step 5: Creating Required Directories ==="
echo ""

mkdir -p "$REPO_ROOT/files_to_parse"
mkdir -p "$REPO_ROOT/deployment/minio-data"

log_ok "Created required directories"

# =============================================================================
# 6. Check for GPU Docker Compose Config
# =============================================================================
echo ""
echo "=== Step 6: Checking GPU Configuration ==="
echo ""

if grep -q "nvidia" "deployment/docker-compose.yaml"; then
    log_ok "GPU configuration found in docker-compose.yaml"
else
    log_warn "GPU configuration not found in docker-compose.yaml"
    log_info "You may need to add GPU device reservations manually"
    log_info "See G6E_STARTUP_GUIDE.md for details"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
log_ok "TrustGraph is configured for x86_64 (g6e.xlarge)"
echo ""
echo "Next steps:"
echo ""
echo "  1. Review configuration:"
echo "     nano deployment/.env"
echo ""
echo "  2. (Optional) One-time Docker setup:"
echo "     cd deployment && sudo ./deploy-trustgraph.sh setup"
echo ""
echo "  3. Build images (if using dev containers):"
echo "     cd deployment && ./deploy-trustgraph.sh build"
echo ""
echo "  4. Start TrustGraph:"
echo "     cd deployment && ./deploy-trustgraph.sh start"
echo ""
echo "  5. Check status:"
echo "     cd deployment && ./deploy-trustgraph.sh status"
echo ""
echo "Access points (after startup):"
echo "  • Workbench UI: http://localhost:8888"
echo "  • API Gateway:  http://localhost:8088"
echo "  • Grafana:      http://localhost:3001"
echo "  • MinIO:        http://localhost:9011"
echo ""
echo "For detailed instructions, see: G6E_STARTUP_GUIDE.md"
echo "=========================================="
