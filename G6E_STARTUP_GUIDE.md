# TrustGraph Startup Guide for AWS g6e.xlarge

Complete guide to deploy TrustGraph on an x86_64 AWS g6e.xlarge instance (with NVIDIA L40S GPU).

---

## Prerequisites

### 1. System Requirements
- **Instance**: AWS g6e.xlarge (or similar x86_64 with GPU)
- **OS**: Ubuntu 22.04+ (recommended)
- **GPU**: NVIDIA L40S (48GB VRAM)
- **Storage**: 100GB+ recommended
- **Network**: Ports 8088, 8888, 3001 open (or as configured)

### 2. Install Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
newgrp docker

# Install Docker Compose V2
sudo apt-get install docker-compose-plugin -y

# Install NVIDIA Container Toolkit (for GPU support)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify GPU is accessible
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

---

## Step 1: Clone Repository

```bash
# Clone your repository
cd ~
git clone <your-repo-url> trustgraph
cd trustgraph

# Verify you're on the correct branch
git branch
git status
```

---

## Step 2: Architecture Optimization (x86_64 + GPU)

Your codebase currently has ARM64 optimizations. For g6e.xlarge (x86_64 with GPU), you have two options:

### **Option A: Use Multi-Arch Dockerfiles (Recommended)**

The multi-arch files auto-detect architecture and install optimal packages:

```bash
# Use multi-arch Dockerfiles (they auto-detect x86_64 and install GPU support)
cp Containerfile.dev.multiarch Containerfile.dev
cp containers/Containerfile.hf.multiarch containers/Containerfile.hf
```

### **Option B: Revert to x86_64-Optimized Version**

If multi-arch files don't exist, revert to the original x86_64 versions:

```bash
# Restore x86_64-optimized Dockerfiles from remote
git show origin/master:Containerfile.dev > Containerfile.dev
git show origin/master:containers/Containerfile.hf > containers/Containerfile.hf
```

---

## Step 3: Configure Environment

### 3.1 Update deployment/.env for g6e.xlarge

```bash
cd deployment
nano .env
```

Update these values:

```bash
# =============================================================================
# TrustGraph Docker Compose Environment Configuration - g6e.xlarge
# =============================================================================

# -----------------------------------------------------------------------------
# VLLM / LLM Configuration (for NVIDIA L40S GPU)
# -----------------------------------------------------------------------------
VLLM_MODEL=google/gemma-3-4b-it
VLLM_MAX_MODEL_LEN=16384
VLLM_GPU_MEMORY_UTILIZATION=0.90   # Increased for L40S (48GB)
VLLM_DTYPE=auto                     # Auto-detect GPU capabilities
VLLM_PORT=8005

# -----------------------------------------------------------------------------
# Hugging Face Configuration
# -----------------------------------------------------------------------------
HUGGING_FACE_HUB_TOKEN=<your-token-here>

# -----------------------------------------------------------------------------
# File Paths (update to your g6e instance paths)
# -----------------------------------------------------------------------------
FILES_TO_PARSE_HOST_PATH=/home/ubuntu/trustgraph/files_to_parse
TRUSTGRAPH_DEV_PATH=/home/ubuntu/trustgraph

# -----------------------------------------------------------------------------
# TrustGraph Version
# -----------------------------------------------------------------------------
TRUSTGRAPH_VERSION=1.6.5

# -----------------------------------------------------------------------------
# MinIO Configuration
# -----------------------------------------------------------------------------
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin
MINIO_OCR_BUCKET=ocr-images

# -----------------------------------------------------------------------------
# API Secrets (UPDATE THESE FOR PRODUCTION!)
# -----------------------------------------------------------------------------
GATEWAY_SECRET=$(openssl rand -hex 32)
MCP_SERVER_SECRET=$(openssl rand -hex 32)
OPENAI_BASE_URL=http://vllm-vision:8000/v1
OPENAI_TOKEN=dummy-token

# -----------------------------------------------------------------------------
# Port Configuration
# -----------------------------------------------------------------------------
MCP_SERVER_PORT=8006
GRAFANA_PORT=3001
MINIO_PORT=9010
MINIO_CONSOLE_PORT=9011
```

### 3.2 Update deploy-trustgraph.sh paths

```bash
nano deploy-trustgraph.sh
```

Change line 18:
```bash
# Before:
COMPOSE_DIR="/home/cloudsineai/Desktop/trustgraph/deployment"

# After (for g6e):
COMPOSE_DIR="$HOME/trustgraph/deployment"
```

Also update lines 282-291 (build_images function):
```bash
# Replace all hardcoded paths with $HOME/trustgraph
```

Or use this quick fix:
```bash
sed -i "s|/home/cloudsineai/Desktop/trustgraph|$HOME/trustgraph|g" deploy-trustgraph.sh
```

---

## Step 4: Enable GPU Support in docker-compose.yaml

Edit `deployment/docker-compose.yaml` to add GPU support to vLLM service:

```bash
cd ~/trustgraph/deployment
nano docker-compose.yaml
```

Find the `vllm-vision` service (or similar GPU service) and add GPU configuration:

```yaml
vllm-vision:
  image: vllm/vllm-openai:latest
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1  # g6e.xlarge has 1 GPU
            capabilities: [gpu]
  environment:
    - VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.90}
  # ... rest of config
```

---

## Step 5: One-Time Docker Setup (Optional but Recommended)

Configure Docker for production resilience:

```bash
cd ~/trustgraph/deployment
sudo ./deploy-trustgraph.sh setup
```

This configures:
- ✅ Live-restore (containers survive Docker restarts)
- ✅ Log rotation
- ✅ Increased file limits

---

## Step 6: Build Images (If Using Dev Containers)

If using local development images:

```bash
cd ~/trustgraph/deployment

# Build dev images with GPU support
./deploy-trustgraph.sh build
```

**Note**: This step is only needed if you're using `Containerfile.dev` or custom images. If using published Docker Hub images, skip this step.

---

## Step 7: Start TrustGraph

### Method 1: Using Deploy Script (Recommended)

```bash
cd ~/trustgraph/deployment
./deploy-trustgraph.sh start
```

This will:
1. Start infrastructure (Zookeeper, Cassandra, Qdrant, Pulsar)
2. Wait for services to be ready
3. Run initialization jobs
4. Start all services

### Method 2: Manual Docker Compose

```bash
cd ~/trustgraph/deployment
docker compose up -d
```

---

## Step 8: Verify Deployment

### Check Service Status
```bash
cd ~/trustgraph/deployment
./deploy-trustgraph.sh status
```

### Check GPU Access
```bash
# Verify vLLM can see GPU
docker compose logs vllm-vision | grep -i gpu

# Should show GPU detection and memory allocation
```

### Test API Endpoints
```bash
# API Gateway
curl http://localhost:8088/health

# Workbench UI
curl http://localhost:8888

# MinIO Console
curl http://localhost:9011
```

---

## Step 9: Monitor Services

### Real-time Logs
```bash
# All services
./deploy-trustgraph.sh logs

# Specific service
./deploy-trustgraph.sh logs vllm-vision
./deploy-trustgraph.sh logs api-gateway
```

### Health Monitoring
```bash
# Check health once
./deploy-trustgraph.sh health

# Continuous monitoring (auto-restart unhealthy services)
./deploy-trustgraph.sh watch

# Run watchdog in background
nohup ./deploy-trustgraph.sh watch &
```

---

## Access Points

After successful startup:

| Service | URL | Description |
|---------|-----|-------------|
| **Workbench UI** | http://\<g6e-ip\>:8888 | Main web interface |
| **API Gateway** | http://\<g6e-ip\>:8088 | REST API |
| **Grafana** | http://\<g6e-ip\>:3001 | Monitoring dashboard |
| **MinIO Console** | http://\<g6e-ip\>:9011 | Object storage UI |
| **MCP Server** | http://\<g6e-ip\>:8006 | MCP API endpoint |

**Default Credentials:**
- MinIO: `minioadmin` / `minioadmin`
- Grafana: `admin` / `admin` (change on first login)

---

## Common Issues & Solutions

### Issue 1: GPU Not Detected
```bash
# Check NVIDIA drivers
nvidia-smi

# Restart Docker with GPU support
sudo systemctl restart docker

# Test GPU in container
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

### Issue 2: Port Already in Use
Edit `deployment/.env` and change conflicting ports:
```bash
GRAFANA_PORT=3002
MINIO_PORT=9020
# etc.
```

### Issue 3: Services Unhealthy
```bash
# Restart unhealthy services
./deploy-trustgraph.sh recover

# Or restart specific service
docker compose restart <service-name>
```

### Issue 4: Out of Memory
```bash
# Check memory usage
docker stats

# Reduce GPU memory utilization in .env
VLLM_GPU_MEMORY_UTILIZATION=0.80  # Lower from 0.90
```

### Issue 5: Slow Startup
This is normal on first run - Docker needs to:
- Pull images (~5-10 minutes)
- Download ML models (~2-5 minutes)
- Initialize databases (~2-3 minutes)

**Total first startup**: 10-20 minutes

---

## Management Commands

```bash
# Start services
./deploy-trustgraph.sh start

# Stop services
./deploy-trustgraph.sh stop

# Restart services
./deploy-trustgraph.sh restart

# Check status
./deploy-trustgraph.sh status

# Recover unhealthy services
./deploy-trustgraph.sh recover

# Watch and auto-recover (every 60s)
./deploy-trustgraph.sh watch

# View logs
./deploy-trustgraph.sh logs [service-name]

# Rebuild images
./deploy-trustgraph.sh build
```

---

## Performance Tuning for g6e.xlarge

### GPU Optimization
```bash
# In deployment/.env
VLLM_GPU_MEMORY_UTILIZATION=0.90  # Use most of 48GB
VLLM_MAX_MODEL_LEN=32768          # Increase context window
VLLM_TENSOR_PARALLEL_SIZE=1       # Single GPU
```

### Memory Optimization
```bash
# Increase Docker memory limits in docker-compose.yaml
services:
  vllm-vision:
    deploy:
      resources:
        limits:
          memory: 40G  # Allow more RAM for model loading
```

---

## Security Checklist for Production

- [ ] Change `GATEWAY_SECRET` and `MCP_SERVER_SECRET` in `.env`
- [ ] Change MinIO credentials (`MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD`)
- [ ] Update Grafana admin password
- [ ] Configure firewall to restrict ports
- [ ] Enable HTTPS/TLS for public-facing services
- [ ] Rotate Hugging Face tokens regularly
- [ ] Set up automated backups for Cassandra/MinIO data

---

## Backup & Restore

### Backup Data
```bash
# Cassandra data
docker compose exec cassandra nodetool snapshot

# MinIO data
cp -r deployment/minio-data ~/trustgraph-backup/

# Qdrant data
docker compose exec qdrant /bin/sh -c "tar czf /backup.tar.gz /qdrant/storage"
docker cp deployment-qdrant-1:/backup.tar.gz ~/trustgraph-backup/
```

### Restore Data
```bash
# Stop services
./deploy-trustgraph.sh stop

# Restore MinIO
rm -rf deployment/minio-data
cp -r ~/trustgraph-backup/minio-data deployment/

# Restart
./deploy-trustgraph.sh start
```

---

## Troubleshooting

### View Full System Logs
```bash
docker compose logs -f --tail 500 > trustgraph-logs.txt
```

### Check Resource Usage
```bash
# CPU/Memory/GPU
docker stats

# Disk usage
df -h
docker system df
```

### Clean Up Disk Space
```bash
# Remove unused images
docker image prune -a

# Remove stopped containers
docker container prune

# Remove unused volumes
docker volume prune
```

---

## Next Steps

1. ✅ Deploy TrustGraph
2. ✅ Verify GPU acceleration
3. 📚 Upload documents via Workbench UI
4. 🔍 Test knowledge graph queries
5. 📊 Monitor performance in Grafana
6. 🔒 Secure production deployment

---

## Support

- Documentation: `docs/README.md`
- Quickstart: `docs/README.quickstart-docker-compose.md`
- API Docs: `docs/apis/`
- Issues: Check logs with `./deploy-trustgraph.sh logs`
