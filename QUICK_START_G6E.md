# TrustGraph Quick Start - g6e.xlarge

**Quick reference for deploying TrustGraph on AWS g6e.xlarge**

---

## 🚀 30-Second Setup

```bash
# On your g6e.xlarge instance:
cd ~
git clone <your-repo-url> trustgraph
cd trustgraph

# Run automated setup
./setup_g6e.sh

# Start TrustGraph
cd deployment
./deploy-trustgraph.sh start
```

**That's it!** 🎉

---

## 📋 Prerequisites Checklist

```bash
# Check everything is ready:
docker --version              # ✓ Docker installed
docker compose version        # ✓ Compose V2 installed  
nvidia-smi                    # ✓ GPU accessible
./check_architecture.sh       # ✓ Architecture check
```

---

## 🔧 Essential Commands

```bash
cd ~/trustgraph/deployment

# Start services
./deploy-trustgraph.sh start

# Check status
./deploy-trustgraph.sh status

# View logs
./deploy-trustgraph.sh logs

# Restart unhealthy services
./deploy-trustgraph.sh recover

# Stop services
./deploy-trustgraph.sh stop
```

---

## 🌐 Access URLs

After startup, access these URLs (replace `<ip>` with your g6e public IP):

| Service | URL | Credentials |
|---------|-----|-------------|
| **Workbench UI** | `http://<ip>:8888` | - |
| **API Gateway** | `http://<ip>:8088` | - |
| **Grafana** | `http://<ip>:3001` | admin/admin |
| **MinIO Console** | `http://<ip>:9011` | minioadmin/minioadmin |

---

## ⚙️ Configuration Files

| File | Purpose |
|------|---------|
| `deployment/.env` | Environment variables (ports, paths, secrets) |
| `deployment/docker-compose.yaml` | Service definitions |
| `Containerfile.dev` | Development container (GPU-optimized for x86_64) |
| `containers/Containerfile.hf` | HuggingFace/AI container |

---

## 🔍 Troubleshooting

### Services won't start
```bash
cd ~/trustgraph/deployment
./deploy-trustgraph.sh status
./deploy-trustgraph.sh recover
docker compose logs --tail 100
```

### GPU not detected
```bash
nvidia-smi                    # Check GPU
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
sudo systemctl restart docker
```

### Port conflicts
```bash
# Edit deployment/.env and change port numbers:
nano deployment/.env
# Then restart:
./deploy-trustgraph.sh restart
```

### Out of disk space
```bash
docker system df              # Check usage
docker system prune -a        # Clean up
df -h                         # Check disk
```

---

## 📊 Monitoring

```bash
# Real-time monitoring (auto-restart unhealthy services)
cd ~/trustgraph/deployment
./deploy-trustgraph.sh watch

# Background monitoring
nohup ./deploy-trustgraph.sh watch > watchdog.log 2>&1 &

# Resource usage
docker stats

# Service logs
./deploy-trustgraph.sh logs vllm-vision
./deploy-trustgraph.sh logs api-gateway
```

---

## 🔐 Security (Production)

**Before going to production, update these in `deployment/.env`:**

```bash
# Generate new secrets
GATEWAY_SECRET=$(openssl rand -hex 32)
MCP_SERVER_SECRET=$(openssl rand -hex 32)

# Change MinIO credentials
MINIO_ROOT_USER=<your-username>
MINIO_ROOT_PASSWORD=<your-strong-password>

# Update Grafana password on first login
```

---

## 🏗️ Architecture Differences

| Component | ARM64 (GB10) | x86_64 (g6e.xlarge) |
|-----------|--------------|---------------------|
| ONNX Runtime | CPU-only | GPU-accelerated |
| PyTorch | Generic | AVX-optimized + CUDA |
| GPU | None | NVIDIA L40S (48GB) |
| Inference Speed | Baseline | 10-50x faster |

---

## 📝 Step-by-Step (Manual)

If `setup_g6e.sh` doesn't work, do this manually:

### 1. Clone & Update Architecture
```bash
cd ~
git clone <your-repo> trustgraph
cd trustgraph

# Use multi-arch Dockerfiles (or revert to x86_64)
cp Containerfile.dev.multiarch Containerfile.dev
cp containers/Containerfile.hf.multiarch containers/Containerfile.hf
```

### 2. Update Paths
```bash
# Edit deployment/.env
nano deployment/.env

# Update these lines:
FILES_TO_PARSE_HOST_PATH=/home/ubuntu/trustgraph/files_to_parse
TRUSTGRAPH_DEV_PATH=/home/ubuntu/trustgraph
VLLM_GPU_MEMORY_UTILIZATION=0.90

# Fix deploy script paths
sed -i "s|/home/cloudsineai/Desktop/trustgraph|$HOME/trustgraph|g" deployment/deploy-trustgraph.sh
```

### 3. Build & Start
```bash
cd deployment

# Optional: Build dev images
./deploy-trustgraph.sh build

# Start services
./deploy-trustgraph.sh start

# Check status
./deploy-trustgraph.sh status
```

---

## 🚦 Startup Order

TrustGraph starts services in phases:

1. **Infrastructure** (2-3 min): Zookeeper → Cassandra → Qdrant → Pulsar
2. **Initialization** (1-2 min): Schema creation, topic setup
3. **Application** (2-5 min): All processing services
4. **Ready** (5-10 min total): First-time startup downloads models

**Be patient!** ☕ First startup takes 10-20 minutes.

---

## 📚 Full Documentation

- **Detailed Guide**: `G6E_STARTUP_GUIDE.md`
- **Architecture Info**: `ARCHITECTURE_MIGRATION_GUIDE.md`
- **Main Docs**: `docs/README.md`
- **Quickstart**: `docs/README.quickstart-docker-compose.md`

---

## 🆘 Getting Help

```bash
# Check service health
cd ~/trustgraph/deployment
./deploy-trustgraph.sh health

# Export logs for debugging
docker compose logs > full-logs.txt

# Check architecture setup
../check_architecture.sh
```

---

## ✅ Post-Deployment Checklist

- [ ] All services showing "healthy" in `./deploy-trustgraph.sh status`
- [ ] GPU detected in vLLM logs: `docker compose logs vllm-vision | grep -i gpu`
- [ ] Workbench UI accessible at http://\<ip\>:8888
- [ ] API Gateway responding at http://\<ip\>:8088/health
- [ ] Security secrets changed in `deployment/.env`
- [ ] Firewall configured (only needed ports open)
- [ ] Watchdog running: `nohup ./deploy-trustgraph.sh watch &`

---

**🎯 You're all set! Start uploading documents and building your knowledge graph!**
