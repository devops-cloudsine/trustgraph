#!/bin/bash
# =============================================================================
# TrustGraph Docker Swarm Production Deployment Script
# =============================================================================
# Usage:
#   ./deploy-swarm.sh init              # Initialize swarm (run on manager node)
#   ./deploy-swarm.sh join-worker       # Get worker join token
#   ./deploy-swarm.sh join-manager      # Get manager join token
#   ./deploy-swarm.sh secrets           # Create Docker secrets (interactive)
#   ./deploy-swarm.sh labels            # Label current node for all roles
#   ./deploy-swarm.sh deploy            # Deploy the stack
#   ./deploy-swarm.sh update            # Update running stack
#   ./deploy-swarm.sh status            # Show stack status
#   ./deploy-swarm.sh services          # List all services
#   ./deploy-swarm.sh logs <service>    # Show service logs
#   ./deploy-swarm.sh scale <svc> <n>   # Scale a service
#   ./deploy-swarm.sh rollback <svc>    # Rollback a service
#   ./deploy-swarm.sh remove            # Remove the stack
#   ./deploy-swarm.sh cleanup           # Full cleanup (stack + secrets + networks)
# =============================================================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STACK_FILE="${SCRIPT_DIR}/docker-stack.yaml"
STACK_NAME="trustgraph"
ENV_FILE="${SCRIPT_DIR}/.env"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_ok() { echo -e "${GREEN}✅ $1${NC}"; }
log_warn() { echo -e "${YELLOW}⚠️  $1${NC}"; }
log_err() { echo -e "${RED}❌ $1${NC}"; }
log_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
log_step() { echo -e "${CYAN}➡️  $1${NC}"; }

# =============================================================================
# SWARM INITIALIZATION
# =============================================================================
init_swarm() {
    log_step "Initializing Docker Swarm..."
    
    if docker info 2>/dev/null | grep -q "Swarm: active"; then
        log_warn "Swarm is already active"
        docker node ls
        return 0
    fi
    
    # Get the primary IP for advertising
    local advertise_addr
    advertise_addr=$(hostname -I | awk '{print $1}')
    
    echo "Detected IP: $advertise_addr"
    read -rp "Use this IP for swarm advertise address? [Y/n]: " confirm
    
    if [[ "${confirm,,}" == "n" ]]; then
        read -rp "Enter advertise address: " advertise_addr
    fi
    
    docker swarm init --advertise-addr "$advertise_addr"
    
    log_ok "Swarm initialized!"
    echo ""
    echo "Next steps:"
    echo "  1. Label this node:     ./deploy-swarm.sh labels"
    echo "  2. Create secrets:      ./deploy-swarm.sh secrets"
    echo "  3. Deploy stack:        ./deploy-swarm.sh deploy"
    echo ""
    echo "To add worker nodes, run on other machines:"
    docker swarm join-token worker | grep "docker swarm join"
}

get_join_token() {
    local role=${1:-worker}
    
    if ! docker info 2>/dev/null | grep -q "Swarm: active"; then
        log_err "Swarm is not active. Run: ./deploy-swarm.sh init"
        exit 1
    fi
    
    echo ""
    echo "Run this command on nodes you want to join as ${role}:"
    echo ""
    docker swarm join-token "$role" | grep "docker swarm join"
    echo ""
}

# =============================================================================
# NODE LABELS
# =============================================================================
label_nodes() {
    log_step "Labeling nodes for service placement..."
    
    local node_id
    node_id=$(docker node ls --filter "role=manager" -q | head -1)
    
    if [ -z "$node_id" ]; then
        log_err "No manager node found"
        exit 1
    fi
    
    echo ""
    echo "Available nodes:"
    docker node ls
    echo ""
    
    read -rp "Enter node ID/name to label (or 'self' for current): " target_node
    
    if [ "$target_node" == "self" ]; then
        target_node=$(docker node ls --filter "role=manager" -q | head -1)
    fi
    
    echo ""
    echo "Available labels:"
    echo "  1. trustgraph.storage=true  - Can run Cassandra, Qdrant, MinIO"
    echo "  2. trustgraph.pulsar=true   - Can run Pulsar cluster (ZK, Bookie, Broker)"
    echo "  3. trustgraph.gpu=true      - Has GPU for vLLM"
    echo "  4. All of the above"
    echo ""
    read -rp "Select option [1-4]: " label_choice
    
    case $label_choice in
        1)
            docker node update --label-add trustgraph.storage=true "$target_node"
            log_ok "Added storage label to $target_node"
            ;;
        2)
            docker node update --label-add trustgraph.pulsar=true "$target_node"
            log_ok "Added pulsar label to $target_node"
            ;;
        3)
            docker node update --label-add trustgraph.gpu=true "$target_node"
            log_ok "Added GPU label to $target_node"
            ;;
        4)
            docker node update --label-add trustgraph.storage=true "$target_node"
            docker node update --label-add trustgraph.pulsar=true "$target_node"
            docker node update --label-add trustgraph.gpu=true "$target_node"
            log_ok "Added all labels to $target_node"
            ;;
        *)
            log_err "Invalid option"
            exit 1
            ;;
    esac
    
    echo ""
    echo "Current node labels:"
    docker node inspect "$target_node" --format '{{ range $k, $v := .Spec.Labels }}{{ $k }}={{ $v }} {{ end }}'
    echo ""
}

label_current_node_all() {
    log_step "Labeling current node with all roles..."
    
    local node_id
    node_id=$(docker info --format '{{.Swarm.NodeID}}')
    
    docker node update --label-add trustgraph.storage=true "$node_id"
    docker node update --label-add trustgraph.pulsar=true "$node_id"
    docker node update --label-add trustgraph.gpu=true "$node_id"
    
    log_ok "All labels added to current node"
}

show_nodes() {
    echo ""
    echo "=== Swarm Nodes ==="
    docker node ls
    echo ""
    echo "=== Node Labels ==="
    for node in $(docker node ls -q); do
        local name labels
        name=$(docker node inspect "$node" --format '{{.Description.Hostname}}')
        labels=$(docker node inspect "$node" --format '{{ range $k, $v := .Spec.Labels }}{{ $k }}={{ $v }} {{ end }}')
        echo "  $name: $labels"
    done
    echo ""
}

# =============================================================================
# DOCKER SECRETS
# =============================================================================
create_secrets() {
    log_step "Creating Docker secrets..."
    echo ""
    echo "This will create the following secrets:"
    echo "  - gateway_secret"
    echo "  - openai_token"
    echo "  - mcp_server_secret"
    echo "  - huggingface_token"
    echo "  - minio_root_user"
    echo "  - minio_root_password"
    echo "  - grafana_admin_password"
    echo ""
    
    # Gateway secret
    if docker secret inspect gateway_secret >/dev/null 2>&1; then
        log_warn "gateway_secret already exists (skipping)"
    else
        read -rsp "Enter GATEWAY_SECRET: " secret_val
        echo ""
        echo -n "$secret_val" | docker secret create gateway_secret -
        log_ok "Created gateway_secret"
    fi
    
    # OpenAI token
    if docker secret inspect openai_token >/dev/null 2>&1; then
        log_warn "openai_token already exists (skipping)"
    else
        read -rsp "Enter OPENAI_TOKEN: " secret_val
        echo ""
        echo -n "$secret_val" | docker secret create openai_token -
        log_ok "Created openai_token"
    fi
    
    # MCP server secret
    if docker secret inspect mcp_server_secret >/dev/null 2>&1; then
        log_warn "mcp_server_secret already exists (skipping)"
    else
        read -rsp "Enter MCP_SERVER_SECRET: " secret_val
        echo ""
        echo -n "$secret_val" | docker secret create mcp_server_secret -
        log_ok "Created mcp_server_secret"
    fi
    
    # Huggingface token
    if docker secret inspect huggingface_token >/dev/null 2>&1; then
        log_warn "huggingface_token already exists (skipping)"
    else
        read -rsp "Enter HUGGING_FACE_HUB_TOKEN: " secret_val
        echo ""
        echo -n "$secret_val" | docker secret create huggingface_token -
        log_ok "Created huggingface_token"
    fi
    
    # MinIO credentials
    if docker secret inspect minio_root_user >/dev/null 2>&1; then
        log_warn "minio_root_user already exists (skipping)"
    else
        read -rp "Enter MINIO_ROOT_USER [minioadmin]: " secret_val
        secret_val=${secret_val:-minioadmin}
        echo -n "$secret_val" | docker secret create minio_root_user -
        log_ok "Created minio_root_user"
    fi
    
    if docker secret inspect minio_root_password >/dev/null 2>&1; then
        log_warn "minio_root_password already exists (skipping)"
    else
        read -rsp "Enter MINIO_ROOT_PASSWORD [minioadmin]: " secret_val
        echo ""
        secret_val=${secret_val:-minioadmin}
        echo -n "$secret_val" | docker secret create minio_root_password -
        log_ok "Created minio_root_password"
    fi
    
    # Grafana admin password
    if docker secret inspect grafana_admin_password >/dev/null 2>&1; then
        log_warn "grafana_admin_password already exists (skipping)"
    else
        read -rsp "Enter GRAFANA_ADMIN_PASSWORD [admin]: " secret_val
        echo ""
        secret_val=${secret_val:-admin}
        echo -n "$secret_val" | docker secret create grafana_admin_password -
        log_ok "Created grafana_admin_password"
    fi
    
    echo ""
    log_ok "All secrets created!"
    echo ""
    echo "Current secrets:"
    docker secret ls
}

remove_secrets() {
    log_step "Removing Docker secrets..."
    
    local secrets="gateway_secret openai_token mcp_server_secret huggingface_token minio_root_user minio_root_password grafana_admin_password"
    
    for secret in $secrets; do
        if docker secret inspect "$secret" >/dev/null 2>&1; then
            docker secret rm "$secret" 2>/dev/null || log_warn "Could not remove $secret (may be in use)"
        fi
    done
    
    log_ok "Secrets removed"
}

# =============================================================================
# STACK DEPLOYMENT
# =============================================================================
deploy_stack() {
    log_step "Deploying TrustGraph stack..."
    
    # Check prerequisites
    if ! docker info 2>/dev/null | grep -q "Swarm: active"; then
        log_err "Swarm is not active. Run: ./deploy-swarm.sh init"
        exit 1
    fi
    
    # Check if secrets exist
    local required_secrets="gateway_secret openai_token mcp_server_secret huggingface_token minio_root_user minio_root_password grafana_admin_password"
    local missing_secrets=""
    
    for secret in $required_secrets; do
        if ! docker secret inspect "$secret" >/dev/null 2>&1; then
            missing_secrets="$missing_secrets $secret"
        fi
    done
    
    if [ -n "$missing_secrets" ]; then
        log_err "Missing secrets:$missing_secrets"
        echo "Run: ./deploy-swarm.sh secrets"
        exit 1
    fi
    
    # Check for labeled nodes
    local storage_nodes pulsar_nodes
    storage_nodes=$(docker node ls -q --filter "node.label=trustgraph.storage=true" | wc -l)
    pulsar_nodes=$(docker node ls -q --filter "node.label=trustgraph.pulsar=true" | wc -l)
    
    if [ "$storage_nodes" -eq 0 ]; then
        log_warn "No nodes labeled with trustgraph.storage=true"
        log_info "Run: ./deploy-swarm.sh labels"
    fi
    
    if [ "$pulsar_nodes" -eq 0 ]; then
        log_warn "No nodes labeled with trustgraph.pulsar=true"
        log_info "Run: ./deploy-swarm.sh labels"
    fi
    
    # Load environment file if exists
    if [ -f "$ENV_FILE" ]; then
        log_info "Loading environment from $ENV_FILE"
        set -a
        source "$ENV_FILE"
        set +a
    fi
    
    # Deploy
    docker stack deploy -c "$STACK_FILE" "$STACK_NAME" --with-registry-auth
    
    log_ok "Stack deployed!"
    
    # Install recovery watchdog (runs once, enables auto-start on boot)
    local recovery_script="${SCRIPT_DIR}/recovery/recovery-master.sh"
    if [ -f "$recovery_script" ]; then
        log_step "Installing recovery watchdog..."
        "$recovery_script" install 2>/dev/null || log_warn "Watchdog install failed (may need sudo)"
        sudo systemctl start trustgraph-watchdog 2>/dev/null || true
    fi
    
    echo ""
    echo "Monitor deployment progress:"
    echo "  ./deploy-swarm.sh status"
    echo "  ./deploy-swarm.sh services"
    echo ""
    echo "Access points (after services are running):"
    echo "  • Workbench UI: http://<node-ip>:8888"
    echo "  • API Gateway:  http://<node-ip>:8088"
    echo "  • Grafana:      http://<node-ip>:3000"
    echo "  • Prometheus:   http://<node-ip>:9090"
    echo "  • Watchdog:     http://<node-ip>:9999/metrics"
}

update_stack() {
    log_step "Updating TrustGraph stack..."
    
    # Load environment file if exists
    if [ -f "$ENV_FILE" ]; then
        log_info "Loading environment from $ENV_FILE"
        set -a
        source "$ENV_FILE"
        set +a
    fi
    
    docker stack deploy -c "$STACK_FILE" "$STACK_NAME" --with-registry-auth
    
    log_ok "Stack updated! Rolling update in progress..."
    echo ""
    echo "Monitor update progress:"
    echo "  watch docker service ls"
}

remove_stack() {
    log_step "Removing TrustGraph stack..."
    
    read -rp "Are you sure you want to remove the stack? [y/N]: " confirm
    if [[ "${confirm,,}" != "y" ]]; then
        echo "Cancelled"
        exit 0
    fi
    
    docker stack rm "$STACK_NAME"
    
    log_ok "Stack removal initiated"
    echo ""
    echo "Note: Volumes are preserved. To remove volumes:"
    echo "  docker volume prune"
}

cleanup_all() {
    log_step "Full cleanup (stack, secrets, networks)..."
    
    read -rp "This will remove EVERYTHING including data volumes. Continue? [y/N]: " confirm
    if [[ "${confirm,,}" != "y" ]]; then
        echo "Cancelled"
        exit 0
    fi
    
    # Remove stack
    docker stack rm "$STACK_NAME" 2>/dev/null || true
    
    # Wait for services to stop
    log_info "Waiting for services to stop..."
    sleep 15
    
    # Remove secrets
    remove_secrets
    
    # Remove networks
    docker network rm "${STACK_NAME}_trustgraph-internal" 2>/dev/null || true
    docker network rm "${STACK_NAME}_trustgraph-public" 2>/dev/null || true
    
    # Remove volumes
    read -rp "Also remove data volumes? [y/N]: " confirm_volumes
    if [[ "${confirm_volumes,,}" == "y" ]]; then
        docker volume rm "${STACK_NAME}_bookie" 2>/dev/null || true
        docker volume rm "${STACK_NAME}_cassandra" 2>/dev/null || true
        docker volume rm "${STACK_NAME}_grafana-storage" 2>/dev/null || true
        docker volume rm "${STACK_NAME}_prometheus-data" 2>/dev/null || true
        docker volume rm "${STACK_NAME}_qdrant" 2>/dev/null || true
        docker volume rm "${STACK_NAME}_zookeeper" 2>/dev/null || true
        docker volume rm "${STACK_NAME}_minio-data" 2>/dev/null || true
        log_ok "Volumes removed"
    fi
    
    log_ok "Cleanup complete"
}

# =============================================================================
# MONITORING & MANAGEMENT
# =============================================================================
show_status() {
    echo ""
    echo "=== Stack Status: $STACK_NAME ==="
    echo ""
    
    if ! docker stack ls | grep -q "$STACK_NAME"; then
        log_warn "Stack '$STACK_NAME' is not deployed"
        return 1
    fi
    
    echo "Services:"
    docker stack services "$STACK_NAME"
    echo ""
    
    echo "Tasks with issues:"
    docker stack ps "$STACK_NAME" --filter "desired-state=running" --format "table {{.Name}}\t{{.CurrentState}}\t{{.Error}}" | grep -v "Running" | head -20 || echo "  None"
    echo ""
}

list_services() {
    echo ""
    docker stack services "$STACK_NAME" --format "table {{.Name}}\t{{.Mode}}\t{{.Replicas}}\t{{.Image}}"
    echo ""
}

show_logs() {
    local service="${1:-}"
    
    if [ -z "$service" ]; then
        echo "Usage: $0 logs <service-name>"
        echo ""
        echo "Available services:"
        docker stack services "$STACK_NAME" --format "  {{.Name}}"
        exit 1
    fi
    
    # Add stack prefix if not present
    if [[ ! "$service" == "${STACK_NAME}_"* ]]; then
        service="${STACK_NAME}_${service}"
    fi
    
    docker service logs -f --tail 100 "$service"
}

scale_service() {
    local service="${1:-}"
    local replicas="${2:-}"
    
    if [ -z "$service" ] || [ -z "$replicas" ]; then
        echo "Usage: $0 scale <service-name> <replicas>"
        echo ""
        echo "Example: $0 scale api-gateway 3"
        exit 1
    fi
    
    # Add stack prefix if not present
    if [[ ! "$service" == "${STACK_NAME}_"* ]]; then
        service="${STACK_NAME}_${service}"
    fi
    
    docker service scale "$service=$replicas"
    log_ok "Scaled $service to $replicas replicas"
}

rollback_service() {
    local service="${1:-}"
    
    if [ -z "$service" ]; then
        echo "Usage: $0 rollback <service-name>"
        exit 1
    fi
    
    # Add stack prefix if not present
    if [[ ! "$service" == "${STACK_NAME}_"* ]]; then
        service="${STACK_NAME}_${service}"
    fi
    
    docker service rollback "$service"
    log_ok "Rollback initiated for $service"
}

# =============================================================================
# HEALTH CHECKS
# =============================================================================
check_health() {
    echo ""
    echo "=== Health Check ==="
    echo ""
    
    local services
    services=$(docker stack services "$STACK_NAME" --format "{{.Name}}")
    
    local healthy=0
    local unhealthy=0
    local pending=0
    
    for svc in $services; do
        local replicas
        replicas=$(docker service ls --filter "name=$svc" --format "{{.Replicas}}")
        local current desired
        current=$(echo "$replicas" | cut -d'/' -f1)
        desired=$(echo "$replicas" | cut -d'/' -f2)
        
        if [ "$current" == "$desired" ] && [ "$current" != "0" ]; then
            echo -e "${GREEN}✓${NC} $svc: $replicas"
            ((healthy++))
        elif [ "$current" == "0" ]; then
            echo -e "${YELLOW}○${NC} $svc: $replicas (starting)"
            ((pending++))
        else
            echo -e "${RED}✗${NC} $svc: $replicas"
            ((unhealthy++))
        fi
    done
    
    echo ""
    echo "Summary: $healthy healthy, $pending pending, $unhealthy unhealthy"
}

# =============================================================================
# VISUALIZE NETWORK
# =============================================================================
show_network() {
    echo ""
    echo "=== Network Topology ==="
    echo ""
    docker network ls --filter "name=${STACK_NAME}"
    echo ""
    echo "Internal network services:"
    docker network inspect "${STACK_NAME}_trustgraph-internal" --format '{{range .Containers}}{{.Name}} {{end}}' 2>/dev/null || echo "  (not yet created)"
    echo ""
}

# =============================================================================
# CREATE ENVIRONMENT FILE TEMPLATE
# =============================================================================
create_env_template() {
    log_step "Creating environment template..."
    
    cat > "$ENV_FILE.template" << 'ENVEOF'
# =============================================================================
# TrustGraph Production Environment Configuration
# =============================================================================
# Copy this to .env.production and fill in the values

# TrustGraph version
TRUSTGRAPH_VERSION=1.4.23

# OpenAI API configuration (or compatible API endpoint)
OPENAI_BASE_URL=https://api.openai.com/v1

# vLLM configuration (if using local vLLM)
VLLM_MODEL=Qwen/Qwen3-VL-4B-Instruct
VLLM_MAX_MODEL_LEN=16384
VLLM_DTYPE=auto
VLLM_GPU_MEMORY_UTILIZATION=0.85
VLLM_PORT=8001

# MinIO configuration
MINIO_OCR_BUCKET=ocr-images

# Container registry (for custom images)
REGISTRY=docker.io
ENVEOF

    log_ok "Created $ENV_FILE.template"
    echo ""
    echo "Copy and customize:"
    echo "  cp $ENV_FILE.template $ENV_FILE"
    echo "  nano $ENV_FILE"
}

# =============================================================================
# HELP
# =============================================================================
show_help() {
    cat << 'HELPEOF'
TrustGraph Docker Swarm Deployment

QUICK START:
  1. Initialize swarm:    ./deploy-swarm.sh init
  2. Label nodes:         ./deploy-swarm.sh labels
  3. Create secrets:      ./deploy-swarm.sh secrets
  4. Deploy:              ./deploy-swarm.sh deploy

COMMANDS:
  Swarm Management:
    init              Initialize Docker Swarm on this node
    join-worker       Get token to join as worker
    join-manager      Get token to join as manager
    labels            Interactively label a node
    nodes             Show all nodes and their labels

  Secrets:
    secrets           Create required Docker secrets (interactive)

  Stack Operations:
    deploy            Deploy the TrustGraph stack
    update            Update running stack (rolling update)
    remove            Remove the stack (preserves volumes)
    cleanup           Full cleanup including secrets and volumes

  Monitoring:
    status            Show stack status overview
    services          List all services
    logs <service>    Show logs for a service
    health            Health check all services
    network           Show network topology

  Scaling:
    scale <svc> <n>   Scale a service to n replicas
    rollback <svc>    Rollback a service to previous version

  Utilities:
    env               Create environment template file
    watchdog <cmd>    Control recovery watchdog (status|start|stop|test|logs)

EXAMPLES:
  # Single-node deployment (development/testing)
  ./deploy-swarm.sh init
  ./deploy-swarm.sh labels      # Select option 4 for all labels
  ./deploy-swarm.sh secrets
  ./deploy-swarm.sh deploy

  # Scale API gateway
  ./deploy-swarm.sh scale api-gateway 3

  # View API gateway logs
  ./deploy-swarm.sh logs api-gateway

  # Rollback a failed update
  ./deploy-swarm.sh rollback embeddings

HIGH AVAILABILITY NOTES:
  • For HA, deploy at least 3 manager nodes
  • Stateful services (Cassandra, Qdrant, Pulsar) are pinned to labeled nodes
  • Stateless services can scale horizontally
  • GPU services (vLLM) require nodes with trustgraph.gpu=true label

HELPEOF
}

# =============================================================================
# MAIN
# =============================================================================
case "${1:-help}" in
    init)
        init_swarm
        ;;
    join-worker)
        get_join_token worker
        ;;
    join-manager)
        get_join_token manager
        ;;
    labels)
        label_nodes
        ;;
    labels-all)
        label_current_node_all
        ;;
    nodes)
        show_nodes
        ;;
    secrets)
        create_secrets
        ;;
    secrets-remove)
        remove_secrets
        ;;
    deploy)
        deploy_stack
        ;;
    update)
        update_stack
        ;;
    remove)
        remove_stack
        ;;
    cleanup)
        cleanup_all
        ;;
    status)
        show_status
        ;;
    services)
        list_services
        ;;
    logs)
        show_logs "${2:-}"
        ;;
    scale)
        scale_service "${2:-}" "${3:-}"
        ;;
    rollback)
        rollback_service "${2:-}"
        ;;
    health)
        check_health
        ;;
    network)
        show_network
        ;;
    env)
        create_env_template
        ;;
    watchdog)
        if [ -f "${SCRIPT_DIR}/recovery/recovery-master.sh" ]; then
            "${SCRIPT_DIR}/recovery/recovery-master.sh" "${2:-status}"
        else
            log_err "Recovery script not found: ${SCRIPT_DIR}/recovery/recovery-master.sh"
            exit 1
        fi
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        log_err "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
