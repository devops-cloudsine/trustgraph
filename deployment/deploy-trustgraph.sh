#!/bin/bash
# =============================================================================
# TrustGraph One-Click Deploy Script
# =============================================================================
# Usage:
#   ./deploy-trustgraph.sh              # Start all services
#   ./deploy-trustgraph.sh stop         # Stop all services
#   ./deploy-trustgraph.sh restart      # Restart all services
#   ./deploy-trustgraph.sh status       # Show service status
#   ./deploy-trustgraph.sh logs [svc]   # Show logs
#   ./deploy-trustgraph.sh recover      # Restart unhealthy services
#   ./deploy-trustgraph.sh setup        # One-time Docker daemon setup (requires sudo)
# =============================================================================

set -euo pipefail

# Configuration
COMPOSE_DIR="/home/cloudsineai/Desktop/trustgraph/deployment"
COMPOSE_FILE="${COMPOSE_DIR}/docker-compose.yaml"
OVERRIDE_FILE="${COMPOSE_DIR}/docker-compose.override.yaml"

# Use override file if it exists (for development)
if [ -f "$OVERRIDE_FILE" ]; then
    COMPOSE_CMD="docker compose -f $COMPOSE_FILE -f $OVERRIDE_FILE"
else
    COMPOSE_CMD="docker compose -f $COMPOSE_FILE"
fi

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_ok() { echo -e "${GREEN}✅ $1${NC}"; }
log_warn() { echo -e "${YELLOW}⚠️  $1${NC}"; }
log_err() { echo -e "${RED}❌ $1${NC}"; }
log_info() { echo -e "ℹ️  $1"; }

# =============================================================================
# ONE-TIME SETUP: Configure Docker for resilience (run with sudo)
# =============================================================================
setup_docker() {
    if [ "$EUID" -ne 0 ]; then
        echo "Setup requires sudo. Run: sudo $0 setup"
        exit 1
    fi

    echo "=== Configuring Docker for Production ==="
    
    # Backup existing config
    [ -f /etc/docker/daemon.json ] && cp /etc/docker/daemon.json /etc/docker/daemon.json.bak
    
    # Create optimized config
    cat > /etc/docker/daemon.json << 'EOF'
{
    "live-restore": true,
    "log-driver": "json-file",
    "log-opts": { "max-size": "50m", "max-file": "5" },
    "default-ulimits": {
        "nofile": { "Name": "nofile", "Hard": 65536, "Soft": 65536 }
    }
}
EOF

    systemctl reload docker || systemctl restart docker
    
    log_ok "Docker configured with live-restore (containers survive daemon restarts)"
    echo ""
    echo "Key features enabled:"
    echo "  • live-restore: Containers keep running during Docker restarts"
    echo "  • Log rotation: 50MB max per file, 5 files max"
    echo "  • Increased file limits: 65536"
}

# =============================================================================
# HEALTH CHECK: Check if critical services are healthy
# =============================================================================
wait_for_service() {
    local service=$1
    local timeout=${2:-120}
    local elapsed=0
    
    log_info "Waiting for $service..."
    while [ $elapsed -lt $timeout ]; do
        if $COMPOSE_CMD ps "$service" 2>/dev/null | grep -q "healthy\|running"; then
            log_ok "$service is ready"
            return 0
        fi
        sleep 5
        ((elapsed+=5))
    done
    log_warn "$service may not be ready yet (timeout)"
    return 1
}

check_health() {
    echo ""
    echo "=== Service Health Check ==="
    
    local unhealthy=""
    local services=$($COMPOSE_CMD ps --services 2>/dev/null)
    
    for svc in $services; do
        local status=$($COMPOSE_CMD ps "$svc" --format "{{.Status}}" 2>/dev/null | head -1)
        
        if echo "$status" | grep -qi "unhealthy\|exit"; then
            unhealthy="$unhealthy $svc"
            echo -e "  ${RED}✗${NC} $svc: $status"
        elif echo "$status" | grep -qi "up\|running\|healthy"; then
            echo -e "  ${GREEN}✓${NC} $svc: $status"
        else
            echo -e "  ${YELLOW}?${NC} $svc: $status"
        fi
    done
    
    echo ""
    if [ -n "$unhealthy" ]; then
        log_warn "Unhealthy services:$unhealthy"
        echo "Run '$0 recover' to restart them"
        return 1
    else
        log_ok "All services healthy!"
    fi
}

# =============================================================================
# START: Start services in proper order
# =============================================================================
start_services() {
    echo "=== Starting TrustGraph ==="
    echo ""
    
    # Phase 1: Infrastructure (must start first)
    log_info "Phase 1: Starting infrastructure..."
    $COMPOSE_CMD up -d zookeeper
    sleep 10
    $COMPOSE_CMD up -d bookie pulsar cassandra qdrant
    
    # Wait for critical infrastructure
    wait_for_service pulsar 180
    wait_for_service cassandra 180
    wait_for_service qdrant 60
    
    # Phase 2: Init jobs
    log_info "Phase 2: Running initialization..."
    $COMPOSE_CMD up -d pulsar-init
    sleep 10
    $COMPOSE_CMD up -d init-trustgraph
    sleep 5
    
    # Phase 3: Start everything else
    log_info "Phase 3: Starting all services..."
    $COMPOSE_CMD up -d
    
    echo ""
    log_ok "TrustGraph started!"
    echo ""
    echo "Access points:"
    echo "  • Workbench UI: http://localhost:8888"
    echo "  • API Gateway:  http://localhost:8088"
    echo "  • Grafana:      http://localhost:3000"
    echo ""
    echo "Run '$0 status' to check service health"
}

# =============================================================================
# STOP: Graceful shutdown
# =============================================================================
stop_services() {
    echo "=== Stopping TrustGraph ==="
    $COMPOSE_CMD down --timeout 60
    log_ok "All services stopped"
}

# =============================================================================
# RESTART: Full restart
# =============================================================================
restart_services() {
    stop_services
    sleep 5
    start_services
}

# =============================================================================
# RECOVER: Restart unhealthy/stopped services
# =============================================================================
recover_services() {
    local silent=${1:-""}
    [ -z "$silent" ] && echo "=== Recovering Unhealthy Services ==="
    
    local services=$($COMPOSE_CMD ps --services 2>/dev/null)
    local recovered=0
    
    for svc in $services; do
        local status=$($COMPOSE_CMD ps "$svc" --format "{{.Status}}" 2>/dev/null | head -1)
        
        if echo "$status" | grep -qi "unhealthy\|exit\|dead\|restarting"; then
            log_warn "Restarting: $svc"
            $COMPOSE_CMD restart "$svc" --timeout 30 2>/dev/null || true
            ((recovered++)) || true
        fi
    done
    
    if [ $recovered -eq 0 ]; then
        [ -z "$silent" ] && log_ok "No unhealthy services found"
    else
        log_ok "Recovered $recovered services"
    fi
    
    return $recovered
}

# =============================================================================
# WATCH: Auto-recovery watchdog (runs in background)
# =============================================================================
watch_services() {
    local interval=${1:-60}
    
    echo "=== TrustGraph Auto-Recovery Watchdog ==="
    echo "Checking every ${interval}s. Press Ctrl+C to stop."
    echo ""
    
    while true; do
        local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        
        # Check for unhealthy services and recover
        local services=$($COMPOSE_CMD ps --services 2>/dev/null)
        local issues=0
        
        for svc in $services; do
            local status=$($COMPOSE_CMD ps "$svc" --format "{{.Status}}" 2>/dev/null | head -1)
            
            if echo "$status" | grep -qi "unhealthy\|exit\|dead"; then
                ((issues++)) || true
                log_warn "[$timestamp] $svc is down - restarting..."
                $COMPOSE_CMD restart "$svc" --timeout 30 2>/dev/null &
            fi
        done
        
        if [ $issues -eq 0 ]; then
            echo -ne "\r[$timestamp] All services healthy ✓   "
        fi
        
        sleep $interval
    done
}

# =============================================================================
# STATUS: Show all service status
# =============================================================================
show_status() {
    echo ""
    echo "=== TrustGraph Service Status ==="
    echo ""
    $COMPOSE_CMD ps -a
    echo ""
    check_health
}

# =============================================================================
# LOGS: Show logs
# =============================================================================
show_logs() {
    local service=${1:-""}
    if [ -n "$service" ]; then
        $COMPOSE_CMD logs -f --tail 100 "$service"
    else
        $COMPOSE_CMD logs -f --tail 50
    fi
}

# =============================================================================
# BUILD: Build dev images
# =============================================================================
build_images() {
    echo "=== Building Development Images ==="
    
    # Build flow dev image
    log_info "Building trustgraph-flow:dev..."
    docker build --no-cache \
        -f /home/cloudsineai/Desktop/trustgraph/Containerfile.dev \
        -t docker.io/trustgraph/trustgraph-flow:dev \
        /home/cloudsineai/Desktop/trustgraph
    
    # Build OCR dev image
    log_info "Building trustgraph-ocr:dev..."
    docker build --no-cache \
        -f /home/cloudsineai/Desktop/trustgraph/Containerfile.ocr.dev \
        -t trustgraph-ocr-dev:latest \
        /home/cloudsineai/Desktop/trustgraph
    
    log_ok "Images built!"
}

# =============================================================================
# MAIN
# =============================================================================
case "${1:-start}" in
    start)
        start_services
        ;;
    stop)
        stop_services
        ;;
    restart)
        restart_services
        ;;
    status)
        show_status
        ;;
    health)
        check_health
        ;;
    recover)
        recover_services
        ;;
    watch)
        watch_services "${2:-60}"
        ;;
    logs)
        show_logs "${2:-}"
        ;;
    build)
        build_images
        ;;
    setup)
        setup_docker
        ;;
    *)
        echo "TrustGraph Deploy Script"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  start     Start all services (default)"
        echo "  stop      Stop all services"
        echo "  restart   Restart all services"
        echo "  status    Show service status"
        echo "  recover   Restart unhealthy services (one-time)"
        echo "  watch     Auto-recovery watchdog (runs continuously)"
        echo "            Optional: watch <seconds> (default: 60)"
        echo "  logs      Show logs (optionally: logs <service>)"
        echo "  build     Build dev images"
        echo "  setup     One-time Docker daemon config (requires sudo)"
        echo ""
        echo "Examples:"
        echo "  $0 start              # Start everything"
        echo "  $0 watch              # Auto-recover every 60s"
        echo "  $0 watch 30           # Auto-recover every 30s"
        echo "  nohup $0 watch &      # Run watchdog in background"
        echo ""
        ;;
esac

