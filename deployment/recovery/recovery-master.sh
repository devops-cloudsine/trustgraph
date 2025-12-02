#!/bin/bash
# =============================================================================
# TrustGraph Recovery Master Script
# =============================================================================
# Consolidated infrastructure recovery system with:
#   - Docker healthchecks
#   - Systemd watchdog daemon
#   - AlertManager notifications
#   - Auto-recovery with dependency ordering
#   - Prometheus metrics
#
# Usage:
#   ./recovery-master.sh install    # First-time setup
#   ./recovery-master.sh start      # Start recovery system
#   ./recovery-master.sh stop       # Stop recovery system
#   ./recovery-master.sh status     # Show status
#   ./recovery-master.sh test       # Test auto-recovery
#   ./recovery-master.sh logs       # View watchdog logs
#   ./recovery-master.sh uninstall  # Remove recovery system
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(dirname "$SCRIPT_DIR")"
RECOVERY_DIR="$SCRIPT_DIR"

# Files
COMPOSE_FILE="$DEPLOY_DIR/docker-compose.yaml"
OVERRIDE_FILE="$DEPLOY_DIR/docker-compose.override.yaml"
HEALTHCHECK_FILE="$DEPLOY_DIR/docker-compose.healthchecks.yaml"
SERVICE_FILE="$RECOVERY_DIR/trustgraph-watchdog.service"
DAEMON_FILE="$RECOVERY_DIR/trustgraph-recovery-daemon.sh"
ALERTMANAGER_DIR="$RECOVERY_DIR/alertmanager"

# Runtime
LOG_DIR="${RECOVERY_DIR}/logs"
LOG_FILE="${LOG_DIR}/watchdog.log"
METRICS_PORT=9999

# Ensure log directory exists
mkdir -p "$LOG_DIR" 2>/dev/null || true

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_ok() { echo -e "${GREEN}✓${NC} $1"; }
log_warn() { echo -e "${YELLOW}⚠${NC} $1"; }
log_err() { echo -e "${RED}✗${NC} $1"; }
log_info() { echo -e "${BLUE}ℹ${NC} $1"; }
log_header() { echo -e "\n${BLUE}═══════════════════════════════════════════════════════════════${NC}"; echo -e "${BLUE}  $1${NC}"; echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}\n"; }

# -----------------------------------------------------------------------------
# Detect deployment mode (Swarm vs Compose)
# -----------------------------------------------------------------------------
detect_mode() {
    local count=$(docker service ls --filter "name=trustgraph_" --quiet 2>/dev/null | wc -l)
    if [ "$count" -gt 0 ]; then
        echo "swarm"
    else
        echo "compose"
    fi
}

DEPLOY_MODE=$(detect_mode)

# Build compose command
build_compose_cmd() {
    local cmd="docker compose -f $COMPOSE_FILE"
    [ -f "$OVERRIDE_FILE" ] && cmd="$cmd -f $OVERRIDE_FILE"
    [ -f "$HEALTHCHECK_FILE" ] && cmd="$cmd -f $HEALTHCHECK_FILE"
    echo "$cmd"
}

COMPOSE_CMD=$(build_compose_cmd)

# -----------------------------------------------------------------------------
# Install: First-time setup
# -----------------------------------------------------------------------------
cmd_install() {
    log_header "TrustGraph Recovery System Installation"
    
    # Check prerequisites
    log_info "Checking prerequisites..."
    command -v docker &>/dev/null || { log_err "Docker not found"; exit 1; }
    command -v systemctl &>/dev/null || { log_err "Systemd not found"; exit 1; }
    
    # Verify compose files exist
    [ -f "$COMPOSE_FILE" ] || { log_err "docker-compose.yaml not found at $COMPOSE_FILE"; exit 1; }
    [ -f "$HEALTHCHECK_FILE" ] || { log_err "docker-compose.healthchecks.yaml not found"; exit 1; }
    
    log_ok "Prerequisites verified"
    
    # Validate compose configuration
    log_info "Validating Docker Compose configuration..."
    if $COMPOSE_CMD config --quiet 2>/dev/null; then
        log_ok "Compose configuration valid"
    else
        log_err "Compose configuration invalid"
        exit 1
    fi
    
    # Make daemon executable
    log_info "Setting up recovery daemon..."
    chmod +x "$DAEMON_FILE" 2>/dev/null || true
    log_ok "Recovery daemon ready"
    
    # Create alertmanager directory if missing
    mkdir -p "$ALERTMANAGER_DIR"
    
    # Install systemd service
    log_info "Installing systemd watchdog service..."
    if [ -f "$SERVICE_FILE" ]; then
        sudo cp "$SERVICE_FILE" /etc/systemd/system/trustgraph-watchdog.service
        sudo systemctl daemon-reload
        log_ok "Systemd service installed"
    else
        log_err "Service file not found: $SERVICE_FILE"
        exit 1
    fi
    
    # Configure Docker live-restore (optional)
    log_info "Checking Docker live-restore configuration..."
    if docker info 2>/dev/null | grep -q "Live Restore Enabled: true"; then
        log_ok "Docker live-restore already enabled"
    else
        log_warn "Docker live-restore not enabled. Run 'sudo $0 setup-docker' to enable"
    fi
    
    echo ""
    log_ok "Installation complete!"
    echo ""
    echo "Next steps:"
    echo "  1. Start recovery system:  $0 start"
    echo "  2. Check status:           $0 status"
    echo "  3. Test auto-recovery:     $0 test"
    echo ""
}

# -----------------------------------------------------------------------------
# Start: Enable and start recovery system
# -----------------------------------------------------------------------------
cmd_start() {
    log_header "Starting TrustGraph Recovery System"
    
    log_info "Detected mode: $DEPLOY_MODE"
    
    if [ "$DEPLOY_MODE" = "swarm" ]; then
        log_info "Swarm mode - services have built-in recovery"
        log_info "Starting watchdog for monitoring only..."
    else
        # Apply healthchecks to running containers (compose mode)
        log_info "Applying healthchecks overlay..."
        cd "$DEPLOY_DIR"
        $COMPOSE_CMD up -d
        log_ok "Healthchecks applied"
    fi
    
    # Start watchdog service
    log_info "Starting watchdog daemon..."
    sudo systemctl enable trustgraph-watchdog 2>/dev/null || true
    sudo systemctl start trustgraph-watchdog
    
    sleep 2
    if systemctl is-active --quiet trustgraph-watchdog; then
        log_ok "Watchdog daemon running"
    else
        log_err "Watchdog failed to start. Check: sudo journalctl -u trustgraph-watchdog"
        exit 1
    fi
    
    echo ""
    log_ok "Recovery system active!"
    echo ""
    echo "Mode: $DEPLOY_MODE"
    echo ""
    echo "Monitoring endpoints:"
    echo "  • Watchdog metrics:  http://localhost:$METRICS_PORT/metrics"
    echo "  • Grafana:           http://localhost:3000"
    echo "  • Prometheus:        http://localhost:9090/alerts"
    echo ""
}

# -----------------------------------------------------------------------------
# Stop: Disable recovery system
# -----------------------------------------------------------------------------
cmd_stop() {
    log_header "Stopping TrustGraph Recovery System"
    
    log_info "Stopping watchdog daemon..."
    sudo systemctl stop trustgraph-watchdog 2>/dev/null || true
    log_ok "Watchdog stopped"
    
    log_info "Note: Containers still running with healthchecks"
    echo ""
}

# -----------------------------------------------------------------------------
# Status: Show current status
# -----------------------------------------------------------------------------
cmd_status() {
    log_header "TrustGraph Recovery System Status"
    
    echo "Deployment Mode: $DEPLOY_MODE"
    echo ""
    
    # Watchdog status
    echo "Watchdog Daemon:"
    if systemctl is-active --quiet trustgraph-watchdog 2>/dev/null; then
        log_ok "  Running"
        echo "  PID: $(systemctl show trustgraph-watchdog --property=MainPID --value)"
    else
        log_warn "  Not running"
    fi
    echo ""
    
    # Metrics
    echo "Watchdog Metrics:"
    if curl -sf "http://localhost:$METRICS_PORT/metrics" &>/dev/null; then
        curl -sf "http://localhost:$METRICS_PORT/metrics" 2>/dev/null | grep trustgraph | head -8
    else
        log_warn "  Metrics endpoint not available"
    fi
    echo ""
    
    # Service health (handles both swarm and compose)
    echo "Service Health:"
    local unhealthy=0
    local healthy=0
    local total=0
    
    if [ "$DEPLOY_MODE" = "swarm" ]; then
        while read -r line; do
            ((total++))
            local replicas=$(echo "$line" | awk '{print $3}')
            local name=$(echo "$line" | awk '{print $2}')
            if echo "$replicas" | grep -qE "^0/|/0$"; then
                ((unhealthy++))
                echo -e "  ${RED}✗${NC} $name ($replicas)"
            elif echo "$replicas" | grep -qE "^[0-9]+/[0-9]+$"; then
                local current=$(echo "$replicas" | cut -d'/' -f1)
                local desired=$(echo "$replicas" | cut -d'/' -f2)
                if [ "$current" -eq "$desired" ]; then
                    ((healthy++))
                    echo -e "  ${GREEN}✓${NC} $name ($replicas)"
                else
                    echo -e "  ${YELLOW}?${NC} $name ($replicas)"
                fi
            fi
        done < <(docker service ls --filter "name=trustgraph_" --format "{{.ID}} {{.Name}} {{.Replicas}}" 2>/dev/null | head -30)
    else
        cd "$DEPLOY_DIR"
        while read -r line; do
            ((total++))
            if echo "$line" | grep -qi "healthy"; then
                ((healthy++))
                echo -e "  ${GREEN}✓${NC} $line"
            elif echo "$line" | grep -qiE "unhealthy|exit|dead"; then
                ((unhealthy++))
                echo -e "  ${RED}✗${NC} $line"
            else
                echo -e "  ${YELLOW}?${NC} $line"
            fi
        done < <($COMPOSE_CMD ps --format "{{.Name}}: {{.Status}}" 2>/dev/null | head -30)
    fi
    
    echo ""
    echo "Summary: $healthy healthy, $unhealthy unhealthy, $total total"
    echo ""
    
    # Grafana status
    echo "Grafana:"
    if curl -sf "http://localhost:3000/api/health" &>/dev/null; then
        log_ok "  Running at http://localhost:3000"
    else
        log_warn "  Not running or not accessible"
    fi
    echo ""
}

# -----------------------------------------------------------------------------
# Test: Test auto-recovery
# -----------------------------------------------------------------------------
cmd_test() {
    log_header "Testing TrustGraph Auto-Recovery"
    
    log_info "Mode: $DEPLOY_MODE"
    echo ""
    
    if [ "$DEPLOY_MODE" = "swarm" ]; then
        # Swarm mode test
        local test_service=$(docker service ls --filter "name=trustgraph_" --format "{{.Name}}" | grep -E "chunker|metering|prompt" | head -1)
        
        if [ -z "$test_service" ]; then
            log_warn "No suitable test service found"
            exit 1
        fi
        
        log_info "Test service: $test_service"
        
        # Find a task/container for this service
        local test_container=$(docker ps --filter "label=com.docker.swarm.service.name=$test_service" --format "{{.ID}}" | head -1)
        
        if [ -z "$test_container" ]; then
            log_warn "No running container found for $test_service"
            exit 1
        fi
        
        log_info "Test container: $test_container"
        echo ""
        
        # Kill the container - swarm will auto-restart
        log_warn "Killing container (Swarm will auto-restart)..."
        docker kill "$test_container" 2>/dev/null || true
        
        # Wait for swarm to restart
        log_info "Waiting for Swarm auto-recovery (up to 30s)..."
        local waited=0
        while [ $waited -lt 30 ]; do
            sleep 5
            ((waited+=5))
            local replicas=$(docker service ls --filter "name=$test_service" --format "{{.Replicas}}" 2>/dev/null)
            local current=$(echo "$replicas" | cut -d'/' -f1)
            local desired=$(echo "$replicas" | cut -d'/' -f2)
            if [ "$current" -eq "$desired" ] && [ "$current" -gt 0 ]; then
                break
            fi
            echo "  Still waiting... ${waited}s (replicas: $replicas)"
        done
        
        # Check replicas
        local replicas=$(docker service ls --filter "name=$test_service" --format "{{.Replicas}}" 2>/dev/null)
        if echo "$replicas" | grep -qE "^1/1|^[0-9]+/[0-9]+$"; then
            local current=$(echo "$replicas" | cut -d'/' -f1)
            local desired=$(echo "$replicas" | cut -d'/' -f2)
            if [ "$current" -eq "$desired" ] && [ "$current" -gt 0 ]; then
                log_ok "Service recovered! Replicas: $replicas"
                echo ""
                log_ok "Swarm auto-recovery test PASSED!"
                return 0
            fi
        fi
        
        log_err "Service may not have recovered. Replicas: $replicas"
        return 1
    else
        # Compose mode test
        # Check watchdog is running
        if ! systemctl is-active --quiet trustgraph-watchdog 2>/dev/null; then
            log_err "Watchdog not running. Start it first: $0 start"
            exit 1
        fi
        
        # Find a non-critical container to kill
        local test_container=$(docker ps --format "{{.Names}}" | grep -E "chunker|metering|prompt" | head -1)
        
        if [ -z "$test_container" ]; then
            log_warn "No suitable test container found"
            exit 1
        fi
        
        log_info "Test container: $test_container"
        echo ""
        
        # Get initial metrics
        local initial_restarts=$(curl -sf "http://localhost:$METRICS_PORT/metrics" 2>/dev/null | grep "trustgraph_watchdog_total_restarts" | awk '{print $2}' || echo "0")
        log_info "Initial restart count: $initial_restarts"
        
        # Kill the container
        log_warn "Killing container..."
        docker kill "$test_container" 2>/dev/null || true
        
        # Wait for recovery
        log_info "Waiting for auto-recovery (up to 60s)..."
        local waited=0
        while [ $waited -lt 60 ]; do
            sleep 5
            ((waited+=5))
            
            local status=$(docker ps --filter "name=$test_container" --format "{{.Status}}" 2>/dev/null)
            if echo "$status" | grep -qi "up"; then
                log_ok "Container recovered after ${waited}s!"
                echo ""
                log_ok "Auto-recovery test PASSED!"
                return 0
            fi
            
            echo "  Still waiting... ${waited}s"
        done
        
        log_err "Container did not recover within 60s"
        return 1
    fi
}

# -----------------------------------------------------------------------------
# Logs: View watchdog logs
# -----------------------------------------------------------------------------
cmd_logs() {
    log_header "TrustGraph Watchdog Logs"
    
    if [ -f "$LOG_FILE" ]; then
        echo "Log file: $LOG_FILE"
        echo "---"
        tail -50 "$LOG_FILE"
    else
        log_info "No log file found. Showing journalctl logs..."
    fi
    
    echo ""
    echo "Streaming live logs (Ctrl+C to stop)..."
    sudo journalctl -u trustgraph-watchdog -f
}

# -----------------------------------------------------------------------------
# Uninstall: Remove recovery system
# -----------------------------------------------------------------------------
cmd_uninstall() {
    log_header "Uninstalling TrustGraph Recovery System"
    
    log_info "Stopping and disabling watchdog..."
    sudo systemctl stop trustgraph-watchdog 2>/dev/null || true
    sudo systemctl disable trustgraph-watchdog 2>/dev/null || true
    
    log_info "Removing systemd service..."
    sudo rm -f /etc/systemd/system/trustgraph-watchdog.service
    sudo systemctl daemon-reload
    
    log_info "Removing log file..."
    rm -f "$LOG_FILE"
    
    log_ok "Recovery system uninstalled"
    echo ""
    echo "Note: Containers and healthchecks still active."
    echo "To restart without healthchecks:"
    echo "  docker compose -f $COMPOSE_FILE down"
    echo "  docker compose -f $COMPOSE_FILE up -d"
    echo ""
}

# -----------------------------------------------------------------------------
# Setup Docker: Configure Docker daemon for resilience
# -----------------------------------------------------------------------------
cmd_setup_docker() {
    log_header "Configuring Docker for Production Resilience"
    
    if [ "$EUID" -ne 0 ]; then
        log_err "This command requires sudo: sudo $0 setup-docker"
        exit 1
    fi
    
    # Backup existing config
    [ -f /etc/docker/daemon.json ] && cp /etc/docker/daemon.json /etc/docker/daemon.json.bak
    
    # Create optimized config
    cat > /etc/docker/daemon.json << 'EOF'
{
    "live-restore": true,
    "log-driver": "json-file",
    "log-opts": {
        "max-size": "50m",
        "max-file": "5"
    },
    "default-ulimits": {
        "nofile": {
            "Name": "nofile",
            "Hard": 65536,
            "Soft": 65536
        }
    }
}
EOF

    systemctl reload docker || systemctl restart docker
    
    log_ok "Docker configured with live-restore"
    echo ""
    echo "Features enabled:"
    echo "  • live-restore: Containers survive Docker daemon restarts"
    echo "  • Log rotation: 50MB max per file, 5 files max"
    echo "  • Increased file limits: 65536"
    echo ""
}

# -----------------------------------------------------------------------------
# Recover: Manual one-time recovery of unhealthy services
# -----------------------------------------------------------------------------
cmd_recover() {
    log_header "Recovering Unhealthy Services"
    
    cd "$DEPLOY_DIR"
    local recovered=0
    
    while read -r svc; do
        local status=$($COMPOSE_CMD ps "$svc" --format "{{.Status}}" 2>/dev/null | head -1)
        
        if echo "$status" | grep -qiE "unhealthy|exit|dead|restarting"; then
            log_warn "Restarting: $svc ($status)"
            $COMPOSE_CMD restart "$svc" --timeout 30 2>/dev/null || true
            ((recovered++))
        fi
    done < <($COMPOSE_CMD ps --services 2>/dev/null)
    
    if [ $recovered -eq 0 ]; then
        log_ok "No unhealthy services found"
    else
        log_ok "Recovered $recovered services"
    fi
    echo ""
}

# -----------------------------------------------------------------------------
# Help
# -----------------------------------------------------------------------------
cmd_help() {
    echo "TrustGraph Recovery System"
    echo ""
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  install       First-time setup (install systemd service)"
    echo "  start         Start recovery system (watchdog + healthchecks)"
    echo "  stop          Stop watchdog daemon"
    echo "  status        Show recovery system status"
    echo "  test          Test auto-recovery by killing a container"
    echo "  logs          View watchdog logs"
    echo "  recover       Manual one-time recovery of unhealthy services"
    echo "  uninstall     Remove recovery system"
    echo "  setup-docker  Configure Docker for resilience (requires sudo)"
    echo ""
    echo "Example workflow:"
    echo "  $0 install     # First time only"
    echo "  $0 start       # Enable auto-recovery"
    echo "  $0 status      # Check everything is working"
    echo "  $0 test        # Verify auto-recovery works"
    echo ""
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
case "${1:-help}" in
    install)     cmd_install ;;
    start)       cmd_start ;;
    stop)        cmd_stop ;;
    status)      cmd_status ;;
    test)        cmd_test ;;
    logs)        cmd_logs ;;
    recover)     cmd_recover ;;
    uninstall)   cmd_uninstall ;;
    setup-docker) cmd_setup_docker ;;
    help|--help|-h) cmd_help ;;
    *)
        log_err "Unknown command: $1"
        cmd_help
        exit 1
        ;;
esac

