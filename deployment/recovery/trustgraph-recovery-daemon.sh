#!/bin/bash
# =============================================================================
# TrustGraph Recovery Daemon
# =============================================================================
# Advanced auto-recovery with:
#   - Dependency-aware restart ordering
#   - Exponential backoff for repeated failures
#   - Prometheus metrics endpoint
#   - Alerting support (webhook/email)
#   - Circuit breaker for external services
#   - Log rotation and management
# =============================================================================

set -uo pipefail

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(dirname "$SCRIPT_DIR")"
COMPOSE_DIR="${COMPOSE_DIR:-$DEPLOY_DIR}"
COMPOSE_FILE="${COMPOSE_DIR}/docker-compose.yaml"
OVERRIDE_FILE="${COMPOSE_DIR}/docker-compose.override.yaml"
HEALTHCHECK_FILE="${COMPOSE_DIR}/docker-compose.healthchecks.yaml"

# Timing
CHECK_INTERVAL="${CHECK_INTERVAL:-30}"           # Seconds between checks
STARTUP_DELAY="${STARTUP_DELAY:-60}"             # Wait after boot before monitoring
MAX_RESTART_ATTEMPTS="${MAX_RESTART_ATTEMPTS:-5}"
BACKOFF_BASE="${BACKOFF_BASE:-30}"               # Base seconds for exponential backoff

# Alerting (optional)
ALERT_WEBHOOK="${ALERT_WEBHOOK:-}"               # Slack/Discord webhook URL
ALERT_EMAIL="${ALERT_EMAIL:-}"                   # Email for alerts

# Logging
LOG_DIR="${SCRIPT_DIR}/logs"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/watchdog.log}"
LOG_MAX_SIZE="${LOG_MAX_SIZE:-10485760}"         # 10MB
METRICS_PORT="${METRICS_PORT:-9999}"             # Prometheus metrics port

# Ensure log directory exists
mkdir -p "$LOG_DIR" 2>/dev/null || true

# Build compose command
COMPOSE_CMD="docker compose -f $COMPOSE_FILE"
[ -f "$OVERRIDE_FILE" ] && COMPOSE_CMD="$COMPOSE_CMD -f $OVERRIDE_FILE"
[ -f "$HEALTHCHECK_FILE" ] && COMPOSE_CMD="$COMPOSE_CMD -f $HEALTHCHECK_FILE"

# Detect deployment mode
detect_mode() {
    local count=$(docker service ls --filter "name=trustgraph_" --quiet 2>/dev/null | wc -l)
    if [ "$count" -gt 0 ]; then
        echo "swarm"
    else
        echo "compose"
    fi
}
DEPLOY_MODE=""  # Will be set at startup

# -----------------------------------------------------------------------------
# Service Dependency Graph (order matters for recovery)
# -----------------------------------------------------------------------------
declare -A SERVICE_DEPS
SERVICE_DEPS=(
    ["zookeeper"]=""
    ["bookie"]="zookeeper"
    ["pulsar"]="zookeeper bookie"
    ["pulsar-init"]="pulsar"
    ["cassandra"]=""
    ["qdrant"]=""
    ["minio"]=""
    ["init-trustgraph"]="pulsar"
    ["embeddings"]="pulsar"
    ["api-gateway"]="pulsar"
    ["vllm"]=""
    ["prometheus"]=""
    ["grafana"]="prometheus"
)

# Critical services that require immediate attention
CRITICAL_SERVICES="pulsar cassandra qdrant api-gateway vllm"

# Tracking failure counts for exponential backoff
declare -A FAILURE_COUNTS
declare -A LAST_RESTART_TIME

# Metrics counters
TOTAL_RESTARTS=0
SUCCESSFUL_RESTARTS=0
FAILED_RESTARTS=0
CURRENT_UNHEALTHY=0

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
log() {
    local level="$1"
    shift
    local msg="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local log_entry="[$timestamp] [$level] $msg"
    
    echo "$log_entry"
    echo "$log_entry" >> "$LOG_FILE"
    
    # Rotate log if too large
    if [ -f "$LOG_FILE" ] && [ $(stat -f%z "$LOG_FILE" 2>/dev/null || stat -c%s "$LOG_FILE" 2>/dev/null) -gt "$LOG_MAX_SIZE" ]; then
        mv "$LOG_FILE" "${LOG_FILE}.1"
        touch "$LOG_FILE"
    fi
}

log_info() { log "INFO" "$@"; }
log_warn() { log "WARN" "$@"; }
log_error() { log "ERROR" "$@"; }

# -----------------------------------------------------------------------------
# Alerting
# -----------------------------------------------------------------------------
send_alert() {
    local subject="$1"
    local message="$2"
    local severity="${3:-warning}"
    
    log_warn "ALERT [$severity]: $subject - $message"
    
    # Webhook (Slack/Discord)
    if [ -n "$ALERT_WEBHOOK" ]; then
        local color="warning"
        [ "$severity" = "critical" ] && color="danger"
        [ "$severity" = "info" ] && color="good"
        
        curl -sf -X POST "$ALERT_WEBHOOK" \
            -H 'Content-Type: application/json' \
            -d "{\"text\":\"[$severity] TrustGraph: $subject\",\"attachments\":[{\"color\":\"$color\",\"text\":\"$message\"}]}" \
            2>/dev/null &
    fi
    
    # Email (if mail is configured)
    if [ -n "$ALERT_EMAIL" ] && command -v mail &>/dev/null; then
        echo "$message" | mail -s "[TrustGraph $severity] $subject" "$ALERT_EMAIL" &
    fi
}

# -----------------------------------------------------------------------------
# Service Status Checking
# -----------------------------------------------------------------------------
get_service_status() {
    local service="$1"
    $COMPOSE_CMD ps "$service" --format "{{.Status}}" 2>/dev/null | head -1
}

is_service_healthy() {
    local service="$1"
    local status=$(get_service_status "$service")
    
    if echo "$status" | grep -qi "healthy"; then
        return 0
    elif echo "$status" | grep -qi "Up\|running" && ! echo "$status" | grep -qi "unhealthy"; then
        return 0
    fi
    return 1
}

is_service_down() {
    local service="$1"
    local status=$(get_service_status "$service")
    
    echo "$status" | grep -qiE "exit|dead|unhealthy|restarting" && return 0
    [ -z "$status" ] && return 0
    return 1
}

# -----------------------------------------------------------------------------
# Dependency-Aware Recovery
# -----------------------------------------------------------------------------
get_dependencies() {
    local service="$1"
    echo "${SERVICE_DEPS[$service]:-}"
}

check_dependencies_healthy() {
    local service="$1"
    local deps=$(get_dependencies "$service")
    
    for dep in $deps; do
        if ! is_service_healthy "$dep"; then
            return 1
        fi
    done
    return 0
}

calculate_backoff() {
    local service="$1"
    local count="${FAILURE_COUNTS[$service]:-0}"
    
    # Exponential backoff with max of 10 minutes
    local backoff=$((BACKOFF_BASE * (2 ** count)))
    [ $backoff -gt 600 ] && backoff=600
    
    echo $backoff
}

should_attempt_restart() {
    local service="$1"
    local count="${FAILURE_COUNTS[$service]:-0}"
    local last_time="${LAST_RESTART_TIME[$service]:-0}"
    local now=$(date +%s)
    
    # Check max attempts
    if [ $count -ge $MAX_RESTART_ATTEMPTS ]; then
        local backoff=$(calculate_backoff "$service")
        if [ $((now - last_time)) -lt $backoff ]; then
            return 1
        fi
        # Reset after backoff period
        FAILURE_COUNTS[$service]=0
    fi
    
    return 0
}

restart_service() {
    local service="$1"
    local reason="$2"
    
    log_info "Attempting restart: $service (reason: $reason)"
    
    # Check dependencies first
    if ! check_dependencies_healthy "$service"; then
        log_warn "Skipping $service - dependencies not healthy"
        return 1
    fi
    
    # Check backoff
    if ! should_attempt_restart "$service"; then
        local backoff=$(calculate_backoff "$service")
        log_warn "Skipping $service - in backoff period (${backoff}s)"
        return 1
    fi
    
    # Track attempt
    FAILURE_COUNTS[$service]=$((${FAILURE_COUNTS[$service]:-0} + 1))
    LAST_RESTART_TIME[$service]=$(date +%s)
    ((TOTAL_RESTARTS++))
    
    # Perform restart
    if $COMPOSE_CMD restart "$service" --timeout 60 2>/dev/null; then
        sleep 10  # Wait for service to start
        
        if is_service_healthy "$service"; then
            log_info "Successfully restarted: $service"
            FAILURE_COUNTS[$service]=0
            ((SUCCESSFUL_RESTARTS++))
            return 0
        fi
    fi
    
    log_error "Failed to restart: $service (attempt ${FAILURE_COUNTS[$service]})"
    ((FAILED_RESTARTS++))
    
    # Alert on critical failures
    if [[ " $CRITICAL_SERVICES " == *" $service "* ]]; then
        local count="${FAILURE_COUNTS[$service]:-0}"
        if [ $count -ge 3 ]; then
            send_alert "Critical service failure" \
                "$service failed to restart after $count attempts" \
                "critical"
        fi
    fi
    
    return 1
}

# -----------------------------------------------------------------------------
# Recovery Ordering (topological sort based on dependencies)
# -----------------------------------------------------------------------------
get_restart_order() {
    local unhealthy_services="$1"
    local ordered=""
    
    # Infrastructure first
    for svc in zookeeper bookie pulsar pulsar-init cassandra qdrant minio; do
        if [[ " $unhealthy_services " == *" $svc "* ]]; then
            ordered="$ordered $svc"
        fi
    done
    
    # Init services
    for svc in init-trustgraph; do
        if [[ " $unhealthy_services " == *" $svc "* ]]; then
            ordered="$ordered $svc"
        fi
    done
    
    # All other services
    for svc in $unhealthy_services; do
        if [[ " $ordered " != *" $svc "* ]]; then
            ordered="$ordered $svc"
        fi
    done
    
    echo "$ordered"
}

# -----------------------------------------------------------------------------
# Prometheus Metrics Endpoint
# -----------------------------------------------------------------------------
start_metrics_server() {
    # Simple metrics endpoint using netcat
    while true; do
        local metrics=$(cat << METRICS
# HELP trustgraph_watchdog_total_restarts Total number of restart attempts
# TYPE trustgraph_watchdog_total_restarts counter
trustgraph_watchdog_total_restarts $TOTAL_RESTARTS

# HELP trustgraph_watchdog_successful_restarts Number of successful restarts
# TYPE trustgraph_watchdog_successful_restarts counter
trustgraph_watchdog_successful_restarts $SUCCESSFUL_RESTARTS

# HELP trustgraph_watchdog_failed_restarts Number of failed restarts
# TYPE trustgraph_watchdog_failed_restarts counter
trustgraph_watchdog_failed_restarts $FAILED_RESTARTS

# HELP trustgraph_watchdog_unhealthy_services Current number of unhealthy services
# TYPE trustgraph_watchdog_unhealthy_services gauge
trustgraph_watchdog_unhealthy_services $CURRENT_UNHEALTHY
METRICS
)
        echo -e "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: ${#metrics}\r\n\r\n$metrics" | \
            nc -l -p "$METRICS_PORT" -q 1 2>/dev/null || sleep 5
    done
}

# -----------------------------------------------------------------------------
# Main Recovery Loop
# -----------------------------------------------------------------------------
run_recovery_cycle_swarm() {
    local unhealthy=""
    
    # Check swarm services
    while read -r line; do
        local name=$(echo "$line" | awk '{print $1}')
        local replicas=$(echo "$line" | awk '{print $2}')
        local current=$(echo "$replicas" | cut -d'/' -f1)
        local desired=$(echo "$replicas" | cut -d'/' -f2)
        
        if [ "$current" != "$desired" ] || [ "$current" -eq 0 ]; then
            unhealthy="$unhealthy $name"
        fi
    done < <(docker service ls --filter "name=trustgraph_" --format "{{.Name}} {{.Replicas}}" 2>/dev/null)
    
    CURRENT_UNHEALTHY=$(echo "$unhealthy" | wc -w)
    
    if [ -n "$unhealthy" ]; then
        log_warn "Unhealthy swarm services detected:$unhealthy"
        
        for svc in $unhealthy; do
            log_info "Force-updating swarm service: $svc"
            docker service update --force "$svc" 2>/dev/null &
            ((TOTAL_RESTARTS++))
        done
    fi
}

run_recovery_cycle_compose() {
    local unhealthy=""
    local services=$($COMPOSE_CMD ps --services 2>/dev/null)
    
    for svc in $services; do
        if is_service_down "$svc"; then
            unhealthy="$unhealthy $svc"
        fi
    done
    
    CURRENT_UNHEALTHY=$(echo "$unhealthy" | wc -w)
    
    if [ -n "$unhealthy" ]; then
        log_warn "Unhealthy services detected:$unhealthy"
        
        # Get proper restart order
        local ordered=$(get_restart_order "$unhealthy")
        
        for svc in $ordered; do
            restart_service "$svc" "unhealthy"
            sleep 5  # Brief pause between restarts
        done
    fi
}

run_recovery_cycle() {
    if [ "$DEPLOY_MODE" = "swarm" ]; then
        run_recovery_cycle_swarm
    else
        run_recovery_cycle_compose
    fi
}

# -----------------------------------------------------------------------------
# Startup Validation
# -----------------------------------------------------------------------------
validate_startup() {
    log_info "Validating TrustGraph installation..."
    
    if ! command -v docker &>/dev/null; then
        log_error "Docker not found!"
        exit 1
    fi
    
    if [ ! -f "$COMPOSE_FILE" ]; then
        log_error "Docker Compose file not found: $COMPOSE_FILE"
        exit 1
    fi
    
    if ! docker info &>/dev/null; then
        log_error "Docker daemon not accessible"
        exit 1
    fi
    
    log_info "Validation passed"
}

# -----------------------------------------------------------------------------
# Graceful Shutdown
# -----------------------------------------------------------------------------
cleanup() {
    log_info "Shutting down TrustGraph Recovery Daemon..."
    
    # Kill metrics server if running
    pkill -f "nc -l -p $METRICS_PORT" 2>/dev/null || true
    
    exit 0
}

trap cleanup SIGTERM SIGINT

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
main() {
    log_info "==========================================="
    log_info "TrustGraph Recovery Daemon Starting"
    log_info "==========================================="
    
    # Detect deployment mode
    DEPLOY_MODE=$(detect_mode)
    log_info "Deployment Mode: $DEPLOY_MODE"
    log_info "Compose Dir: $COMPOSE_DIR"
    log_info "Check Interval: ${CHECK_INTERVAL}s"
    log_info "Max Restart Attempts: $MAX_RESTART_ATTEMPTS"
    log_info "Metrics Port: $METRICS_PORT"
    log_info "==========================================="
    
    validate_startup
    
    # Start metrics server in background
    start_metrics_server &
    
    # Initial delay to let services start
    log_info "Waiting ${STARTUP_DELAY}s for initial startup..."
    sleep "$STARTUP_DELAY"
    
    log_info "Starting recovery monitoring loop..."
    
    while true; do
        run_recovery_cycle
        sleep "$CHECK_INTERVAL"
    done
}

main "$@"
