#!/bin/bash
# Script to connect containers between Docker networks

# Networks to connect
NETWORK1="deployment_default"
NETWORK2="cospacegpt_default"

echo "Connecting containers between $NETWORK1 and $NETWORK2..."

# List of containers that need cross-network access
# Add containers from NETWORK1 that need access to NETWORK2
CONTAINERS_TO_CONNECT=(
    "deployment-text-completion-1"
    "deployment-api-gateway-1"
    "vllm-vision-server"
    "vllm-gpt-oss-server"
)

# Connect each container to both networks
for container in "${CONTAINERS_TO_CONNECT[@]}"; do
    # Check if container exists
    if docker ps -a --format '{{.Names}}' | grep -q "^${container}$"; then
        echo "Connecting $container..."
        
        # Connect to NETWORK1 if not already connected
        if ! docker inspect "$container" --format '{{range $net, $conf := .NetworkSettings.Networks}}{{$net}}{{end}}' | grep -q "$NETWORK1"; then
            docker network connect "$NETWORK1" "$container" 2>/dev/null && echo "  ✓ Connected to $NETWORK1" || echo "  ✗ Failed to connect to $NETWORK1"
        else
            echo "  - Already connected to $NETWORK1"
        fi
        
        # Connect to NETWORK2 if not already connected
        if ! docker inspect "$container" --format '{{range $net, $conf := .NetworkSettings.Networks}}{{$net}}{{end}}' | grep -q "$NETWORK2"; then
            docker network connect "$NETWORK2" "$container" 2>/dev/null && echo "  ✓ Connected to $NETWORK2" || echo "  ✗ Failed to connect to $NETWORK2"
        else
            echo "  - Already connected to $NETWORK2"
        fi
    else
        echo "  ✗ Container $container not found"
    fi
done

echo ""
echo "Done! Checking network connections..."
docker ps --format "table {{.Names}}\t{{.Networks}}" | grep -E "NAME|vllm|text-completion|api-gateway"









