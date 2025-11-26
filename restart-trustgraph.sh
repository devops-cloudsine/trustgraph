#!/bin/bash
set -e

cd /home/ubuntu

echo "=== Stopping containers ==="
docker compose -f docker-compose.yaml down -v -t 0
sleep 10

echo "=== Starting containers ==="
docker compose -f /home/ubuntu/docker-compose.yaml -f /home/ubuntu/docker-compose.override.yaml up -d

echo "=== Waiting for services (60s) ==="
sleep 60

# echo "=== Pushing flow classes ==="
# python3 -c "
# import json, requests, time
# for i in range(30):
#     try:
#         if requests.post('http://localhost:8088/api/v1/flow', json={'operation': 'list-flows'}, timeout=5).status_code == 200:
#             break
#     except: pass
#     time.sleep(2)

# with open('/home/ubuntu/trustgraph/config.json') as f:
#     config = json.load(f)
# for name, defn in config['flow-classes'].items():
#     r = requests.post('http://localhost:8088/api/v1/flow', json={'operation': 'put-class', 'class-name': name, 'class-definition': json.dumps(defn)})
#     print(f'{name}: {r.status_code}')
# "

# echo "=== Restarting flow ==="
# curl -s -X POST http://localhost:8088/api/v1/flow -H 'Content-Type: application/json' -d '{"operation": "stop-flow", "flow-id": "docproc-1"}' || true
# sleep 2
# curl -s -X POST http://localhost:8088/api/v1/flow -H 'Content-Type: application/json' -d '{"operation": "start-flow", "flow-id": "docproc-1", "class-name": "everything", "description": "Document processing pipeline", "parameters": {"chunk-overlap": "50", "chunk-size": "2000", "llm-model": "Qwen/Qwen3-VL-4B-Instruct", "llm-rag-model": "Qwen/Qwen3-VL-4B-Instruct", "llm-rag-temperature": "0.3", "llm-temperature": "0.3"}}'

# echo ""
echo "=== Complete! ==="