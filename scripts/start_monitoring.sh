#!/bin/bash

# Start monitoring infrastructure for the security system
# This script starts Prometheus and Grafana for monitoring the security metrics

# Stop script on first error
set -e

# Ensure we're in the project root directory
cd "$(dirname "$0")/.."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
  echo "Error: Docker is not running or not accessible"
  exit 1
fi

echo "Starting monitoring infrastructure..."

# Check if monitoring volumes exist, create them if not
if [ ! -d "./monitoring/prometheus_data" ]; then
  echo "Creating Prometheus data directory..."
  mkdir -p ./monitoring/prometheus_data
  chmod 777 ./monitoring/prometheus_data
fi

if [ ! -d "./monitoring/grafana_data" ]; then
  echo "Creating Grafana data directory..."
  mkdir -p ./monitoring/grafana_data
  chmod 777 ./monitoring/grafana_data
fi

# Check if the containers are already running
if docker ps | grep -q "security-prometheus" || docker ps | grep -q "security-grafana"; then
  echo "Monitoring containers already running. Stopping them first..."
  docker stop security-prometheus security-grafana 2>/dev/null || true
  docker rm security-prometheus security-grafana 2>/dev/null || true
fi

# Start Prometheus
echo "Starting Prometheus..."
docker run -d \
  --name security-prometheus \
  --restart unless-stopped \
  -p 9090:9090 \
  -v "$(pwd)/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml" \
  -v "$(pwd)/monitoring/prometheus_data:/prometheus" \
  prom/prometheus:v2.45.0 \
  --config.file=/etc/prometheus/prometheus.yml \
  --storage.tsdb.path=/prometheus \
  --web.console.libraries=/usr/share/prometheus/console_libraries \
  --web.console.templates=/usr/share/prometheus/consoles

# Start Grafana
echo "Starting Grafana..."
docker run -d \
  --name security-grafana \
  --restart unless-stopped \
  -p 3000:3000 \
  -v "$(pwd)/monitoring/grafana_data:/var/lib/grafana" \
  -v "$(pwd)/monitoring/grafana/provisioning:/etc/grafana/provisioning" \
  -v "$(pwd)/monitoring/grafana/dashboards:/var/lib/grafana/dashboards" \
  -e "GF_SECURITY_ADMIN_PASSWORD=securepassword" \
  -e "GF_USERS_ALLOW_SIGN_UP=false" \
  -e "GF_INSTALL_PLUGINS=grafana-piechart-panel,grafana-worldmap-panel" \
  grafana/grafana:10.0.3

echo "Monitoring infrastructure started!"
echo "Prometheus is available at http://localhost:9090"
echo "Grafana is available at http://localhost:3000"
echo "Grafana login: admin / securepassword"
echo ""
echo "To stop monitoring, run: docker stop security-prometheus security-grafana" 