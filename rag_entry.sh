#!/bin/bash
set -e

PORT_TO_USE="${APP_PORT:-${PORT:-8080}}"

echo "Starting Streamlit app on port ${PORT_TO_USE}..."

exec streamlit run rag-chat-custom.py \
  --server.port="${PORT_TO_USE}" \
  --server.address="0.0.0.0" \
  --server.headless=true \
  --browser.gatherUsageStats=false \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false
