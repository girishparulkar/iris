#!/bin/bash

# 1. Exit immediately if any command fails
set -e

# 2. Set a default port if APP_PORT is not provided by your platform
export APP_PORT=${APP_PORT:-8501}

# 3. Print the port for debugging logs
echo "Starting Streamlit app on port ${APP_PORT}..."

# 4. Start the Streamlit application
exec streamlit run app.py \
    --server.port="${APP_PORT}" \
    --server.address="0.0.0.0" \
    --server.headless=true
