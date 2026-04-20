#!/bin/bash
set -e

export APP_PORT="${CDSW_APP_PORT:-8080}"

gunicorn -b 0.0.0.0:${APP_PORT} app:app
