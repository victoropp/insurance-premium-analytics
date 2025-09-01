#!/bin/bash
# Render deployment start script for Insurance Premium Analytics

echo "Starting Videbimus AI - Insurance Premium Analytics Platform..."

# Use environment variable PORT if available, otherwise default to 8050
PORT=${PORT:-8050}

echo "Running on port: $PORT"

# Start the application with gunicorn
exec gunicorn dashboard:server --bind 0.0.0.0:$PORT --workers 2 --timeout 120 --log-level info