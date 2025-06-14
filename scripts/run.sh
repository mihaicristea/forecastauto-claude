#!/bin/bash

# Use modern Docker GPU support
DOCKER_CMD="docker run --gpus all"

# Get absolute path of current directory
PROJECT_DIR=$(pwd)

echo "📁 Project directory: $PROJECT_DIR"
echo "🚀 Starting car-image-editor..."

# Run the container with proper volume mounts
$DOCKER_CMD \
    -v "$PROJECT_DIR/src:/app/src:ro" \
    -v "$PROJECT_DIR/input:/app/input" \
    -v "$PROJECT_DIR/output:/app/output" \
    -v "$PROJECT_DIR/backgrounds:/app/backgrounds" \
    car-image-editor \
    python3 /app/src/main.py "$@"
