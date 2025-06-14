#!/bin/bash

# Use modern Docker GPU support
DOCKER_CMD="docker run --gpus all"

# Get absolute path of current directory
PROJECT_DIR=$(pwd)

echo "📁 Project directory: $PROJECT_DIR"
echo "🚀 Starting professional car image test..."

# Run the container with proper volume mounts
$DOCKER_CMD \
    -v "$PROJECT_DIR/src:/app/src:ro" \
    -v "$PROJECT_DIR/test_professional.py:/app/test_professional.py:ro" \
    -v "$PROJECT_DIR/background_processor.py:/app/background_processor.py:ro" \
    -v "$PROJECT_DIR/image_enhancer.py:/app/image_enhancer.py:ro" \
    -v "$PROJECT_DIR/text_overlay.py:/app/text_overlay.py:ro" \
    -v "$PROJECT_DIR/config.yaml:/app/config.yaml:ro" \
    -v "$PROJECT_DIR/input:/app/input" \
    -v "$PROJECT_DIR/output:/app/output" \
    -v "$PROJECT_DIR/backgrounds:/app/backgrounds" \
    car-image-editor \
    python3 /app/test_professional.py "$@"
