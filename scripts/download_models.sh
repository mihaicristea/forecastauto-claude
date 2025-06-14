#!/bin/bash

echo "🚀 Pre-downloading AI models for caching..."
echo "This will download models once and cache them locally."
echo "Future runs will be much faster!"
echo ""

# Create cache directory
mkdir -p models/cache

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose not found. Please install Docker Compose first."
    exit 1
fi

echo "🐳 Building Docker image if needed..."
# Use docker compose or docker-compose based on availability
if docker compose version &> /dev/null; then
    DOCKER_COMPOSE_CMD="docker compose"
else
    DOCKER_COMPOSE_CMD="docker-compose"
fi

# Build the image
$DOCKER_COMPOSE_CMD build car-editor

echo "📥 Running model download inside Docker container..."

# Run the model download inside Docker container
$DOCKER_COMPOSE_CMD run --rm car-editor python3 -c "
import sys
sys.path.append('src')
from ai_beautifier import AIBeautifier
import torch

print('📥 Initializing AI Beautifier to download models...')
try:
    beautifier = AIBeautifier()
    print('✅ Models downloaded and cached successfully!')
    beautifier.cleanup()
except Exception as e:
    print(f'❌ Error downloading models: {e}')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Model download complete!"
    echo "   Models are now cached in: models/cache/"
    echo "   Future runs will use cached models and start much faster."
else
    echo ""
    echo "❌ Model download failed!"
    echo "   You can still run the main script - models will download on first use."
    exit 1
fi
