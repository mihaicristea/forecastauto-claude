#!/bin/bash

# Script pentru pre-download și cache al modelelor AI
# Rulează acest script o dată pentru a evita re-descărcarea modelelor

echo "🚀 Pre-downloading AI models to persistent cache..."
echo "This will download ~3GB of models and cache them permanently."
echo ""

# Build the Docker image first
echo "🔨 Building Docker image..."
docker compose build car-editor

# Run the model caching script
echo "📥 Downloading and caching models..."
docker compose run --rm car-editor python3 scripts/cache_models.py

# Check cache status
echo ""
echo "📊 Checking cache status..."
docker compose run --rm car-editor ls -lah /app/models/cache/

echo ""
echo "✅ Model caching complete!"
echo "🚀 Future runs will use cached models and be much faster."
echo ""
echo "Usage:"
echo "  ./scripts/run.sh              # Normal usage (will use cache)"
echo "  ./scripts/test_ai_beautifier.sh  # Test AI beautifier (with cache)"
