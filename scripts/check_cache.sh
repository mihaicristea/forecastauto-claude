#!/bin/bash

# Test cache functionality for AI models
# This script helps debug cache issues

echo "🔍 AI Models Cache Diagnostic Tool"
echo "=================================="
echo ""

# Check local cache directories
echo "📁 Local cache directories:"
if [ -d "models/cache" ]; then
    echo "✅ models/cache exists"
    du -sh models/cache/* 2>/dev/null | while read size path; do
        echo "   $size - $(basename "$path")"
    done
else
    echo "❌ models/cache does not exist"
fi
echo ""

# Check Docker volume mapping
echo "🐳 Docker volume mapping test:"
docker compose run --rm car-editor ls -la /app/models/cache/ 2>/dev/null && echo "✅ Docker can access cache" || echo "❌ Docker cannot access cache"
echo ""

# Check environment variables in Docker
echo "🌍 Environment variables in Docker:"
docker compose run --rm car-editor bash -c 'echo "TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"; echo "HF_HOME=$HF_HOME"; echo "HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE"' 2>/dev/null || echo "❌ Cannot check environment"
echo ""

# Test model loading without network (if cache exists)
if [ -d "models/cache/controlnet-canny" ] && [ -d "models/cache/stable-diffusion-v1-5" ]; then
    echo "🧪 Testing offline model loading..."
    echo "   This should work without internet if cache is properly set up"
    # TODO: Add offline test when ready
else
    echo "⚠️  Cannot test offline loading - cache not complete"
fi
echo ""

echo "💡 Recommendations:"
if [ ! -d "models/cache/controlnet-canny" ] || [ ! -d "models/cache/stable-diffusion-v1-5" ]; then
    echo "   1. Run: ./scripts/setup_cache.sh"
    echo "   2. This will download and cache models permanently"
fi

if [ ! -d "models/cache" ]; then
    echo "   1. Create cache directory: mkdir -p models/cache"
    echo "   2. Run setup: ./scripts/setup_cache.sh"
fi

echo "   3. After caching, models won't be re-downloaded"
echo "   4. Cache persists between Docker container restarts"
