#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🧪 AI Beautifier Test Suite${NC}"
echo "=============================="

# Check if models are cached
if [ -d "models/cache/controlnet-canny" ] && [ -d "models/cache/stable-diffusion-v1-5" ]; then
    echo -e "${GREEN}✅ Models are cached - test will run quickly${NC}"
else
    echo -e "${YELLOW}⚠️  Models not cached - first run will download ~3GB${NC}"
    echo -e "${YELLOW}   Consider running: ./scripts/setup_cache.sh${NC}"
    echo ""
fi

# Build and run AI beautifier tests with Docker
echo -e "${BLUE}🔨 Building Docker image...${NC}"
docker compose build ai-beautifier-test

echo -e "${BLUE}🧪 Running AI beautifier tests...${NC}"
docker compose run --rm ai-beautifier-test

echo -e "${GREEN}✅ Test results saved in output/ directory${NC}"

# Show cache size after test
echo ""
echo -e "${BLUE}📊 Cache status:${NC}"
docker compose run --rm ai-beautifier-test ls -lah /app/models/cache/ 2>/dev/null || echo "Cache not accessible"
