#!/bin/bash

# Build Docker image for the car editor application
echo "Building Docker image..."
docker compose build

echo "Docker image built successfully!"
echo ""
echo "Available commands:"
echo "  ./scripts/run.sh           - Run the main car editor"
echo "  ./scripts/run_demo.sh      - Run AI beautifier demo"
echo "  ./scripts/test_ai_beautifier.sh - Run AI beautifier tests"
echo "  ./scripts/run_test.sh      - Run all tests"
echo ""
echo "Or use docker compose directly:"
echo "  docker compose run --rm car-editor"
echo "  docker compose run --rm ai-beautifier-demo"
echo "  docker compose run --rm ai-beautifier-test"
