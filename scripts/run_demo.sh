#!/bin/bash

# Build and run AI beautifier demo with Docker
echo "Building Docker image..."
docker compose build ai-beautifier-demo

echo "Running AI beautifier demo..."
docker compose run --rm ai-beautifier-demo

echo "Demo results saved in output/ directory"
