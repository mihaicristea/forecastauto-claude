#!/bin/bash

# Build and run AI beautifier tests with Docker
echo "Building Docker image..."
docker compose build ai-beautifier-test

echo "Running AI beautifier tests..."
docker compose run --rm ai-beautifier-test

echo "Test results saved in output/ directory"
