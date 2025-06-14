#!/bin/bash

# Build and run all tests with Docker
echo "Building Docker image..."
docker compose build

echo "Running enhanced car editor tests..."
docker compose run --rm car-editor python3 test_enhanced.py

echo "Running professional tests..."
docker compose run --rm car-editor python3 test_professional.py

echo "Running AI beautifier tests..."
docker compose run --rm ai-beautifier-test

echo "All test results saved in output/ directory"
