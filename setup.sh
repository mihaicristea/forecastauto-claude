#!/bin/bash

echo "=== Car Editor Docker Setup ==="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed. Please install Docker first."
    echo "Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! docker compose version &> /dev/null; then
    echo "Error: Docker Compose is not available. Please install Docker Compose first."
    echo "Visit: https://docs.docker.com/compose/install/"
    exit 1
fi

# Check if NVIDIA Docker runtime is available (for GPU support)
if docker info | grep -q nvidia; then
    echo "✓ NVIDIA Docker runtime detected - GPU acceleration will be available"
else
    echo "⚠ NVIDIA Docker runtime not detected - running in CPU mode"
    echo "For GPU acceleration, install nvidia-docker2:"
    echo "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
fi

echo ""
echo "Creating necessary directories..."
mkdir -p input output backgrounds models

echo "Making scripts executable..."
chmod +x scripts/*.sh

echo ""
echo "Building Docker image..."
./scripts/build.sh

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "Usage:"
echo "  ./scripts/run.sh           - Run the main car editor"
echo "  ./scripts/run_demo.sh      - Run AI beautifier demo"
echo "  ./scripts/test_ai_beautifier.sh - Run AI beautifier tests"
echo "  ./scripts/run_test.sh      - Run all tests"
echo ""
echo "Make sure to place your input images in the 'input/' directory"
echo "Results will be saved in the 'output/' directory"
