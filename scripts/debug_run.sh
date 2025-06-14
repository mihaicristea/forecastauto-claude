#!/bin/bash

# Debug script for Professional Car Image Editor
# This script helps identify issues with the main run.sh script

echo "🔍 Debug Script for Car Image Editor"
echo "===================================="

# Check current directory
echo "📁 Current directory: $(pwd)"

# Check if we're in the right directory
if [[ ! -f "src/main.py" ]]; then
    echo "❌ Error: Not in the correct project directory"
    echo "   Please run this script from the project root directory"
    exit 1
fi

# Check input directory and files
echo ""
echo "📂 Checking input directory..."
if [[ -d "input" ]]; then
    echo "✅ input/ directory exists"
    echo "📋 Files in input/:"
    ls -la input/
else
    echo "❌ input/ directory not found"
    echo "💡 Creating input/ directory..."
    mkdir -p input
fi

# Check for car.jpg specifically
if [[ -f "input/car.jpg" ]]; then
    echo "✅ input/car.jpg found"
    echo "📊 File info:"
    ls -lh input/car.jpg
    file input/car.jpg
else
    echo "❌ input/car.jpg not found"
    echo "💡 Please add a car image as input/car.jpg"
fi

# Check output directory
echo ""
echo "📂 Checking output directory..."
if [[ -d "output" ]]; then
    echo "✅ output/ directory exists"
else
    echo "⚠️  output/ directory not found"
    echo "💡 Creating output/ directory..."
    mkdir -p output
fi

# Check Docker
echo ""
echo "🐳 Checking Docker..."
if command -v docker &> /dev/null; then
    echo "✅ Docker is installed"
    docker --version
    
    if command -v docker-compose &> /dev/null || docker compose version &> /dev/null; then
        echo "✅ Docker Compose is available"
        if docker compose version &> /dev/null; then
            docker compose version
        else
            docker-compose --version
        fi
    else
        echo "❌ Docker Compose not found"
    fi
else
    echo "❌ Docker not found"
fi

# Check Python environment
echo ""
echo "🐍 Checking Python environment..."
if [[ -f "requirements.txt" ]]; then
    echo "✅ requirements.txt found"
    echo "📋 Required packages:"
    head -10 requirements.txt
else
    echo "❌ requirements.txt not found"
fi

# Check scripts
echo ""
echo "📜 Checking scripts..."
if [[ -f "scripts/run.sh" ]]; then
    echo "✅ scripts/run.sh found"
    echo "🔧 Permissions:"
    ls -la scripts/run.sh
    
    if [[ -x "scripts/run.sh" ]]; then
        echo "✅ scripts/run.sh is executable"
    else
        echo "⚠️  scripts/run.sh is not executable"
        echo "💡 Making it executable..."
        chmod +x scripts/run.sh
    fi
else
    echo "❌ scripts/run.sh not found"
fi

# Check environment variables
echo ""
echo "🌍 Checking environment variables..."
echo "INPUT_FILE: ${INPUT_FILE:-'not set'}"
echo "OUTPUT_FILE: ${OUTPUT_FILE:-'not set'}"
echo "BACKGROUND: ${BACKGROUND:-'not set'}"

# Test basic Python import
echo ""
echo "🧪 Testing Python imports..."
if python3 -c "import sys; print('Python version:', sys.version)" 2>/dev/null; then
    echo "✅ Python3 is working"
else
    echo "❌ Python3 not working"
fi

# Test Docker image
echo ""
echo "🐳 Checking Docker image..."
if docker images | grep -q "claude-forecastauto-car-editor"; then
    echo "✅ Docker image exists"
    docker images | grep "claude-forecastauto-car-editor"
else
    echo "⚠️  Docker image not found"
    echo "💡 You may need to build it with: docker compose build car-editor"
fi

# Summary
echo ""
echo "📊 SUMMARY"
echo "=========="

# Check all requirements
all_good=true

if [[ ! -f "input/car.jpg" ]]; then
    echo "❌ Missing input/car.jpg"
    all_good=false
fi

if [[ ! -x "scripts/run.sh" ]]; then
    echo "❌ scripts/run.sh not executable"
    all_good=false
fi

if ! command -v docker &> /dev/null; then
    echo "❌ Docker not installed"
    all_good=false
fi

if $all_good; then
    echo "✅ All checks passed! You should be able to run:"
    echo "   ./scripts/run.sh"
else
    echo "⚠️  Some issues found. Please fix them before running the main script."
fi

echo ""
echo "💡 If you continue to have issues, try:"
echo "   1. Make sure you're in the project root directory"
echo "   2. Add a car image as input/car.jpg"
echo "   3. Run: chmod +x scripts/run.sh"
echo "   4. Run: docker compose build car-editor"
echo "   5. Then try: ./scripts/run.sh"
