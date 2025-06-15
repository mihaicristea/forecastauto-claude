#!/bin/bash

# Fix dependencies and rebuild Docker image to resolve warnings

echo "🔧 Fixing AI Beautifier warnings and dependencies..."
echo ""

# Rebuild Docker image with updated requirements
echo "🐳 Rebuilding Docker image with updated dependencies..."
docker compose build --no-cache car-editor

echo ""
echo "✅ Dependencies updated and Docker image rebuilt!"
echo ""
echo "Now run your commands - warnings should be reduced:"
echo "  ./scripts/run.sh -i input/car.jpg -o output/enhanced_gradient.jpg -b gradient"
echo ""
echo "Note: Some warnings may still appear but are harmless."
