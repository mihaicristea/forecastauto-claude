#!/bin/bash

echo "🚗 Testing Car Editor WITHOUT AI Beautifier"
echo "============================================="
echo ""

# Create output directory if it doesn't exist
mkdir -p output

echo "1. Testing with gradient background..."
docker compose run --rm car-editor python3 src/main.py \
    --input input/car.jpg \
    --output output/car_gradient_no_ai.jpg \
    --disable-ai \
    --background gradient \
    --plate-text "FOR CAST" \
    --plate-style eu \
    --enhance

echo ""
echo "2. Testing with urban background..."
docker compose run --rm car-editor python3 src/main.py \
    --input input/car.jpg \
    --output output/car_urban_no_ai.jpg \
    --disable-ai \
    --background urban \
    --plate-text "MIHAI AUTO" \
    --plate-style eu \
    --enhance

echo ""
echo "3. Testing just enhancement without backgrounds..."
docker compose run --rm car-editor python3 src/main.py \
    --input input/car.jpg \
    --output output/car_enhanced_only.jpg \
    --disable-ai \
    --background gradient \
    --plate-text "TEST AUTO" \
    --plate-style eu \
    --enhance \
    --verbose

echo ""
echo "4. Testing Romanian plate style..."
docker compose run --rm car-editor python3 src/main.py \
    --input input/car.jpg \
    --output output/car_ro_plate.jpg \
    --disable-ai \
    --background showroom \
    --plate-text "B 123 FOR" \
    --plate-style ro \
    --enhance

echo ""
echo "✅ All tests completed! Results:"
echo ""
ls -la output/car_*_no_ai*.jpg output/car_enhanced_only.jpg output/car_ro_plate*.jpg 2>/dev/null || echo "Some files might not exist"

echo ""
echo "🎯 Files generated (without AI interference):"
echo "   - car_gradient_no_ai_with_plate.jpg"
echo "   - car_urban_no_ai_with_plate.jpg" 
echo "   - car_enhanced_only_with_plate.jpg"
echo "   - car_ro_plate_with_plate.jpg"
echo ""
echo "👀 You can view these to see the car preserved naturally!"
