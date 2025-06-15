#!/bin/bash

echo "🚗 Testing Perspective-Aware License Plate Processor"
echo "=" * 60

# Test 1: EU style plate with studio background
echo "🧪 Test 1: EU style plate with studio background..."
docker compose run --rm car-editor python3 src/main.py \
    --input input/car.jpg \
    --output output/test_eu_studio.jpg \
    --background studio \
    --disable-ai \
    --plate-text "FOR AUTO" \
    --plate-style eu \
    --verbose

# Test 2: Romanian style plate with showroom background
echo "🧪 Test 2: Romanian style plate with showroom background..."
docker compose run --rm car-editor python3 src/main.py \
    --input input/car.jpg \
    --output output/test_ro_showroom.jpg \
    --background showroom \
    --disable-ai \
    --plate-text "FORECAST" \
    --plate-style ro \
    --verbose

# Test 3: US style plate with gradient background
echo "🧪 Test 3: US style plate with gradient background..."
docker compose run --rm car-editor python3 src/main.py \
    --input input/car.jpg \
    --output output/test_us_gradient.jpg \
    --background gradient \
    --disable-ai \
    --plate-text "AUTO123" \
    --plate-style us \
    --verbose

# Test 4: Direct perspective processor test
echo "🧪 Test 4: Direct perspective processor test..."
docker compose run --rm car-editor python3 -c "
from src.license_plate_perspective import PerspectivePlateProcessor
processor = PerspectivePlateProcessor()
result = processor.process_image('input/car.jpg', 'PERSPECTIVE', 'eu', 'output/direct_perspective_test.jpg')
print('✅ Direct perspective test completed!')
"

echo "🎉 All perspective plate tests completed!"
echo "📁 Check output/ directory for results"
