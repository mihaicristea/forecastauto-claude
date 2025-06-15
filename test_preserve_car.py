#!/usr/bin/env python3
"""
Test script pentru AI beautifier cu opțiunea preserve_car
Demonstrează diferența între full enhancement și preserve_car mode
"""

import os
import sys
from PIL import Image

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

from ai_beautifier import AIBeautifier

def test_preserve_car_modes():
    """Test both preserve_car modes"""
    print("🧪 Testing AI Beautifier - Preserve Car Modes")
    print("=" * 50)
    
    # Input and output paths
    input_path = "input/car.jpg"
    output_dir = "output"
    
    # Check if input exists
    if not os.path.exists(input_path):
        print(f"❌ Input image not found: {input_path}")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize AI beautifier
    try:
        beautifier = AIBeautifier()
        print("✅ AI Beautifier initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize AI Beautifier: {e}")
        return
    
    # Load input image
    try:
        image = Image.open(input_path)
        print(f"✅ Loaded input image: {image.size}")
    except Exception as e:
        print(f"❌ Failed to load image: {e}")
        return
    
    # Test configurations
    test_configs = [
        # preserve_car=True (doar background enhancement)
        ("preserve_car_glossy", "medium", "glossy", True),
        ("preserve_car_luxury", "medium", "luxury", True),
        
        # preserve_car=False (full enhancement)
        ("full_enhancement_glossy", "medium", "glossy", False),
        ("full_enhancement_luxury", "medium", "luxury", False),
        
        # Different enhancement levels with preserve_car=True
        ("preserve_light", "light", "glossy", True),
        ("preserve_strong", "strong", "glossy", True),
    ]
    
    results = []
    
    for name, level, style, preserve_car in test_configs:
        print(f"\n🎨 Testing: {name}")
        print(f"   Style: {style}, Level: {level}, Preserve Car: {preserve_car}")
        
        try:
            # Apply AI beautification
            result = beautifier.beautify_car(
                car_image=image,
                enhancement_level=level,
                style=style,
                preserve_car=preserve_car
            )
            
            # Save result
            output_path = f"{output_dir}/test_{name}.jpg"
            result.save(output_path, "JPEG", quality=95)
            print(f"   ✅ Saved: {output_path}")
            
            results.append((name, output_path, result))
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    # Create comparison grid
    if len(results) >= 4:
        print(f"\n📊 Creating comparison grid...")
        create_comparison_grid(image, results, f"{output_dir}/preserve_car_comparison.jpg")
    
    print(f"\n✨ Testing complete!")
    print(f"📁 Results saved in: {output_dir}/")
    print(f"\n🔍 Compare results:")
    print(f"   • preserve_car=True: Background enhanced, car unchanged")
    print(f"   • preserve_car=False: Full image enhanced")
    
    # Cleanup
    beautifier.cleanup()

def create_comparison_grid(original, results, output_path):
    """Create a comparison grid showing different modes"""
    try:
        # Resize images for grid
        target_size = (300, 225)
        
        # Prepare images
        images = [original.resize(target_size, Image.Resampling.LANCZOS)]
        labels = ["Original"]
        
        # Add first 5 results
        for name, path, result_img in results[:5]:
            images.append(result_img.resize(target_size, Image.Resampling.LANCZOS))
            labels.append(name.replace("_", " ").title())
        
        # Create grid (3x2)
        grid_width = target_size[0] * 3
        grid_height = target_size[1] * 2
        grid = Image.new('RGB', (grid_width, grid_height), 'white')
        
        # Paste images
        positions = [
            (0, 0), (target_size[0], 0), (target_size[0] * 2, 0),
            (0, target_size[1]), (target_size[0], target_size[1]), (target_size[0] * 2, target_size[1])
        ]
        
        for i, (img, pos) in enumerate(zip(images, positions)):
            if i < len(images):
                grid.paste(img, pos)
        
        grid.save(output_path, quality=95)
        print(f"   📊 Comparison grid saved: {output_path}")
        
    except Exception as e:
        print(f"   ⚠️ Failed to create comparison grid: {e}")

if __name__ == "__main__":
    test_preserve_car_modes()
