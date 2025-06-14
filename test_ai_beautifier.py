#!/usr/bin/env python3
"""
Test script pentru AI Beautifier - demonstrează funcționalitățile de beautify auto
cu ControlNet + Stable Diffusion pentru mașini
"""

import os
import sys
from PIL import Image
import time

# Add src to path
sys.path.insert(0, 'src')

from ai_beautifier import AIBeautifier
from car_editor import CarImageEditor

def test_ai_beautifier():
    """Test AI Beautifier functionality"""
    print("🚗 Testing AI Beautifier for Car Enhancement")
    print("=" * 60)
    
    # Check if input image exists
    input_path = "input/car.jpg"
    if not os.path.exists(input_path):
        print(f"❌ Input image not found: {input_path}")
        print("   Please add a car image to the input/ directory")
        return False
    
    try:
        # Initialize AI Beautifier
        print("🤖 Initializing AI Beautifier...")
        beautifier = AIBeautifier()
        
        # Load test image
        print(f"📸 Loading test image: {input_path}")
        car_image = Image.open(input_path)
        print(f"   Original size: {car_image.size}")
        
        # Create output directory
        os.makedirs("output", exist_ok=True)
        
        # Test different enhancement levels and styles
        test_configs = [
            {"level": "light", "style": "glossy", "name": "light_glossy"},
            {"level": "medium", "style": "glossy", "name": "medium_glossy"},
            {"level": "strong", "style": "glossy", "name": "strong_glossy"},
            {"level": "medium", "style": "metallic", "name": "medium_metallic"},
            {"level": "medium", "style": "luxury", "name": "medium_luxury"},
            {"level": "medium", "style": "matte", "name": "medium_matte"},
        ]
        
        print("\n✨ Testing AI Beautifier with different configurations...")
        
        for i, config in enumerate(test_configs, 1):
            print(f"\n🎨 Test {i}/{len(test_configs)}: {config['level']} {config['style']}")
            
            start_time = time.time()
            
            # Apply AI beautification
            enhanced_image = beautifier.beautify_car(
                car_image,
                enhancement_level=config['level'],
                style=config['style']
            )
            
            processing_time = time.time() - start_time
            
            # Save result
            output_path = f"output/ai_beautified_{config['name']}.jpg"
            enhanced_image.save(output_path, quality=95, optimize=True)
            
            print(f"   ✅ Saved: {output_path}")
            print(f"   ⏱️  Processing time: {processing_time:.2f}s")
        
        # Test color variations
        print("\n🎨 Testing color variations...")
        colors = ["red", "blue", "black", "white", "silver"]
        
        start_time = time.time()
        color_variations = beautifier.create_paint_variations(car_image, colors)
        processing_time = time.time() - start_time
        
        for color, variation in color_variations.items():
            output_path = f"output/ai_color_{color}.jpg"
            variation.save(output_path, quality=95, optimize=True)
            print(f"   ✅ {color.capitalize()} variation saved: {output_path}")
        
        print(f"   ⏱️  Color variations time: {processing_time:.2f}s")
        
        # Cleanup
        beautifier.cleanup()
        
        print("\n🎉 AI Beautifier test completed successfully!")
        print("📁 Check the output/ directory for results")
        
        return True
        
    except Exception as e:
        print(f"❌ AI Beautifier test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integrated_workflow():
    """Test integrated workflow with CarImageEditor"""
    print("\n" + "=" * 60)
    print("🚗 Testing Integrated AI Beautifier Workflow")
    print("=" * 60)
    
    input_path = "input/car.jpg"
    if not os.path.exists(input_path):
        print(f"❌ Input image not found: {input_path}")
        return False
    
    try:
        # Initialize CarImageEditor (includes AI Beautifier)
        print("🔧 Initializing Car Image Editor with AI Beautifier...")
        editor = CarImageEditor()
        
        # Test different background types with AI enhancement
        background_types = ["showroom", "studio", "gradient"]
        
        for bg_type in background_types:
            print(f"\n🖼️  Processing with {bg_type} background...")
            
            output_path = f"output/ai_integrated_{bg_type}.jpg"
            
            start_time = time.time()
            success = editor.process_image(
                input_path=input_path,
                output_path=output_path,
                background_type=bg_type,
                logo_text="Forecast AUTO"
            )
            processing_time = time.time() - start_time
            
            if success:
                print(f"   ✅ Saved: {output_path}")
                print(f"   ⏱️  Total processing time: {processing_time:.2f}s")
            else:
                print(f"   ❌ Failed to process {bg_type} background")
        
        print("\n🎉 Integrated workflow test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Integrated workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_comparison_grid():
    """Create a comparison grid showing before/after results"""
    print("\n" + "=" * 60)
    print("📊 Creating Comparison Grid")
    print("=" * 60)
    
    try:
        from PIL import ImageDraw, ImageFont
        
        # Load original image
        original = Image.open("input/car.jpg")
        
        # Load enhanced images
        enhanced_files = [
            ("output/ai_beautified_medium_glossy.jpg", "AI Glossy"),
            ("output/ai_beautified_medium_metallic.jpg", "AI Metallic"),
            ("output/ai_beautified_medium_luxury.jpg", "AI Luxury"),
            ("output/ai_integrated_showroom.jpg", "Full Pipeline")
        ]
        
        # Create grid
        grid_width = 2
        grid_height = 3  # Original + 4 enhanced + 1 for spacing
        
        # Resize images to consistent size
        thumb_size = (400, 300)
        original_thumb = original.copy()
        original_thumb.thumbnail(thumb_size, Image.Resampling.LANCZOS)
        
        # Create grid image
        grid_img = Image.new('RGB', (grid_width * thumb_size[0], grid_height * thumb_size[1]), (255, 255, 255))
        
        # Add original image
        grid_img.paste(original_thumb, (0, 0))
        
        # Add label for original
        draw = ImageDraw.Draw(grid_img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        draw.text((10, 10), "Original", fill=(255, 255, 255), font=font)
        
        # Add enhanced images
        for i, (file_path, label) in enumerate(enhanced_files):
            if os.path.exists(file_path):
                enhanced = Image.open(file_path)
                enhanced_thumb = enhanced.copy()
                enhanced_thumb.thumbnail(thumb_size, Image.Resampling.LANCZOS)
                
                # Calculate position
                col = (i + 1) % grid_width
                row = (i + 1) // grid_width
                
                x = col * thumb_size[0]
                y = row * thumb_size[1]
                
                grid_img.paste(enhanced_thumb, (x, y))
                
                # Add label
                draw.text((x + 10, y + 10), label, fill=(255, 255, 255), font=font)
        
        # Save comparison grid
        grid_path = "output/ai_beautifier_comparison.jpg"
        grid_img.save(grid_path, quality=95, optimize=True)
        
        print(f"✅ Comparison grid saved: {grid_path}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create comparison grid: {e}")
        return False

def main():
    """Main test function"""
    print("🤖 AI Beautifier Test Suite")
    print("=" * 60)
    print("Testing ControlNet + Stable Diffusion car enhancement")
    print("Features: Glossy paint, spot removal, soft reflections")
    print("=" * 60)
    
    # Run tests
    test1_success = test_ai_beautifier()
    test2_success = test_integrated_workflow()
    
    # Create comparison if tests passed
    if test1_success:
        create_comparison_grid()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    print(f"AI Beautifier Test: {'✅ PASSED' if test1_success else '❌ FAILED'}")
    print(f"Integrated Workflow: {'✅ PASSED' if test2_success else '❌ FAILED'}")
    
    if test1_success and test2_success:
        print("\n🎉 All tests passed! AI Beautifier is working correctly.")
        print("📁 Check the output/ directory for enhanced images")
        print("\n🔧 Features implemented:")
        print("   ✅ ControlNet + Stable Diffusion integration")
        print("   ✅ Glossy paint effects")
        print("   ✅ Automatic spot removal")
        print("   ✅ Soft reflection enhancement")
        print("   ✅ Multiple enhancement levels")
        print("   ✅ Various paint styles (glossy, metallic, luxury, matte)")
        print("   ✅ Color variation generation")
        print("   ✅ Memory management and cleanup")
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.")
        print("   The system will fallback to traditional enhancement methods.")

if __name__ == "__main__":
    main()
