#!/usr/bin/env python3
"""
Comprehensive Demo for Professional Car Image Editor with AI Beautifier v2.0
Demonstrates all features including AI beautification, color variations, and different styles
"""

import os
import sys
import time
from PIL import Image

# Add src to path
sys.path.insert(0, 'src')

from car_editor import CarImageEditor
from ai_beautifier import AIBeautifier

def create_demo_grid(images_dict, output_path, title="Car Enhancement Comparison"):
    """Create a comparison grid from multiple images"""
    try:
        from PIL import ImageDraw, ImageFont
        
        # Calculate grid dimensions
        num_images = len(images_dict)
        cols = min(3, num_images)
        rows = (num_images + cols - 1) // cols
        
        # Image dimensions
        thumb_size = (400, 300)
        margin = 20
        title_height = 60
        
        # Grid dimensions
        grid_width = cols * thumb_size[0] + (cols + 1) * margin
        grid_height = rows * thumb_size[1] + (rows + 1) * margin + title_height
        
        # Create grid image
        grid = Image.new('RGB', (grid_width, grid_height), (240, 240, 240))
        draw = ImageDraw.Draw(grid)
        
        # Add title
        try:
            font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            font_title = ImageFont.load_default()
            font_label = ImageFont.load_default()
        
        # Center title
        title_bbox = draw.textbbox((0, 0), title, font=font_title)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = (grid_width - title_width) // 2
        draw.text((title_x, 20), title, fill=(50, 50, 50), font=font_title)
        
        # Add images
        for i, (label, image_path) in enumerate(images_dict.items()):
            if not os.path.exists(image_path):
                continue
                
            # Calculate position
            col = i % cols
            row = i // cols
            
            x = margin + col * (thumb_size[0] + margin)
            y = title_height + margin + row * (thumb_size[1] + margin)
            
            # Load and resize image
            img = Image.open(image_path)
            img.thumbnail(thumb_size, Image.Resampling.LANCZOS)
            
            # Center image in thumbnail area
            img_x = x + (thumb_size[0] - img.size[0]) // 2
            img_y = y + (thumb_size[1] - img.size[1]) // 2
            
            grid.paste(img, (img_x, img_y))
            
            # Add label
            label_bbox = draw.textbbox((0, 0), label, font=font_label)
            label_width = label_bbox[2] - label_bbox[0]
            label_x = x + (thumb_size[0] - label_width) // 2
            label_y = y + thumb_size[1] + 5
            
            # Add background for label
            draw.rectangle([label_x - 5, label_y - 2, label_x + label_width + 5, label_y + 20], 
                         fill=(255, 255, 255), outline=(200, 200, 200))
            draw.text((label_x, label_y), label, fill=(50, 50, 50), font=font_label)
        
        # Save grid
        grid.save(output_path, quality=95, optimize=True)
        print(f"✅ Comparison grid saved: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create comparison grid: {e}")
        return False

def demo_ai_beautifier_styles():
    """Demo different AI beautifier styles"""
    print("🎨 AI Beautifier Styles Demo")
    print("=" * 50)
    
    input_path = "input/car.jpg"
    if not os.path.exists(input_path):
        print(f"❌ Input image not found: {input_path}")
        print("   Please add a car image to the input/ directory")
        return False
    
    try:
        # Initialize editor
        editor = CarImageEditor()
        
        # Test different styles and levels
        test_configs = [
            {"style": "glossy", "level": "light", "name": "Glossy Light"},
            {"style": "glossy", "level": "medium", "name": "Glossy Medium"},
            {"style": "glossy", "level": "strong", "name": "Glossy Strong"},
            {"style": "metallic", "level": "medium", "name": "Metallic"},
            {"style": "luxury", "level": "medium", "name": "Luxury"},
            {"style": "matte", "level": "medium", "name": "Matte"},
        ]
        
        results = {}
        
        for config in test_configs:
            print(f"\n🎨 Processing: {config['name']}")
            
            output_path = f"output/demo_{config['style']}_{config['level']}.jpg"
            
            start_time = time.time()
            success = editor.process_image(
                input_path=input_path,
                output_path=output_path,
                background_type="showroom",
                ai_style=config['style'],
                ai_level=config['level'],
                logo_text="Forecast AUTO"
            )
            processing_time = time.time() - start_time
            
            if success:
                results[config['name']] = output_path
                print(f"   ✅ Completed in {processing_time:.2f}s")
            else:
                print(f"   ❌ Failed")
        
        # Create comparison grid
        if results:
            create_demo_grid(results, "output/demo_ai_styles_comparison.jpg", 
                           "AI Beautifier Styles Comparison")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False

def demo_background_types():
    """Demo different background types"""
    print("\n🖼️  Background Types Demo")
    print("=" * 50)
    
    input_path = "input/car.jpg"
    if not os.path.exists(input_path):
        print(f"❌ Input image not found: {input_path}")
        return False
    
    try:
        editor = CarImageEditor()
        
        backgrounds = ["showroom", "studio", "gradient"]
        results = {}
        
        for bg_type in backgrounds:
            print(f"\n🖼️  Processing: {bg_type.capitalize()} background")
            
            output_path = f"output/demo_bg_{bg_type}.jpg"
            
            start_time = time.time()
            success = editor.process_image(
                input_path=input_path,
                output_path=output_path,
                background_type=bg_type,
                ai_style="glossy",
                ai_level="medium",
                logo_text="Forecast AUTO"
            )
            processing_time = time.time() - start_time
            
            if success:
                results[f"{bg_type.capitalize()} Background"] = output_path
                print(f"   ✅ Completed in {processing_time:.2f}s")
            else:
                print(f"   ❌ Failed")
        
        # Create comparison grid
        if results:
            create_demo_grid(results, "output/demo_backgrounds_comparison.jpg", 
                           "Background Types Comparison")
        
        return True
        
    except Exception as e:
        print(f"❌ Background demo failed: {e}")
        return False

def demo_color_variations():
    """Demo color variations feature"""
    print("\n🌈 Color Variations Demo")
    print("=" * 50)
    
    input_path = "input/car.jpg"
    if not os.path.exists(input_path):
        print(f"❌ Input image not found: {input_path}")
        return False
    
    try:
        # First create a base enhanced image
        editor = CarImageEditor()
        base_output = "output/demo_base_for_colors.jpg"
        
        print("🎨 Creating base enhanced image...")
        success = editor.process_image(
            input_path=input_path,
            output_path=base_output,
            background_type="showroom",
            ai_style="glossy",
            ai_level="medium",
            logo_text="Forecast AUTO"
        )
        
        if not success:
            print("❌ Failed to create base image")
            return False
        
        # Generate color variations
        if editor.ai_beautifier:
            print("\n🎨 Generating color variations...")
            
            # Load the enhanced car image
            enhanced_car = Image.open(base_output)
            
            colors = ["red", "blue", "black", "white", "silver", "gold"]
            
            start_time = time.time()
            variations = editor.ai_beautifier.create_paint_variations(enhanced_car, colors)
            processing_time = time.time() - start_time
            
            results = {"Original": base_output}
            
            for color, variation in variations.items():
                output_path = f"output/demo_color_{color}.jpg"
                variation.save(output_path, quality=95, optimize=True)
                results[f"{color.capitalize()}"] = output_path
                print(f"   ✅ {color.capitalize()} variation saved")
            
            print(f"   ⏱️  Total color variations time: {processing_time:.2f}s")
            
            # Create comparison grid
            create_demo_grid(results, "output/demo_color_variations.jpg", 
                           "Color Variations Showcase")
            
            return True
        else:
            print("⚠️  AI Beautifier not available for color variations")
            return False
        
    except Exception as e:
        print(f"❌ Color variations demo failed: {e}")
        return False

def demo_enhancement_levels():
    """Demo different enhancement levels"""
    print("\n⚡ Enhancement Levels Demo")
    print("=" * 50)
    
    input_path = "input/car.jpg"
    if not os.path.exists(input_path):
        print(f"❌ Input image not found: {input_path}")
        return False
    
    try:
        editor = CarImageEditor()
        
        levels = ["light", "medium", "strong"]
        results = {}
        
        for level in levels:
            print(f"\n⚡ Processing: {level.capitalize()} enhancement")
            
            output_path = f"output/demo_level_{level}.jpg"
            
            start_time = time.time()
            success = editor.process_image(
                input_path=input_path,
                output_path=output_path,
                background_type="showroom",
                ai_style="glossy",
                ai_level=level,
                logo_text="Forecast AUTO"
            )
            processing_time = time.time() - start_time
            
            if success:
                results[f"{level.capitalize()} Enhancement"] = output_path
                print(f"   ✅ Completed in {processing_time:.2f}s")
            else:
                print(f"   ❌ Failed")
        
        # Create comparison grid
        if results:
            create_demo_grid(results, "output/demo_enhancement_levels.jpg", 
                           "Enhancement Levels Comparison")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhancement levels demo failed: {e}")
        return False

def create_final_showcase():
    """Create a final showcase with the best results"""
    print("\n🏆 Creating Final Showcase")
    print("=" * 50)
    
    # Select best images for final showcase
    showcase_images = {
        "Original": "input/car.jpg",
        "Glossy Medium": "output/demo_glossy_medium.jpg",
        "Luxury Style": "output/demo_luxury_medium.jpg",
        "Metallic Finish": "output/demo_metallic_medium.jpg",
        "Studio Background": "output/demo_bg_studio.jpg",
        "Red Variation": "output/demo_color_red.jpg"
    }
    
    # Filter existing images
    existing_images = {k: v for k, v in showcase_images.items() if os.path.exists(v)}
    
    if len(existing_images) >= 2:
        create_demo_grid(existing_images, "output/final_showcase.jpg", 
                        "Professional Car Image Editor - AI Beautifier Showcase")
        return True
    else:
        print("⚠️  Not enough demo images for final showcase")
        return False

def main():
    """Main demo function"""
    print("🚗 Professional Car Image Editor with AI Beautifier v2.0")
    print("🎯 Comprehensive Feature Demonstration")
    print("=" * 70)
    
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # Check for input image
    if not os.path.exists("input/car.jpg"):
        print("❌ No input image found!")
        print("💡 Please add a car image as 'input/car.jpg' to run the demo")
        return
    
    # Run all demos
    demos = [
        ("AI Beautifier Styles", demo_ai_beautifier_styles),
        ("Background Types", demo_background_types),
        ("Enhancement Levels", demo_enhancement_levels),
        ("Color Variations", demo_color_variations),
    ]
    
    results = {}
    total_start = time.time()
    
    for demo_name, demo_func in demos:
        print(f"\n{'='*70}")
        print(f"🎯 Running: {demo_name}")
        print(f"{'='*70}")
        
        start_time = time.time()
        success = demo_func()
        demo_time = time.time() - start_time
        
        results[demo_name] = success
        print(f"\n⏱️  {demo_name} completed in {demo_time:.2f}s")
        
        if success:
            print(f"✅ {demo_name}: SUCCESS")
        else:
            print(f"❌ {demo_name}: FAILED")
    
    # Create final showcase
    create_final_showcase()
    
    total_time = time.time() - total_start
    
    # Summary
    print(f"\n{'='*70}")
    print("📊 DEMO SUMMARY")
    print(f"{'='*70}")
    
    for demo_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{demo_name}: {status}")
    
    successful_demos = sum(results.values())
    total_demos = len(results)
    
    print(f"\n🎯 Results: {successful_demos}/{total_demos} demos successful")
    print(f"⏱️  Total time: {total_time:.2f}s")
    
    if successful_demos == total_demos:
        print("\n🎉 All demos completed successfully!")
        print("📁 Check the output/ directory for all generated images")
        print("\n🔧 Features demonstrated:")
        print("   ✅ AI Beautifier with ControlNet + Stable Diffusion")
        print("   ✅ Multiple enhancement styles (glossy, metallic, luxury, matte)")
        print("   ✅ Different enhancement levels (light, medium, strong)")
        print("   ✅ Professional backgrounds (showroom, studio, gradient)")
        print("   ✅ Automatic color variations")
        print("   ✅ Professional compositing with shadows and reflections")
        print("   ✅ Memory management and GPU optimization")
    else:
        print(f"\n⚠️  {total_demos - successful_demos} demos failed")
        print("   Check error messages above for details")
        print("   The system includes fallback methods for robustness")

if __name__ == "__main__":
    main()
