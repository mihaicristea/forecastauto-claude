#!/usr/bin/env python3
"""
Demo script pentru AI Beautifier - Demonstrează toate funcționalitățile
ControlNet + Stable Diffusion pentru îmbunătățirea imaginilor de mașini
"""

import os
import sys
import time
from PIL import Image, ImageDraw, ImageFont

# Add src to path
sys.path.insert(0, 'src')

def create_demo_grid():
    """Creează o grilă demo cu toate stilurile și nivelurile"""
    print("🎨 Creating AI Beautifier Demo Grid...")
    
    # Check if we have test results
    test_files = [
        ("input/car.jpg", "Original"),
        ("output/ai_beautified_light_glossy.jpg", "Light Glossy"),
        ("output/ai_beautified_medium_glossy.jpg", "Medium Glossy"),
        ("output/ai_beautified_strong_glossy.jpg", "Strong Glossy"),
        ("output/ai_beautified_medium_metallic.jpg", "Metallic"),
        ("output/ai_beautified_medium_luxury.jpg", "Luxury"),
        ("output/ai_beautified_medium_matte.jpg", "Matte"),
        ("output/ai_integrated_showroom.jpg", "Full Pipeline")
    ]
    
    # Filter existing files
    existing_files = [(path, label) for path, label in test_files if os.path.exists(path)]
    
    if len(existing_files) < 2:
        print("❌ Not enough test results found. Run test_ai_beautifier.py first.")
        return False
    
    # Create grid layout
    cols = 4
    rows = (len(existing_files) + cols - 1) // cols
    
    # Thumbnail size
    thumb_size = (300, 225)
    
    # Create grid image
    grid_width = cols * thumb_size[0]
    grid_height = rows * thumb_size[1]
    grid_img = Image.new('RGB', (grid_width, grid_height), (40, 40, 40))
    
    # Load font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()
    
    draw = ImageDraw.Draw(grid_img)
    
    # Add images to grid
    for i, (file_path, label) in enumerate(existing_files):
        try:
            # Load and resize image
            img = Image.open(file_path)
            img_thumb = img.copy()
            img_thumb.thumbnail(thumb_size, Image.Resampling.LANCZOS)
            
            # Calculate position
            col = i % cols
            row = i // cols
            x = col * thumb_size[0]
            y = row * thumb_size[1]
            
            # Center the thumbnail
            thumb_w, thumb_h = img_thumb.size
            offset_x = (thumb_size[0] - thumb_w) // 2
            offset_y = (thumb_size[1] - thumb_h) // 2
            
            # Paste image
            grid_img.paste(img_thumb, (x + offset_x, y + offset_y))
            
            # Add label with background
            label_y = y + thumb_size[1] - 25
            draw.rectangle([(x, label_y), (x + thumb_size[0], y + thumb_size[1])], fill=(0, 0, 0, 180))
            
            # Center text
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_x = x + (thumb_size[0] - text_width) // 2
            
            draw.text((text_x, label_y + 5), label, fill=(255, 255, 255), font=font)
            
        except Exception as e:
            print(f"   Warning: Could not process {file_path}: {e}")
    
    # Add title
    title = "🤖 AI Beautifier Demo - ControlNet + Stable Diffusion"
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = (grid_width - title_width) // 2
    
    # Add title background
    draw.rectangle([(0, 0), (grid_width, 30)], fill=(20, 20, 20))
    draw.text((title_x, 5), title, fill=(255, 255, 255), font=title_font)
    
    # Save demo grid
    demo_path = "output/ai_beautifier_demo.jpg"
    grid_img.save(demo_path, quality=95, optimize=True)
    
    print(f"✅ Demo grid saved: {demo_path}")
    return True

def print_feature_summary():
    """Afișează un rezumat al funcționalităților implementate"""
    print("\n" + "="*70)
    print("🤖 AI BEAUTIFIER - FEATURE SUMMARY")
    print("="*70)
    
    print("\n🎯 CORE FEATURES:")
    print("   ✅ ControlNet + Stable Diffusion Integration")
    print("   ✅ Professional Car Enhancement")
    print("   ✅ Multiple Enhancement Levels (Light, Medium, Strong)")
    print("   ✅ Various Paint Styles (Glossy, Matte, Metallic, Luxury)")
    print("   ✅ Automatic Spot and Imperfection Removal")
    print("   ✅ Soft Reflection Enhancement")
    print("   ✅ Color Variation Generation")
    print("   ✅ Memory Management and GPU Optimization")
    
    print("\n🎨 PAINT EFFECTS:")
    print("   🔸 Glossy - Mirror-like reflections, wet look")
    print("   🔸 Matte - Smooth elegant finish")
    print("   🔸 Metallic - Pearl finish with sparkle effects")
    print("   🔸 Luxury - Premium showroom quality")
    
    print("\n📊 ENHANCEMENT LEVELS:")
    print("   🔸 Light - Subtle improvements, natural look")
    print("   🔸 Medium - Balanced enhancement (recommended)")
    print("   🔸 Strong - Dramatic improvements, perfect finish")
    
    print("\n🔧 TECHNICAL FEATURES:")
    print("   ✅ Automatic fallback to traditional methods")
    print("   ✅ GPU memory optimization")
    print("   ✅ Configurable inference parameters")
    print("   ✅ Edge refinement and alpha matting")
    print("   ✅ Professional compositing with shadows")
    print("   ✅ Integration with existing pipeline")
    
    print("\n🚀 USAGE:")
    print("   • Test AI features: python3 test_ai_beautifier.py")
    print("   • Run full pipeline: ./scripts/run.sh --input input/car.jpg")
    print("   • Test script: ./scripts/test_ai_beautifier.sh")
    print("   • Demo grid: python3 demo_ai_beautifier.py")
    
    print("\n📁 OUTPUT FILES:")
    output_files = [
        "ai_beautified_*.jpg - Various enhancement styles",
        "ai_color_*.jpg - Color variations",
        "ai_integrated_*.jpg - Full pipeline results",
        "ai_beautifier_comparison.jpg - Before/after comparison",
        "ai_beautifier_demo.jpg - Complete demo grid"
    ]
    
    for file_desc in output_files:
        if any(os.path.exists(f"output/{f}") for f in os.listdir("output") if f.startswith(file_desc.split(" -")[0].replace("*", "").replace(".jpg", ""))):
            print(f"   ✅ {file_desc}")
        else:
            print(f"   ⏳ {file_desc}")

def main():
    """Main demo function"""
    print("🤖 AI Beautifier Demo")
    print("="*50)
    print("Demonstrating ControlNet + Stable Diffusion")
    print("Professional car image enhancement")
    print("="*50)
    
    # Check if output directory exists
    if not os.path.exists("output"):
        os.makedirs("output")
    
    # Create demo grid if test results exist
    demo_created = create_demo_grid()
    
    # Print feature summary
    print_feature_summary()
    
    # Final instructions
    print("\n" + "="*70)
    print("🎯 NEXT STEPS")
    print("="*70)
    
    if not demo_created:
        print("\n1. Run the AI Beautifier test first:")
        print("   python3 test_ai_beautifier.py")
        print("\n2. Or use the test script:")
        print("   ./scripts/test_ai_beautifier.sh")
        print("\n3. Then run this demo again:")
        print("   python3 demo_ai_beautifier.py")
    else:
        print("\n✅ Demo completed successfully!")
        print("\n📁 Check these files in the output/ directory:")
        print("   • ai_beautifier_demo.jpg - Complete demo grid")
        print("   • ai_beautifier_comparison.jpg - Before/after comparison")
        print("   • ai_beautified_*.jpg - Individual enhancement results")
        print("   • ai_integrated_*.jpg - Full pipeline results")
    
    print("\n🔧 For production use:")
    print("   • Integrate AI Beautifier in your workflow")
    print("   • Adjust config.yaml for your needs")
    print("   • Use CarImageEditor class with AI enhancement")
    
    print("\n🎉 AI Beautifier is ready for professional car image enhancement!")

if __name__ == "__main__":
    main()
