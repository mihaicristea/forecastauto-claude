#!/usr/bin/env python3
import os
import sys

# Add src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

from enhanced_car_editor import EnhancedCarImageEditor
from ai_beautifier import AIBeautifier
from PIL import Image

# Enhance image with AI-based beautify functionality
def beautify_image(input_path, output_path):
    print("✨ Applying AI-based beautify enhancements...")
    
    # Initialize AI beautifier
    beautifier = AIBeautifier()
    
    # Load input image
    input_image = Image.open(input_path)
    
    # Test different enhancement styles
    styles = [
        ("professional", "glossy"),
        ("dramatic", "metallic"),
        ("subtle", "chrome")
    ]
    
    for i, (enhancement_level, style) in enumerate(styles):
        print(f"   🎨 Applying {enhancement_level} {style} enhancement...")
        
        # Apply AI beautification
        beautified = beautifier.beautify_car(
            car_image=input_image,
            enhancement_level=enhancement_level,
            style=style
        )
        
        # Save with style suffix
        style_output = output_path.replace('.jpg', f'_{style}_{enhancement_level}.jpg')
        beautified.save(style_output, quality=95)
        print(f"   ✅ Saved: {style_output}")
    
    # Create a comparison image with all styles
    comparison_path = output_path.replace('.jpg', '_ai_comparison.jpg')
    
    # Load all beautified versions for comparison
    beautified_images = []
    for enhancement_level, style in styles:
        style_path = output_path.replace('.jpg', f'_{style}_{enhancement_level}.jpg')
        beautified_images.append(Image.open(style_path))
    
    # Create comparison grid
    create_comparison_grid([input_image] + beautified_images, 
                          ["Original", "Glossy Pro", "Metallic Drama", "Chrome Subtle"], 
                          comparison_path)
    
    print(f"✅ AI beautification complete with comparison: {comparison_path}")
    
    # Cleanup
    beautifier.cleanup()

def create_comparison_grid(images, labels, output_path):
    """Create a comparison grid of images"""
    # Resize all images to same size
    target_size = (400, 300)
    resized_images = [img.resize(target_size, Image.Resampling.LANCZOS) for img in images]
    
    # Create grid (2x2)
    grid_width = target_size[0] * 2
    grid_height = target_size[1] * 2
    grid = Image.new('RGB', (grid_width, grid_height), 'white')
    
    # Paste images
    positions = [(0, 0), (target_size[0], 0), (0, target_size[1]), (target_size[0], target_size[1])]
    for i, (img, pos) in enumerate(zip(resized_images, positions)):
        if i < len(resized_images):
            grid.paste(img, pos)
    
    grid.save(output_path, quality=95)
    print(f"   📊 Comparison grid saved: {output_path}")

def main():
    # Initialize enhanced editor
    print("🚀 Initializing Enhanced Car Image Editor...")
    print("   ✨ Combines perfect background removal from your original script")
    print("   ✨ With professional finishing from the current implementation")
    
    editor = EnhancedCarImageEditor()
    
    # Test with different background types
    input_image = "input/car.jpg"
    
    if not os.path.exists(input_image):
        print(f"❌ Error: Input image not found at {input_image}")
        return
    
    # Test different background styles with enhanced processing
    backgrounds = ['showroom', 'gradient', 'studio']
    
    for bg_type in backgrounds:
        output_path = f"output/enhanced_{bg_type}.jpg"
        print(f"\n🎨 Creating enhanced {bg_type} style image...")
        
        success = editor.process_image(
            input_path=input_image,
            output_path=output_path,
            background_type=bg_type,
            logo_text='Forecast AUTO',
            use_logo_plate=True  # Enable smart license plate replacement
        )
        
        if success:
            print(f"✅ Successfully created enhanced version: {output_path}")
        else:
            print(f"❌ Failed to create: {output_path}")
    
    # Create a comparison version with original method for testing
    print(f"\n📊 Creating comparison with original background...")
    success = editor.process_image(
        input_path=input_image,
        output_path="output/enhanced_comparison.jpg",
        background_type='showroom',
        logo_text='Forecast AUTO',
        use_logo_plate=True
    )
    
    if success:
        print(f"✅ Successfully created comparison: output/enhanced_comparison.jpg")
    
    # Beautify the image using AI enhancements
    beautify_output_path = "output/beautified_car.jpg"
    beautify_image(input_image, beautify_output_path)
    
    print("\n✨ Enhanced car image processing complete!")
    print("\n🔍 Key improvements:")
    print("   • Perfect background removal using simple rembg (like your original)")
    print("   • Smart license plate detection and replacement")
    print("   • Professional color matching with background")
    print("   • Natural edge blending")
    print("   • Realistic shadow generation")
    print("   • Professional finishing touches")
    print("\n📸 Check the output folder for results!")

if __name__ == "__main__":
    main()
