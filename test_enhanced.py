#!/usr/bin/env python3
import os
import sys

# Add src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

from enhanced_car_editor import EnhancedCarImageEditor

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
