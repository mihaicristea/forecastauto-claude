#!/usr/bin/env python3
"""Test script for professional car image editing"""

import os
import sys
from src.car_editor import CarImageEditor

def main():
    # Initialize editor
    print("🚀 Initializing Professional Car Image Editor...")
    editor = CarImageEditor()
    
    # Test with different background types
    input_image = "input/car.jpg"
    
    if not os.path.exists(input_image):
        print(f"❌ Error: Input image not found at {input_image}")
        return
    
    # Test different background styles
    backgrounds = ['showroom', 'gradient', 'studio']
    
    for bg_type in backgrounds:
        output_path = f"output/professional_{bg_type}.jpg"
        print(f"\n🎨 Creating {bg_type} style image...")
        
        success = editor.process_image(
            input_path=input_image,
            output_path=output_path,
            background_type=bg_type,
            logo_text='Forecast AUTO'
        )
        
        if success:
            print(f"✅ Successfully created: {output_path}")
        else:
            print(f"❌ Failed to create: {output_path}")
    
    print("\n✨ Professional car image processing complete!")

if __name__ == "__main__":
    main()
