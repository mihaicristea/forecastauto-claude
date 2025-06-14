#!/usr/bin/env python3
import os
import sys
import argparse

# Simple path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from car_editor import CarImageEditor

def main():
    print("🚗 Car Image Editor v1.0")
    print("=" * 50)
    
    parser = argparse.ArgumentParser(description='Professional Car Image Editor')
    parser.add_argument('--input', type=str, required=True, help='Input image path')
    parser.add_argument('--output', type=str, default='/app/output/edited.jpg', help='Output image path')
    parser.add_argument('--logo-text', type=str, default='Forecast AUTO', help='Logo text')
    
    args = parser.parse_args()
    
    # Check input
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        sys.exit(1)
    
    # Process
    editor = CarImageEditor()
    result = editor.process_image(
        input_path=args.input,
        output_path=args.output,
        logo_text=args.logo_text
    )
    
    if result:
        print("\n✨ Done!")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
