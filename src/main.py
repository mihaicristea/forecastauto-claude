#!/usr/bin/env python3
import os
import sys
import argparse

# Simple path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from enhanced_car_editor import EnhancedCarImageEditor

def main():
    print("🚗 Professional Car Image Editor with AI Beautifier v2.0")
    print("=" * 60)
    
    parser = argparse.ArgumentParser(
        description='Professional Car Image Editor with AI Beautification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic processing
  python3 src/main.py --input input/car.jpg
  
  # With custom background and AI enhancement
  python3 src/main.py --input input/car.jpg --background showroom --ai-style glossy --ai-level medium
  
  # Generate color variations
  python3 src/main.py --input input/car.jpg --color-variations red,blue,black,white
  
  # Professional studio setup
  python3 src/main.py --input input/car.jpg --background studio --ai-style luxury --ai-level strong
        """
    )
    
    # Input/Output
    parser.add_argument('--input', type=str, required=True, 
                       help='Input car image path')
    parser.add_argument('--output', type=str, default='/app/output/edited.jpg', 
                       help='Output image path (default: /app/output/edited.jpg)')
    
    # Background options
    parser.add_argument('--background', type=str, default='showroom',
                       choices=['showroom', 'studio', 'gradient', 'urban', 'custom'],
                       help='Background type (default: showroom)')
    parser.add_argument('--custom-bg', type=str,
                       help='Path to custom background image (use with --background custom)')
    
    # AI Beautifier options
    parser.add_argument('--ai-style', type=str, default='glossy',
                       choices=['glossy', 'matte', 'metallic', 'luxury'],
                       help='AI beautification style (default: glossy)')
    parser.add_argument('--ai-level', type=str, default='medium',
                       choices=['light', 'medium', 'strong', 'professional', 'dramatic', 'subtle'],
                       help='AI enhancement level (default: medium). Standard: light/medium/strong, Enhanced: professional/dramatic/subtle')
    parser.add_argument('--preserve-car', action='store_true',
                       help='Preserve car unchanged, enhance only background')
    parser.add_argument('--disable-ai', action='store_true',
                       help='Disable AI beautification (use traditional enhancement)')
    
    # Color variations
    parser.add_argument('--color-variations', type=str,
                       help='Generate color variations (comma-separated: red,blue,black,white,silver,gold)')
    
    # Branding
    parser.add_argument('--logo-text', type=str, default='Forecast AUTO',
                       help='Logo text (default: Forecast AUTO)')
    parser.add_argument('--no-logo', action='store_true',
                       help='Disable logo overlay')
    
    # Advanced options
    parser.add_argument('--enhance', action='store_true',
                       help='Apply additional image enhancement')
    parser.add_argument('--quality', type=int, default=95, choices=range(70, 101),
                       help='Output JPEG quality (70-100, default: 95)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        sys.exit(1)
    
    if args.background == 'custom' and not args.custom_bg:
        print("❌ --custom-bg is required when using --background custom")
        sys.exit(1)
    
    if args.custom_bg and not os.path.exists(args.custom_bg):
        print(f"❌ Custom background file not found: {args.custom_bg}")
        sys.exit(1)
    
    # Show configuration
    if args.verbose:
        print("🔧 Configuration:")
        print(f"   Input: {args.input}")
        print(f"   Output: {args.output}")
        print(f"   Background: {args.background}")
        print(f"   AI Style: {args.ai_style}")
        print(f"   AI Level: {args.ai_level}")
        print(f"   AI Enabled: {not args.disable_ai}")
        print(f"   Preserve Car: {args.preserve_car}")
        if args.color_variations:
            print(f"   Color Variations: {args.color_variations}")
        print()
    
    try:
        # Initialize editor
        print("🔧 Initializing Enhanced Car Image Editor...")
        editor = EnhancedCarImageEditor()
        
        # Process main image
        print("🎨 Processing main image...")
        result = editor.process_image(
            input_path=args.input,
            output_path=args.output,
            background_type=args.background,
            custom_background=args.custom_bg,
            logo_text=args.logo_text if not args.no_logo else None,
            ai_style=args.ai_style if not args.disable_ai else None,
            ai_level=args.ai_level if not args.disable_ai else None,
            enhance=args.enhance,
            quality=args.quality,
            preserve_car=args.preserve_car
        )
        
        if not result:
            print("❌ Main image processing failed")
            sys.exit(1)
        
        print(f"✅ Main image saved: {args.output}")
        
        # Generate color variations if requested
        if args.color_variations:
            print("\n🎨 Generating color variations...")
            colors = [c.strip() for c in args.color_variations.split(',')]
            
            # Load the processed car for color variations
            from PIL import Image
            processed_car = Image.open(args.output)
            
            # Initialize AI Beautifier for color variations
            if not args.disable_ai and editor.ai_beautifier:
                try:
                    variations = editor.ai_beautifier.create_paint_variations(processed_car, colors)
                    
                    # Save variations
                    base_name = os.path.splitext(args.output)[0]
                    for color, variation in variations.items():
                        variation_path = f"{base_name}_color_{color}.jpg"
                        variation.save(variation_path, quality=args.quality, optimize=True)
                        print(f"   ✅ {color.capitalize()} variation: {variation_path}")
                    
                except Exception as e:
                    print(f"   ⚠️  Color variations failed: {e}")
            else:
                print("   ⚠️  Color variations require AI beautifier (--disable-ai not compatible)")
        
        print(f"\n🎉 Processing completed successfully!")
        print(f"📁 Output saved to: {args.output}")
        
        if args.color_variations:
            print(f"🎨 Color variations saved with suffix '_color_[colorname]'")
        
    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
