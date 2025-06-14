#!/usr/bin/env python3
"""
Simple test script for Car Image Editor
Tests basic functionality without Docker
"""

import os
import sys
import time

# Add src to path
sys.path.insert(0, 'src')

def test_basic_functionality():
    """Test basic functionality"""
    print("🚗 Simple Car Image Editor Test")
    print("=" * 40)
    
    # Check if input image exists
    input_path = "input/car.jpg"
    if not os.path.exists(input_path):
        print(f"❌ Input image not found: {input_path}")
        print("💡 Please add a car image to the input/ directory")
        return False
    
    print(f"✅ Input image found: {input_path}")
    
    try:
        # Test imports
        print("\n🔧 Testing imports...")
        
        from car_editor import CarImageEditor
        print("✅ CarImageEditor imported successfully")
        
        from ai_beautifier import AIBeautifier
        print("✅ AIBeautifier imported successfully")
        
        # Test basic initialization
        print("\n🚀 Testing initialization...")
        
        editor = CarImageEditor()
        print("✅ CarImageEditor initialized")
        
        # Test basic processing
        print("\n🎨 Testing basic processing...")
        
        output_path = "output/test_simple.jpg"
        os.makedirs("output", exist_ok=True)
        
        start_time = time.time()
        
        success = editor.process_image(
            input_path=input_path,
            output_path=output_path,
            background_type="studio",
            ai_style="glossy",
            ai_level="light",  # Use light for faster processing
            logo_text="Test"
        )
        
        processing_time = time.time() - start_time
        
        if success:
            print(f"✅ Processing completed in {processing_time:.2f}s")
            print(f"📁 Output saved: {output_path}")
            
            # Check if file was actually created
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                print(f"📊 Output file size: {file_size} bytes")
                return True
            else:
                print("❌ Output file was not created")
                return False
        else:
            print("❌ Processing failed")
            return False
            
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ai_beautifier_only():
    """Test AI Beautifier in isolation"""
    print("\n🤖 Testing AI Beautifier separately...")
    
    input_path = "input/car.jpg"
    if not os.path.exists(input_path):
        print("❌ No input image for AI test")
        return False
    
    try:
        from PIL import Image
        from ai_beautifier import AIBeautifier
        
        # Load image
        car_image = Image.open(input_path)
        print(f"✅ Loaded image: {car_image.size}")
        
        # Initialize AI Beautifier
        beautifier = AIBeautifier()
        print("✅ AI Beautifier initialized")
        
        # Test beautification
        print("🎨 Testing AI beautification...")
        
        start_time = time.time()
        enhanced = beautifier.beautify_car(
            car_image,
            enhancement_level="light",
            style="glossy"
        )
        processing_time = time.time() - start_time
        
        print(f"✅ AI beautification completed in {processing_time:.2f}s")
        
        # Save result
        output_path = "output/test_ai_only.jpg"
        enhanced.save(output_path, quality=95)
        print(f"📁 AI result saved: {output_path}")
        
        # Cleanup
        beautifier.cleanup()
        print("🧹 AI Beautifier cleaned up")
        
        return True
        
    except Exception as e:
        print(f"❌ AI Beautifier test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Car Image Editor - Simple Test Suite")
    print("=" * 50)
    
    # Check basic requirements
    if not os.path.exists("src/main.py"):
        print("❌ Not in correct directory. Please run from project root.")
        return
    
    if not os.path.exists("input/car.jpg"):
        print("❌ Please add a car image as input/car.jpg")
        return
    
    # Run tests
    test1_success = test_basic_functionality()
    test2_success = test_ai_beautifier_only()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    print(f"Basic Functionality: {'✅ PASSED' if test1_success else '❌ FAILED'}")
    print(f"AI Beautifier Only: {'✅ PASSED' if test2_success else '❌ FAILED'}")
    
    if test1_success and test2_success:
        print("\n🎉 All tests passed!")
        print("📁 Check output/ directory for test results")
        print("\n💡 You can now try the full Docker version:")
        print("   ./scripts/run.sh")
    elif test1_success:
        print("\n✅ Basic functionality works!")
        print("⚠️  AI Beautifier had issues (may need GPU or models)")
        print("💡 Try running with Docker for full AI support")
    else:
        print("\n❌ Basic functionality failed")
        print("💡 Check error messages above")
        print("💡 Try running the debug script: ./scripts/debug_run.sh")

if __name__ == "__main__":
    main()
