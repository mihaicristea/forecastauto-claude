#!/usr/bin/env python3
"""
Script to pre-cache rembg models to avoid downloading at runtime
"""
import os
import sys

# Add src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
src_dir = os.path.join(project_dir, 'src')
sys.path.insert(0, src_dir)

def cache_rembg_models():
    """Pre-download and cache rembg models"""
    print("🔧 Pre-caching rembg models...")
    
    try:
        # Set cache directory
        cache_dir = os.path.join(project_dir, 'models', 'cache', 'rembg')
        os.makedirs(cache_dir, exist_ok=True)
        
        # Set environment variable for rembg
        os.environ['U2NET_HOME'] = cache_dir
        
        # Import rembg to trigger model download
        import rembg
        from rembg import remove, new_session
        
        # Create a session to download the u2net model
        print("   📥 Downloading u2net model...")
        session = new_session('u2net')
        
        print(f"   ✅ Rembg models cached to: {cache_dir}")
        return True
        
    except Exception as e:
        print(f"   ⚠️ Failed to cache rembg models: {e}")
        return False

if __name__ == "__main__":
    success = cache_rembg_models()
    if success:
        print("🎉 Rembg model caching completed successfully!")
    else:
        print("❌ Rembg model caching failed!")
        sys.exit(1)
