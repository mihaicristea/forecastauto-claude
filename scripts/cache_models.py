#!/usr/bin/env python3
"""
Script pentru pre-download și cache al modelelor AI.
Rulează acest script o dată pentru a descărca toate modelele.
"""

import os
import sys
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent.parent
sys.path.append(str(current_dir / 'src'))

def main():
    print("🚀 Pre-downloading and caching AI models...")
    print("This will download ~3GB of models and cache them locally.")
    print("Future runs will be much faster!")
    
    # Set cache directories
    cache_dir = Path("/app/models/cache") if Path("/app").exists() else Path("models/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    os.environ['TRANSFORMERS_CACHE'] = str(cache_dir)
    os.environ['HF_HOME'] = str(cache_dir)
    os.environ['HUGGINGFACE_HUB_CACHE'] = str(cache_dir)
    
    print(f"📁 Cache directory: {cache_dir}")
    
    try:
        from ai_beautifier import AIBeautifier
        
        print("\n🤖 Initializing AI Beautifier (this will download models)...")
        beautifier = AIBeautifier()
        
        print("\n✅ Models downloaded and cached successfully!")
        print(f"💾 Cache location: {cache_dir}")
        print(f"📊 Cache size: {get_cache_size(cache_dir)}")
        
        # Cleanup memory
        beautifier.cleanup()
        
    except Exception as e:
        print(f"❌ Failed to cache models: {e}")
        return False
    
    return True

def get_cache_size(path):
    """Get human readable cache size"""
    total_size = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = Path(root) / file
            if file_path.exists():
                total_size += file_path.stat().st_size
    
    # Convert to human readable
    for unit in ['B', 'KB', 'MB', 'GB']:
        if total_size < 1024:
            return f"{total_size:.1f} {unit}"
        total_size /= 1024
    return f"{total_size:.1f} TB"

if __name__ == "__main__":
    main()
