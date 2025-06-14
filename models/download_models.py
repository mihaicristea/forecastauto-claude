#!/usr/bin/env python3
import os
import requests
from tqdm import tqdm
from pathlib import Path

def download_file(url, dest):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest, 'wb') as file:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=dest.name) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
                pbar.update(len(chunk))

def main():
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    # Download SAM model if needed
    sam_path = models_dir / 'sam_vit_h_4b8939.pth'
    if not sam_path.exists():
        print("Downloading SAM model...")
        download_file(
            'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
            sam_path
        )
    
    print("✅ All models downloaded")

if __name__ == "__main__":
    main()