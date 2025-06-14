import cv2
import numpy as np
from PIL import Image, ImageFilter
import torch
from diffusers import StableDiffusionInpaintPipeline

class BackgroundProcessor:
    def __init__(self, config):
        self.config = config
        self.backgrounds = {
            'showroom': 'backgrounds/showroom_1.jpg',
            'urban': 'backgrounds/urban_1.jpg',
            'studio': 'backgrounds/studio_1.jpg'
        }
        
    def get_background(self, bg_type, target_size):
        """Load and prepare background image"""
        bg_path = self.backgrounds.get(bg_type)
        if not bg_path:
            # Generate AI background if not found
            return self._generate_background(bg_type, target_size)
        
        bg = Image.open(bg_path)
        bg = bg.resize(target_size, Image.Resampling.LANCZOS)
        return bg
    
    def _generate_background(self, bg_type, size):
        """Generate background using AI if needed"""
        # Simple gradient background as fallback
        w, h = size
        bg = Image.new('RGB', (w, h))
        pixels = bg.load()
        
        # Create gradient
        for i in range(w):
            for j in range(h):
                r = int(255 * (1 - j/h) * 0.9)
                g = int(255 * (1 - j/h) * 0.9)
                b = int(255 * (1 - j/h))
                pixels[i, j] = (r, g, b)
        
        return bg
    
    def composite_images(self, foreground, background):
        """Composite car onto background with realistic shadows"""
        # Ensure same size
        background = background.resize(foreground.size, Image.Resampling.LANCZOS)
        
        # Create shadow
        shadow = self._create_shadow(foreground)
        
        # Composite shadow first
        background.paste(shadow, (0, 50), shadow)
        
        # Composite car
        background.paste(foreground, (0, 0), foreground)
        
        return background
    
    def _create_shadow(self, car_image):
        """Create realistic shadow for the car"""
        # Get alpha channel
        if car_image.mode != 'RGBA':
            return None
            
        alpha = car_image.split()[-1]
        
        # Create shadow from alpha
        shadow = Image.new('RGBA', car_image.size, (0, 0, 0, 0))
        shadow.paste((0, 0, 0, 100), mask=alpha)
        
        # Blur shadow
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=10))
        
        # Transform shadow (perspective)
        width, height = shadow.size
        shadow = shadow.transform(
            (width, height),
            Image.PERSPECTIVE,
            (0, 0, width*0.1, height*0.2, width*0.9, height*0.2, width, 0),
            Image.Resampling.BICUBIC
        )
        
        return shadow