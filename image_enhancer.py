import cv2
import numpy as np
from PIL import Image, ImageEnhance
import torch
from torchvision import transforms

class ImageEnhancer:
    def __init__(self, config):
        self.config = config
        
    def enhance(self, image):
        """Apply professional image enhancements"""
        # Handle RGBA images
        has_alpha = image.mode == 'RGBA'
        alpha_channel = None
        
        if has_alpha:
            # Separate alpha channel
            alpha_channel = image.split()[-1]
            image = image.convert('RGB')
        
        # Convert to numpy for OpenCV processing
        img_np = np.array(image)
        
        # 1. Denoise
        img_np = cv2.fastNlMeansDenoisingColored(img_np, None, 10, 10, 7, 21)
        
        # 2. Sharpen
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        img_np = cv2.filter2D(img_np, -1, kernel)
        
        # Convert back to PIL
        image = Image.fromarray(img_np)
        
        # 3. Adjust brightness/contrast
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.1)
        
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)
        
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(1.1)
        
        # 4. Auto white balance
        image = self._auto_white_balance(image)
        
        # Restore alpha channel if it existed
        if has_alpha and alpha_channel:
            image = image.convert('RGBA')
            image.putalpha(alpha_channel)
        
        return image
    
    def _auto_white_balance(self, image):
        """Apply automatic white balance"""
        img_np = np.array(image)
        
        # Calculate average color
        avg_r = np.mean(img_np[:,:,0])
        avg_g = np.mean(img_np[:,:,1])
        avg_b = np.mean(img_np[:,:,2])
        avg_all = (avg_r + avg_g + avg_b) / 3
        
        # Calculate scaling factors
        r_scale = avg_all / avg_r if avg_r > 0 else 1
        g_scale = avg_all / avg_g if avg_g > 0 else 1
        b_scale = avg_all / avg_b if avg_b > 0 else 1
        
        # Apply scaling
        img_np[:,:,0] = np.clip(img_np[:,:,0] * r_scale, 0, 255)
        img_np[:,:,1] = np.clip(img_np[:,:,1] * g_scale, 0, 255)
        img_np[:,:,2] = np.clip(img_np[:,:,2] * b_scale, 0, 255)
        
        return Image.fromarray(img_np.astype(np.uint8))
