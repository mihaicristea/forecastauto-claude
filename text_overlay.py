import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class TextOverlay:
    def __init__(self, config):
        self.config = config
        # Removed heavy transformer models for faster processing
        
    def add_logo(self, image, logo_text):
        """Detect license plate area and add logo text"""
        # Handle RGBA images
        has_alpha = image.mode == 'RGBA'
        alpha_channel = None
        
        if has_alpha:
            alpha_channel = image.split()[-1]
            rgb_image = image.convert('RGB')
        else:
            rgb_image = image
        
        # Convert to CV2 format
        img_cv = cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2BGR)
        
        # Detect license plate area (simplified - using bottom center area)
        height, width = img_cv.shape[:2]
        plate_area = self._detect_plate_area(img_cv)
        
        if plate_area is None:
            # Default position if no plate detected
            plate_area = (width//2 - 100, height - 150, 200, 60)
        
        # Remove plate area content
        x, y, w, h = plate_area
        mask = np.zeros(img_cv.shape[:2], dtype=np.uint8)
        mask[y:y+h, x:x+w] = 255
        img_cv = cv2.inpaint(img_cv, mask, 3, cv2.INPAINT_TELEA)
        
        # Convert back to PIL
        processed_image = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        
        # Restore alpha channel if it existed
        if has_alpha and alpha_channel:
            processed_image = processed_image.convert('RGBA')
            processed_image.putalpha(alpha_channel)
        
        # Add logo text
        draw = ImageDraw.Draw(processed_image)
        
        # Try to load custom font, fallback to default
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 40)
            except:
                font = ImageFont.load_default()
        
        # Calculate text position
        text_bbox = draw.textbbox((0, 0), logo_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        text_x = x + (w - text_width) // 2
        text_y = y + (h - text_height) // 2
        
        # Create professional logo styling
        # Background rectangle for logo
        rect_padding = 10
        rect_x1 = text_x - rect_padding
        rect_y1 = text_y - rect_padding
        rect_x2 = text_x + text_width + rect_padding
        rect_y2 = text_y + text_height + rect_padding
        
        # Draw subtle background rectangle
        draw.rectangle([rect_x1, rect_y1, rect_x2, rect_y2], 
                      fill=(0, 0, 0, 100), outline=(255, 255, 255, 150))
        
        # Add text with subtle effects
        # Shadow
        draw.text((text_x+1, text_y+1), logo_text, font=font, fill=(0, 0, 0, 180))
        # Main text
        draw.text((text_x, text_y), logo_text, font=font, fill=(255, 255, 255, 255))
        
        return processed_image
    
    def _detect_plate_area(self, img_cv):
        """Detect license plate area using edge detection"""
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter for plate-like shapes
        height, width = img_cv.shape[:2]
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if contour is plate-like
            aspect_ratio = w / h
            if 2 < aspect_ratio < 5 and 50 < w < 300 and y > height * 0.6:
                return (x, y, w, h)
        
        return None
