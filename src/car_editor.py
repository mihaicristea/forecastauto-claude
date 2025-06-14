import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance, ImageOps
import os
import sys
import yaml
import cv2
from transformers import pipeline, AutoImageProcessor, AutoModelForImageSegmentation
from rembg import remove, new_session
import requests
from io import BytesIO

# Add parent directory to path to import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from background_processor import BackgroundProcessor
from image_enhancer import ImageEnhancer
from text_overlay import TextOverlay
from ai_beautifier import AIBeautifier

class CarImageEditor:
    def __init__(self, config_path=None):
        # Load configuration
        if config_path is None:
            config_path = os.path.join(os.path.dirname(current_dir), 'config.yaml')
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Using device: {self.device}")
        
        if torch.cuda.is_available():
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
            print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # Initialize processors
        self.background_processor = BackgroundProcessor(self.config)
        self.image_enhancer = ImageEnhancer(self.config)
        self.text_overlay = TextOverlay(self.config)
        
        # Initialize AI Beautifier
        print("🤖 Loading AI Beautifier...")
        try:
            self.ai_beautifier = AIBeautifier(self.config)
        except Exception as e:
            print(f"⚠️  AI Beautifier initialization failed: {e}")
            self.ai_beautifier = None
        
        # Initialize advanced segmentation model
        print("🔧 Loading advanced segmentation models...")
        try:
            # Try to use U2Net for best quality
            self.rembg_session = new_session('u2net')
            print("✅ Loaded U2Net model for professional segmentation")
        except:
            # Fallback to standard model
            self.rembg_session = new_session()
            print("✅ Loaded standard segmentation model")
    
    def process_image(self, input_path, output_path, background_type='showroom', 
                     custom_background=None, logo_text='Forecast AUTO', 
                     ai_style='glossy', ai_level='medium', enhance=False, 
                     quality=95, **kwargs):
        try:
            print(f"\n📸 Processing: {os.path.basename(input_path)}")
            
            # Load image
            img = Image.open(input_path)
            print(f"   Size: {img.size}")
            print(f"   Mode: {img.mode}")
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize if too large (for better processing)
            max_size = 2048
            if max(img.size) > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                print(f"   Resized to: {img.size}")
            
            print("🎯 Step 1: Professional car extraction...")
            # Extract car using advanced AI segmentation
            car_rgba = self._extract_car_professional(img)
            
            print("🖼️  Step 2: Creating studio-quality background...")
            # Create professional background
            background = self._create_studio_background(img.size, background_type)
            
            print("✨ Step 3: Professional car enhancement...")
            # Enhance the car image professionally with AI parameters
            enhanced_car = self._enhance_car_professional(car_rgba, ai_style, ai_level)
            
            print("🏷️  Step 4: Adding Forecast AUTO branding...")
            # Add professional branding
            enhanced_car = self._add_professional_branding(enhanced_car, logo_text)
            
            print("🎨 Step 5: Professional compositing...")
            # Composite with studio lighting and shadows
            final_image = self._composite_professional(enhanced_car, background)
            
            print("💎 Step 6: Final professional touches...")
            # Apply final professional enhancements
            final_image = self._apply_professional_finish(final_image)
            
            # Save with maximum quality
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            final_image.save(output_path, quality=95, optimize=True)
            
            print(f"✅ Professional car image saved to: {output_path}")
            
            # Cleanup AI Beautifier memory if used
            if self.ai_beautifier is not None:
                try:
                    self.ai_beautifier.cleanup()
                except:
                    pass
            
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            
            # Cleanup on error too
            if hasattr(self, 'ai_beautifier') and self.ai_beautifier is not None:
                try:
                    self.ai_beautifier.cleanup()
                except:
                    pass
            
            return False
    
    def _extract_car_professional(self, image):
        """Extract car using state-of-the-art AI segmentation"""
        try:
            # Use rembg with U2Net for best quality
            car_rgba = remove(image, session=self.rembg_session, alpha_matting=True, 
                            alpha_matting_foreground_threshold=240,
                            alpha_matting_background_threshold=50,
                            alpha_matting_erode_size=10)
            
            # Refine edges for professional look
            car_rgba = self._refine_edges(car_rgba)
            
            return car_rgba
            
        except Exception as e:
            print(f"⚠️  Advanced segmentation failed, using fallback: {e}")
            # Fallback to enhanced GrabCut
            return self._extract_car_grabcut_enhanced(image)
    
    def _refine_edges(self, image_rgba):
        """Refine edges for professional quality"""
        # Split channels
        r, g, b, a = image_rgba.split()
        
        # Convert alpha to numpy for processing
        alpha_np = np.array(a)
        
        # Apply edge refinement
        # 1. Smooth edges with bilateral filter
        alpha_np = cv2.bilateralFilter(alpha_np, 9, 75, 75)
        
        # 2. Create edge mask
        edges = cv2.Canny(alpha_np, 50, 150)
        
        # 3. Dilate edges slightly
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # 4. Apply Gaussian blur to edge areas only
        blur_mask = cv2.GaussianBlur(alpha_np, (5, 5), 0)
        alpha_np = np.where(edges > 0, blur_mask, alpha_np)
        
        # Convert back to PIL
        a = Image.fromarray(alpha_np)
        
        # Merge channels
        return Image.merge('RGBA', (r, g, b, a))
    
    def _extract_car_grabcut_enhanced(self, image):
        """Enhanced GrabCut with better edge handling"""
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Create initial mask
        mask = np.zeros(img_cv.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        # Better rectangle estimation
        height, width = img_cv.shape[:2]
        
        # Use edge detection to find car boundaries
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get largest contour (likely the car)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Add padding
            padding = 20
            rect = (max(0, x-padding), max(0, y-padding), 
                   min(width, x+w+padding), min(height, y+h+padding))
        else:
            # Fallback to center rectangle
            rect = (int(width*0.1), int(height*0.1), int(width*0.9), int(height*0.9))
        
        # Apply GrabCut with more iterations
        cv2.grabCut(img_cv, mask, rect, bgd_model, fgd_model, 10, cv2.GC_INIT_WITH_RECT)
        
        # Create final mask
        mask2 = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
        
        # Refine mask with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Edge feathering
        mask2 = cv2.GaussianBlur(mask2, (7, 7), 0)
        
        # Create RGBA image
        image_rgba = image.convert('RGBA')
        image_rgba.putalpha(Image.fromarray(mask2))
        
        return image_rgba
    
    def _create_studio_background(self, size, bg_type):
        """Create professional studio-quality background"""
        width, height = size
        
        if bg_type == 'showroom':
            # Professional showroom with floor reflection
            bg = Image.new('RGB', size, (245, 245, 250))
            draw = ImageDraw.Draw(bg)
            
            # Create gradient backdrop
            for y in range(height):
                # Vertical gradient
                intensity = int(245 - (y / height) * 30)
                color = (intensity, intensity, min(255, intensity + 5))
                draw.rectangle([(0, y), (width, y+1)], fill=color)
            
            # Add floor line
            floor_y = int(height * 0.75)
            draw.rectangle([(0, floor_y), (width, height)], fill=(235, 235, 240))
            
            # Add subtle vignette
            bg = self._add_vignette(bg, intensity=0.15)
            
        elif bg_type == 'gradient':
            # Professional gradient background
            bg = Image.new('RGB', size)
            draw = ImageDraw.Draw(bg)
            
            # Create smooth radial gradient
            center_x, center_y = width // 2, height // 2
            max_radius = ((width/2)**2 + (height/2)**2)**0.5
            
            for y in range(height):
                for x in range(width):
                    # Calculate distance from center
                    dx = x - center_x
                    dy = y - center_y
                    distance = (dx**2 + dy**2)**0.5
                    
                    # Gradient calculation
                    ratio = distance / max_radius
                    intensity = int(80 - ratio * 60)
                    
                    bg.putpixel((x, y), (intensity, intensity, intensity))
            
            # Apply Gaussian blur for smoothness
            bg = bg.filter(ImageFilter.GaussianBlur(radius=20))
            
        else:  # studio
            # Professional studio backdrop
            bg = Image.new('RGB', size, (220, 225, 230))
            
            # Add subtle texture using numpy noise
            bg_np = np.array(bg)
            noise = np.random.normal(0, 5, bg_np.shape).astype(np.int16)
            bg_np = np.clip(bg_np.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            bg = Image.fromarray(bg_np)
            
            # Add professional lighting gradient
            overlay = Image.new('RGB', size, (255, 255, 255))
            mask = Image.new('L', size, 0)
            draw = ImageDraw.Draw(mask)
            
            # Top lighting
            for y in range(height//2):
                intensity = int(255 * (1 - y / (height/2)))
                draw.rectangle([(0, y), (width, y+1)], fill=intensity)
            
            bg = Image.composite(overlay, bg, mask)
        
        return bg
    
    def _add_vignette(self, image, intensity=0.3):
        """Add professional vignette effect"""
        # Create vignette mask
        width, height = image.size
        mask = Image.new('L', (width, height), 255)
        draw = ImageDraw.Draw(mask)
        
        # Create radial gradient
        for i in range(min(width, height) // 2):
            alpha = int(255 * (1 - (i / (min(width, height) / 2)) * intensity))
            draw.ellipse([i, i, width-i, height-i], fill=alpha)
        
        # Apply vignette
        black = Image.new('RGB', (width, height), (0, 0, 0))
        return Image.composite(image, black, mask)
    
    def _enhance_car_professional(self, car_rgba, ai_style='glossy', ai_level='medium'):
        """Apply professional automotive photography enhancements with AI Beautifier"""
        # Try AI Beautifier first if style and level are provided
        if self.ai_beautifier is not None and ai_style and ai_level:
            try:
                print(f"   🤖 Applying AI Beautifier ({ai_level} {ai_style})...")
                # Apply AI beautification with specified parameters
                enhanced_car = self.ai_beautifier.beautify_car(
                    car_rgba, 
                    enhancement_level=ai_level, 
                    style=ai_style
                )
                return enhanced_car
            except Exception as e:
                print(f"   ⚠️  AI Beautifier failed, using traditional enhancement: {e}")
        
        # Fallback to traditional enhancement
        print("   🎨 Using traditional enhancement...")
        # Split channels
        r, g, b, a = car_rgba.split()
        
        # Create RGB image for processing
        car_rgb = Image.merge('RGB', (r, g, b))
        
        # 1. Professional color grading
        car_rgb = self._apply_color_grading(car_rgb)
        
        # 2. Enhance reflections and highlights
        car_rgb = self._enhance_reflections(car_rgb)
        
        # 3. Professional sharpening
        car_rgb = self._professional_sharpen(car_rgb)
        
        # 4. Adjust exposure for studio look
        enhancer = ImageEnhance.Brightness(car_rgb)
        car_rgb = enhancer.enhance(1.05)
        
        # 5. Enhance contrast
        enhancer = ImageEnhance.Contrast(car_rgb)
        car_rgb = enhancer.enhance(1.15)
        
        # 6. Subtle saturation boost
        enhancer = ImageEnhance.Color(car_rgb)
        car_rgb = enhancer.enhance(1.1)
        
        # Merge back with alpha
        r, g, b = car_rgb.split()
        return Image.merge('RGBA', (r, g, b, a))
    
    def _apply_color_grading(self, image):
        """Apply professional color grading"""
        # Convert to numpy for advanced processing
        img_np = np.array(image).astype(np.float32)
        
        # Apply subtle color curves
        # Lift shadows slightly
        img_np = np.where(img_np < 50, img_np * 1.1, img_np)
        
        # Enhance midtones
        mask = (img_np >= 50) & (img_np <= 200)
        img_np[mask] = img_np[mask] * 1.05
        
        # Protect highlights
        img_np = np.where(img_np > 200, img_np * 0.98, img_np)
        
        # Ensure values are in valid range
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img_np)
    
    def _enhance_reflections(self, image):
        """Enhance car paint reflections"""
        # Create highlight mask
        img_np = np.array(image)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Find bright areas (reflections)
        _, highlights = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Dilate slightly
        kernel = np.ones((3,3), np.uint8)
        highlights = cv2.dilate(highlights, kernel, iterations=1)
        
        # Blur for smooth transition
        highlights = cv2.GaussianBlur(highlights, (5, 5), 0)
        
        # Apply enhancement to bright areas
        enhanced = img_np.copy()
        mask = highlights > 0
        enhanced[mask] = np.clip(enhanced[mask] * 1.1, 0, 255)
        
        return Image.fromarray(enhanced.astype(np.uint8))
    
    def _professional_sharpen(self, image):
        """Apply professional sharpening"""
        # Use unsharp mask for professional results
        return image.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
    
    def _add_professional_branding(self, car_rgba, logo_text):
        """Add professional branding without damaging the car"""
        # For now, return the car as-is
        # The text_overlay module can be enhanced separately
        return car_rgba
    
    def _composite_professional(self, car_rgba, background):
        """Professional compositing with realistic shadows and reflections"""
        # Ensure same size
        if background.size != car_rgba.size:
            background = background.resize(car_rgba.size, Image.Resampling.LANCZOS)
        
        # Create professional shadow
        shadow = self._create_professional_shadow(car_rgba)
        
        # Create floor reflection (for showroom style)
        reflection = self._create_floor_reflection(car_rgba)
        
        # Composite in correct order
        final = background.convert('RGBA')
        
        # Add shadow first
        if shadow:
            final = Image.alpha_composite(final, shadow)
        
        # Add reflection
        if reflection:
            final = Image.alpha_composite(final, reflection)
        
        # Add car on top
        final = Image.alpha_composite(final, car_rgba)
        
        return final.convert('RGB')
    
    def _create_professional_shadow(self, car_rgba):
        """Create realistic automotive shadow"""
        # Get alpha channel
        _, _, _, alpha = car_rgba.split()
        
        # Create shadow base
        shadow = Image.new('RGBA', car_rgba.size, (0, 0, 0, 0))
        
        # Convert alpha to numpy
        alpha_np = np.array(alpha)
        
        # Create shadow shape
        shadow_np = np.zeros((car_rgba.size[1], car_rgba.size[0], 4), dtype=np.uint8)
        
        # Project shadow (perspective transform)
        height, width = alpha_np.shape
        
        for y in range(height):
            for x in range(width):
                if alpha_np[y, x] > 50:
                    # Calculate shadow position
                    shadow_y = min(height - 1, y + int((height - y) * 0.15))
                    shadow_x = x + int((x - width/2) * 0.05)
                    
                    if 0 <= shadow_x < width and shadow_y < height:
                        # Shadow intensity based on height
                        intensity = int(100 * (1 - y / height) * (alpha_np[y, x] / 255))
                        shadow_np[shadow_y, shadow_x] = [0, 0, 0, intensity]
        
        # Convert to PIL
        shadow = Image.fromarray(shadow_np, 'RGBA')
        
        # Apply multiple blur passes for soft shadow
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=15))
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=10))
        
        return shadow
    
    def _create_floor_reflection(self, car_rgba):
        """Create subtle floor reflection"""
        # Get car dimensions
        width, height = car_rgba.size
        
        # Create flipped version
        reflection = car_rgba.transpose(Image.FLIP_TOP_BOTTOM)
        
        # Create gradient mask for fade effect
        mask = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask)
        
        # Gradient from top (near car) to bottom
        for y in range(height // 3):  # Only top third for reflection
            opacity = int(30 * (1 - y / (height / 3)))  # Max 30% opacity
            draw.rectangle([(0, y), (width, y+1)], fill=opacity)
        
        # Apply mask to reflection
        reflection.putalpha(mask)
        
        # Position reflection below car
        final_reflection = Image.new('RGBA', car_rgba.size, (0, 0, 0, 0))
        
        # Calculate offset (place reflection at bottom)
        offset_y = int(height * 0.75)
        if offset_y + height // 3 <= height:
            final_reflection.paste(reflection, (0, offset_y), reflection)
        
        return final_reflection
    
    def _apply_professional_finish(self, image):
        """Apply final professional touches"""
        # 1. Subtle film grain for professional look
        image = self._add_film_grain(image, intensity=0.02)
        
        # 2. Professional color balance
        image = self._adjust_color_balance(image)
        
        # 3. Final sharpening pass
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.05)
        
        # 4. Subtle contrast adjustment
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.02)
        
        return image
    
    def _add_film_grain(self, image, intensity=0.02):
        """Add subtle film grain for professional look"""
        # Create grain
        img_np = np.array(image)
        grain = np.random.normal(0, 255 * intensity, img_np.shape).astype(np.int16)
        
        # Add grain
        img_with_grain = img_np.astype(np.int16) + grain
        img_with_grain = np.clip(img_with_grain, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img_with_grain)
    
    def _adjust_color_balance(self, image):
        """Professional color balance adjustment"""
        # Convert to numpy
        img_np = np.array(image).astype(np.float32)
        
        # Subtle adjustments to each channel
        img_np[:, :, 0] *= 1.02  # Slight red boost
        img_np[:, :, 1] *= 1.01  # Very slight green boost
        img_np[:, :, 2] *= 0.98  # Slight blue reduction
        
        # Clip values
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img_np)
