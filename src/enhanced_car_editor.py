import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance, ImageOps
import os
import sys
import yaml
import cv2
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

class EnhancedCarImageEditor:
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
        
        # Initialize advanced segmentation model
        print("🔧 Loading rembg model for perfect background removal...")
        try:
            # Use the simple but effective rembg approach like in your original script
            self.rembg_session = new_session('u2net')
            print("✅ Loaded U2Net model for perfect segmentation")
        except:
            # Fallback to standard model
            self.rembg_session = new_session()
            print("✅ Loaded standard segmentation model")
    
    def process_image(self, input_path, output_path, background_type='showroom', logo_text='Forecast AUTO', use_logo_plate=True, **kwargs):
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
            
            print("🎯 Step 1: Perfect car extraction using rembg...")
            # Extract car using the simple but perfect rembg approach (like your original script)
            car_rgba = self._extract_car_perfect(img)
            
            print("🏷️  Step 2: Smart license plate replacement...")
            # Replace license plate with logo (like your original script)
            if use_logo_plate:
                car_rgba = self._replace_license_plate_smart(car_rgba, logo_text)
            
            print("✨ Step 3: Professional car enhancement...")
            # Enhance the car image professionally
            enhanced_car = self._enhance_car_professional(car_rgba)
            
            print("🖼️  Step 4: Creating studio-quality background...")
            # Create professional background
            background = self._create_studio_background(img.size, background_type)
            
            print("🎨 Step 5: Perfect color matching and integration...")
            # Color match car with background (like your original script)
            enhanced_car = self._color_match_with_background(enhanced_car, background)
            
            print("🌟 Step 6: Professional compositing with shadows...")
            # Composite with realistic shadows (combining both approaches)
            final_image = self._composite_with_perfect_shadows(enhanced_car, background)
            
            print("💎 Step 7: Final professional touches...")
            # Apply final professional enhancements
            final_image = self._apply_professional_finish(final_image)
            
            # Save with maximum quality
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            final_image.save(output_path, quality=95, optimize=True)
            
            print(f"✅ Enhanced professional car image saved to: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _extract_car_perfect(self, image):
        """Extract car using the simple but perfect rembg approach (like your original script)"""
        try:
            # Use simple rembg remove - this is what worked perfectly in your original script
            car_rgba = remove(image, session=self.rembg_session)
            
            # Simple edge improvement for natural look
            car_rgba = self._improve_edges_naturally(car_rgba)
            
            return car_rgba
            
        except Exception as e:
            print(f"⚠️  rembg extraction failed: {e}")
            return self._fallback_extraction(image)
    
    def _improve_edges_naturally(self, image_rgba):
        """Improve edges naturally without over-processing"""
        # Split channels
        r, g, b, a = image_rgba.split()
        
        # Convert alpha to numpy for light processing
        alpha_np = np.array(a)
        
        # Very light edge smoothing only
        alpha_np = cv2.GaussianBlur(alpha_np, (3, 3), 0.5)
        
        # Convert back to PIL
        a = Image.fromarray(alpha_np)
        
        # Merge channels
        return Image.merge('RGBA', (r, g, b, a))
    
    def _replace_license_plate_smart(self, car_rgba, logo_text):
        """Smart license plate detection and replacement (from your original script)"""
        try:
            # Convert to format compatible with detection
            car_rgb = car_rgba.convert('RGB')
            
            # Try multiple detection methods like in your original script
            plate_coords = self._detect_license_plate_enhanced(car_rgb)
            
            if plate_coords is None:
                plate_coords = self._detect_license_plate_simple(car_rgb)
            
            if plate_coords is None:
                # Fallback to estimated position
                plate_coords = self._estimate_plate_position(car_rgb.size)
            
            if plate_coords:
                return self._replace_plate_with_logo(car_rgba, plate_coords, logo_text)
            
            return car_rgba
            
        except Exception as e:
            print(f"⚠️  License plate replacement failed: {e}")
            return car_rgba
    
    def _detect_license_plate_enhanced(self, car_rgb):
        """Enhanced license plate detection (from your original script)"""
        try:
            car_cv = cv2.cvtColor(np.array(car_rgb), cv2.COLOR_RGB2BGR)
            height, width = car_cv.shape[:2]
            
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(car_cv, cv2.COLOR_BGR2HSV)
            
            # Mask for white/light colors (license plates)
            lower_white = np.array([0, 0, 140])
            upper_white = np.array([180, 60, 255])
            mask_white = cv2.inRange(hsv, lower_white, upper_white)
            
            # Mask for yellow (EU plates)
            lower_yellow = np.array([15, 60, 60])
            upper_yellow = np.array([40, 255, 255])
            mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
            
            # Combine color masks
            color_mask = cv2.bitwise_or(mask_white, mask_yellow)
            
            # Morphological operations to connect characters
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 3))
            morphed = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
            morphed = cv2.dilate(morphed, None, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            candidates = []
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h) if h > 0 else 0
                area = cv2.contourArea(contour)
                
                # Filter potential license plates
                if (2.0 <= aspect_ratio <= 6.0 and
                    area > 200 and
                    w > 40 and h > 10 and
                    w < width * 0.5 and h < height * 0.15 and
                    y > height * 0.5):
                    
                    # Calculate position scores
                    center_x = x + w/2
                    horizontal_score = 1.0 - abs(center_x - width/2) / (width/2)
                    vertical_score = (y + h/2) / height
                    aspect_score = 1.0 / (1.0 + abs(aspect_ratio - 4.5))
                    
                    total_score = (horizontal_score * 0.4 + 
                                 vertical_score * 0.3 + 
                                 aspect_score * 0.3)
                    
                    candidates.append((x, y, w, h, total_score))
            
            if candidates:
                # Sort by score and return best
                candidates.sort(key=lambda x: x[4], reverse=True)
                best = candidates[0]
                if best[4] > 0.5:
                    return best[:4]
            
            return None
            
        except Exception as e:
            print(f"Enhanced detection failed: {e}")
            return None
    
    def _detect_license_plate_simple(self, car_rgb):
        """Simple license plate detection fallback"""
        try:
            car_cv = cv2.cvtColor(np.array(car_rgb), cv2.COLOR_RGB2BGR)
            height, width = car_cv.shape[:2]
            gray = cv2.cvtColor(car_cv, cv2.COLOR_BGR2GRAY)
            
            # Threshold for white areas
            _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            candidates = []
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h) if h > 0 else 0
                area = cv2.contourArea(contour)
                
                if (3.0 <= aspect_ratio <= 8.0 and
                    area > 200 and
                    w > 60 and h > 12 and
                    w < width * 0.4 and h < height * 0.1 and
                    y > height * 0.6):
                    
                    center_x = x + w/2
                    horizontal_score = 1.0 - abs(center_x - width/2) / (width/2)
                    vertical_score = (y + h/2) / height
                    aspect_score = 1.0 / (1.0 + abs(aspect_ratio - 5.0))
                    
                    total_score = (horizontal_score * 0.4 + 
                                 vertical_score * 0.3 + 
                                 aspect_score * 0.3)
                    
                    candidates.append((x, y, w, h, total_score))
            
            if candidates:
                candidates.sort(key=lambda x: x[4], reverse=True)
                best = candidates[0]
                if best[4] > 0.4:
                    return best[:4]
            
            return None
            
        except Exception as e:
            print(f"Simple detection failed: {e}")
            return None
    
    def _estimate_plate_position(self, size):
        """Estimate license plate position as fallback"""
        width, height = size
        
        # Estimate typical position
        plate_width = int(width * 0.20)
        plate_height = int(height * 0.06)
        plate_x = int(width * 0.40)
        plate_y = int(height * 0.80)
        
        return (plate_x, plate_y, plate_width, plate_height)
    
    def _replace_plate_with_logo(self, car_rgba, plate_coords, logo_text):
        """Replace license plate area with logo text"""
        try:
            plate_x, plate_y, plate_w, plate_h = plate_coords
            
            # Add some padding
            padding = int(plate_h * 0.10)
            plate_x = max(0, plate_x - padding)
            plate_y = max(0, plate_y - padding)
            plate_w = min(car_rgba.width - plate_x, plate_w + 2 * padding)
            plate_h = min(car_rgba.height - plate_y, plate_h + 2 * padding)
            
            # Create white rectangle with black border
            car_with_plate = car_rgba.copy()
            draw = ImageDraw.Draw(car_with_plate)
            
            # Draw white rectangle
            draw.rectangle([plate_x, plate_y, plate_x + plate_w, plate_y + plate_h], 
                         fill=(255, 255, 255, 255), outline=(0, 0, 0, 255), width=2)
            
            # Add logo text
            try:
                # Calculate font size to fit
                font_size = int(plate_h * 0.5)
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
            except:
                font = ImageFont.load_default()
            
            # Calculate text position for centering
            bbox = draw.textbbox((0, 0), logo_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            text_x = plate_x + (plate_w - text_width) // 2
            text_y = plate_y + (plate_h - text_height) // 2
            
            # Draw text
            draw.text((text_x, text_y), logo_text, fill=(0, 0, 0, 255), font=font)
            
            return car_with_plate
            
        except Exception as e:
            print(f"⚠️  Logo replacement failed: {e}")
            return car_rgba
    
    def _enhance_car_professional(self, car_rgba):
        """Professional car enhancement (from current script but optimized)"""
        # Split channels
        r, g, b, a = car_rgba.split()
        
        # Create RGB image for processing
        car_rgb = Image.merge('RGB', (r, g, b))
        
        # Professional enhancements
        car_rgb = self._apply_color_grading(car_rgb)
        car_rgb = self._enhance_reflections(car_rgb)
        car_rgb = self._professional_sharpen(car_rgb)
        
        # Brightness and contrast
        enhancer = ImageEnhance.Brightness(car_rgb)
        car_rgb = enhancer.enhance(1.05)
        
        enhancer = ImageEnhance.Contrast(car_rgb)
        car_rgb = enhancer.enhance(1.15)
        
        # Color saturation
        enhancer = ImageEnhance.Color(car_rgb)
        car_rgb = enhancer.enhance(1.1)
        
        # Merge back with alpha
        r, g, b = car_rgb.split()
        return Image.merge('RGBA', (r, g, b, a))
    
    def _color_match_with_background(self, car_rgba, background):
        """Color match car with background (from your original script)"""
        try:
            # Calculate background average color
            bg_array = np.array(background.convert("RGB"))
            avg_color = np.mean(bg_array, axis=(0, 1))
            
            # Adjust car colors
            car_array = np.array(car_rgba.convert("RGBA"))
            
            # Subtle adjustment factor
            adjustment_factor = 0.08  # Reduced for more natural look
            
            for i in range(3):  # RGB channels
                car_array[:, :, i] = car_array[:, :, i] * (1 - adjustment_factor) + avg_color[i] * adjustment_factor
            
            # Convert back to image
            car_adjusted = Image.fromarray(car_array.astype(np.uint8), "RGBA")
            
            # Blend edges naturally
            return self._blend_with_background_naturally(car_adjusted)
            
        except Exception as e:
            print(f"⚠️  Color matching failed: {e}")
            return car_rgba
    
    def _blend_with_background_naturally(self, car_rgba):
        """Natural edge blending (from your original script)"""
        try:
            # Light blur on alpha channel for natural edges
            alpha = car_rgba.split()[-1]
            alpha_blurred = alpha.filter(ImageFilter.GaussianBlur(radius=0.8))
            
            # Reconstruct image with blended edges
            car_channels = car_rgba.split()[:3]
            return Image.merge("RGBA", car_channels + (alpha_blurred,))
            
        except Exception as e:
            print(f"⚠️  Edge blending failed: {e}")
            return car_rgba
    
    def _composite_with_perfect_shadows(self, car_rgba, background):
        """Perfect shadow compositing (combining both approaches)"""
        try:
            # Ensure same size
            if background.size != car_rgba.size:
                background = background.resize(car_rgba.size, Image.Resampling.LANCZOS)
            
            # Create realistic shadow (from your original script approach)
            shadow = self._create_realistic_shadow(car_rgba)
            
            # Convert background to RGBA for compositing
            final = background.convert('RGBA')
            
            # Add shadow first
            if shadow:
                final = Image.alpha_composite(final, shadow)
            
            # Add car on top
            final = Image.alpha_composite(final, car_rgba)
            
            return final.convert('RGB')
            
        except Exception as e:
            print(f"⚠️  Shadow compositing failed: {e}")
            # Fallback to simple composite
            background_rgba = background.convert('RGBA')
            final = Image.alpha_composite(background_rgba, car_rgba)
            return final.convert('RGB')
    
    def _create_realistic_shadow(self, car_rgba):
        """Create realistic shadow (from your original script approach)"""
        try:
            width, height = car_rgba.size
            
            # Create shadow base
            shadow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(shadow)
            
            # Calculate shadow area (bottom area)
            shadow_y = int(height * 0.85)
            shadow_height = int(height * 0.15)
            
            # Create gradient shadow
            for i in range(shadow_height):
                alpha = int(40 * (1 - i / shadow_height))  # Shadow that fades
                y = shadow_y + i
                if y < height:
                    draw.line([(0, y), (width, y)], fill=(0, 0, 0, alpha))
            
            # Blur shadow for natural effect
            shadow = shadow.filter(ImageFilter.GaussianBlur(radius=6))
            
            return shadow
            
        except Exception as e:
            print(f"⚠️  Shadow creation failed: {e}")
            return None
    
    # Professional enhancement methods from current script
    def _apply_color_grading(self, image):
        """Apply professional color grading"""
        img_np = np.array(image).astype(np.float32)
        
        # Apply subtle color curves
        img_np = np.where(img_np < 50, img_np * 1.05, img_np)
        mask = (img_np >= 50) & (img_np <= 200)
        img_np[mask] = img_np[mask] * 1.03
        img_np = np.where(img_np > 200, img_np * 0.99, img_np)
        
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        return Image.fromarray(img_np)
    
    def _enhance_reflections(self, image):
        """Enhance car paint reflections"""
        img_np = np.array(image)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Find bright areas (reflections)
        _, highlights = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3,3), np.uint8)
        highlights = cv2.dilate(highlights, kernel, iterations=1)
        highlights = cv2.GaussianBlur(highlights, (5, 5), 0)
        
        # Apply enhancement to bright areas
        enhanced = img_np.copy()
        mask = highlights > 0
        enhanced[mask] = np.clip(enhanced[mask] * 1.08, 0, 255)
        
        return Image.fromarray(enhanced.astype(np.uint8))
    
    def _professional_sharpen(self, image):
        """Apply professional sharpening"""
        return image.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
    
    def _create_studio_background(self, size, bg_type):
        """Create professional studio background"""
        width, height = size
        
        if bg_type == 'showroom':
            # Professional showroom with floor reflection
            bg = Image.new('RGB', size, (245, 245, 250))
            draw = ImageDraw.Draw(bg)
            
            # Create gradient backdrop
            for y in range(height):
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
            
            center_x, center_y = width // 2, height // 2
            max_radius = ((width/2)**2 + (height/2)**2)**0.5
            
            for y in range(height):
                for x in range(width):
                    dx = x - center_x
                    dy = y - center_y
                    distance = (dx**2 + dy**2)**0.5
                    ratio = distance / max_radius
                    intensity = int(80 - ratio * 60)
                    bg.putpixel((x, y), (intensity, intensity, intensity))
            
            bg = bg.filter(ImageFilter.GaussianBlur(radius=20))
            
        else:  # studio
            # Professional studio backdrop
            bg = Image.new('RGB', size, (220, 225, 230))
            
            # Add subtle texture
            bg_np = np.array(bg)
            noise = np.random.normal(0, 5, bg_np.shape).astype(np.int16)
            bg_np = np.clip(bg_np.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            bg = Image.fromarray(bg_np)
            
            # Add lighting gradient
            overlay = Image.new('RGB', size, (255, 255, 255))
            mask = Image.new('L', size, 0)
            draw = ImageDraw.Draw(mask)
            
            for y in range(height//2):
                intensity = int(255 * (1 - y / (height/2)))
                draw.rectangle([(0, y), (width, y+1)], fill=intensity)
            
            bg = Image.composite(overlay, bg, mask)
        
        return bg
    
    def _add_vignette(self, image, intensity=0.3):
        """Add professional vignette effect"""
        width, height = image.size
        
        # Create vignette mask
        vignette = Image.new('L', (width, height), 255)
        draw = ImageDraw.Draw(vignette)
        
        center_x, center_y = width // 2, height // 2
        max_radius = ((width/2)**2 + (height/2)**2)**0.5
        
        for y in range(height):
            for x in range(width):
                dx = x - center_x
                dy = y - center_y
                distance = (dx**2 + dy**2)**0.5
                
                ratio = distance / max_radius
                fade = int(255 * (1 - ratio * intensity))
                fade = max(0, min(255, fade))
                
                vignette.putpixel((x, y), fade)
        
        # Apply vignette
        return Image.composite(image, Image.new('RGB', image.size, (0, 0, 0)), vignette)
    
    def _apply_professional_finish(self, image):
        """Apply final professional touches"""
        # Subtle film grain
        image = self._add_film_grain(image, intensity=0.015)
        
        # Professional color balance
        image = self._adjust_color_balance(image)
        
        # Final sharpening
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.03)
        
        # Subtle contrast adjustment
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.01)
        
        return image
    
    def _add_film_grain(self, image, intensity=0.02):
        """Add subtle film grain"""
        img_np = np.array(image).astype(np.float32)
        noise = np.random.normal(0, intensity * 255, img_np.shape)
        img_np = img_np + noise
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        return Image.fromarray(img_np)
    
    def _adjust_color_balance(self, image):
        """Professional color balance adjustment"""
        img_np = np.array(image).astype(np.float32)
        
        # Subtle adjustments
        img_np[:, :, 0] *= 1.01  # Slight red boost
        img_np[:, :, 1] *= 1.005  # Very slight green boost
        img_np[:, :, 2] *= 0.995  # Slight blue reduction
        
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        return Image.fromarray(img_np)
    
    def _fallback_extraction(self, image):
        """Fallback extraction method"""
        try:
            # Simple color-based extraction as fallback
            img_rgba = image.convert('RGBA')
            return img_rgba
        except:
            return image.convert('RGBA')
