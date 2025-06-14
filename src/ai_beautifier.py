import torch
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
from diffusers import (
    StableDiffusionControlNetPipeline, 
    ControlNetModel,
    StableDiffusionImg2ImgPipeline,
    DDIMScheduler
)
from controlnet_aux import CannyDetector, OpenposeDetector
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

class AIBeautifier:
    """
    AI Beautifier pentru mașini folosind ControlNet + Stable Diffusion
    Creează efecte profesionale: vopsea lucioasă, eliminare pete, reflexii soft
    """
    
    def __init__(self, config=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config or {}
        
        print("🤖 Initializing AI Beautifier...")
        print(f"   Device: {self.device}")
        
        # Initialize models
        self._load_models()
        
        # Initialize preprocessors
        self.canny_detector = CannyDetector()
        
        print("✅ AI Beautifier ready!")
    
    def _load_models(self):
        """Load ControlNet and Stable Diffusion models with local caching"""
        try:
            # Setup local cache directories
            cache_dir = Path("models/cache")
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            controlnet_cache = cache_dir / "controlnet-canny"
            sd_cache = cache_dir / "stable-diffusion-v1-5"
            
            # Load ControlNet model for car enhancement
            print("📥 Loading ControlNet model...")
            if controlnet_cache.exists() and any(controlnet_cache.iterdir()):
                print("   Using cached ControlNet model...")
                self.controlnet = ControlNetModel.from_pretrained(
                    str(controlnet_cache),
                    torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                    use_safetensors=True,
                    local_files_only=True
                )
            else:
                print("   Downloading ControlNet model (first time only)...")
                self.controlnet = ControlNetModel.from_pretrained(
                    "lllyasviel/sd-controlnet-canny",
                    torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                    use_safetensors=True,
                    cache_dir=str(cache_dir)
                )
                # Save to local cache
                self.controlnet.save_pretrained(str(controlnet_cache))
            
            # Load Stable Diffusion pipeline with ControlNet
            print("📥 Loading Stable Diffusion pipeline...")
            if sd_cache.exists() and any(sd_cache.iterdir()):
                print("   Using cached Stable Diffusion model...")
                self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                    str(sd_cache),
                    controlnet=self.controlnet,
                    torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False,
                    use_safetensors=True,
                    local_files_only=True
                )
            else:
                print("   Downloading Stable Diffusion model (first time only)...")
                self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    controlnet=self.controlnet,
                    torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False,
                    use_safetensors=True,
                    cache_dir=str(cache_dir)
                )
                # Save to local cache
                self.pipe.save_pretrained(str(sd_cache))
            
            # Optimize for memory and speed
            self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
            self.pipe = self.pipe.to(self.device)
            
            if self.device.type == 'cuda':
                # Enable memory efficient attention
                self.pipe.enable_attention_slicing()
                self.pipe.enable_model_cpu_offload()
                
                # Try to enable xformers if available
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                    print("✅ XFormers enabled for better performance")
                except:
                    print("⚠️  XFormers not available, using standard attention")
            
            # Load img2img pipeline for refinement (reuse the same base model)
            print("📥 Loading img2img pipeline...")
            if sd_cache.exists() and any(sd_cache.iterdir()):
                print("   Using cached model for img2img...")
                self.img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                    str(sd_cache),
                    torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False,
                    use_safetensors=True,
                    local_files_only=True
                )
            else:
                print("   Creating img2img pipeline from cached components...")
                self.img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False,
                    use_safetensors=True,
                    cache_dir=str(cache_dir)
                )
            
            self.img2img_pipe = self.img2img_pipe.to(self.device)
            
            if self.device.type == 'cuda':
                self.img2img_pipe.enable_attention_slicing()
                self.img2img_pipe.enable_model_cpu_offload()
            
            print("✅ All models loaded successfully!")
                
        except Exception as e:
            print(f"⚠️  Error loading AI models: {e}")
            print("   Falling back to traditional enhancement...")
            self.controlnet = None
            self.pipe = None
            self.img2img_pipe = None
    
    def beautify_car(self, car_image, enhancement_level="medium", style="glossy"):
        """
        Aplică beautify AI pe imaginea mașinii
        
        Args:
            car_image: PIL Image - imaginea mașinii (RGBA sau RGB)
            enhancement_level: str - "light", "medium", "strong"
            style: str - "glossy", "matte", "metallic", "luxury"
        
        Returns:
            PIL Image - imaginea îmbunătățită
        """
        try:
            print(f"✨ AI Beautifying car with {enhancement_level} {style} enhancement...")
            
            # Convert to RGB if needed
            if car_image.mode == 'RGBA':
                # Save alpha channel
                alpha_channel = car_image.split()[-1]
                car_rgb = Image.new('RGB', car_image.size, (255, 255, 255))
                car_rgb.paste(car_image, mask=alpha_channel)
            else:
                car_rgb = car_image.convert('RGB')
                alpha_channel = None
            
            # Apply AI enhancement
            if self.pipe is not None:
                enhanced_rgb = self._ai_enhance_with_controlnet(car_rgb, enhancement_level, style)
            else:
                enhanced_rgb = self._traditional_enhance(car_rgb, enhancement_level, style)
            
            # Apply post-processing
            enhanced_rgb = self._post_process_enhancement(enhanced_rgb, style)
            
            # Restore alpha channel if it existed
            if alpha_channel:
                enhanced_rgba = enhanced_rgb.convert('RGBA')
                enhanced_rgba.putalpha(alpha_channel)
                return enhanced_rgba
            
            return enhanced_rgb
            
        except Exception as e:
            print(f"⚠️  AI Beautify error: {e}")
            # Fallback to original image
            return car_image
    
    def _ai_enhance_with_controlnet(self, car_rgb, enhancement_level, style):
        """Enhance using ControlNet + Stable Diffusion"""
        try:
            # Resize for optimal processing
            original_size = car_rgb.size
            process_size = self._get_optimal_size(original_size)
            
            if process_size != original_size:
                car_resized = car_rgb.resize(process_size, Image.Resampling.LANCZOS)
            else:
                car_resized = car_rgb
            
            # Generate Canny edge map
            print("   Generating edge map...")
            canny_image = self.canny_detector(car_resized)
            
            # Create enhancement prompt based on style
            prompt = self._get_enhancement_prompt(style, enhancement_level)
            negative_prompt = self._get_negative_prompt()
            
            # Generate enhanced image
            print("   Generating AI enhancement...")
            with torch.autocast(self.device.type):
                enhanced = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=canny_image,
                    num_inference_steps=self._get_inference_steps(enhancement_level),
                    guidance_scale=self._get_guidance_scale(enhancement_level),
                    controlnet_conditioning_scale=0.8,
                    generator=torch.Generator(device=self.device).manual_seed(42)
                ).images[0]
            
            # Refine with img2img
            print("   Refining with img2img...")
            enhanced = self._refine_with_img2img(enhanced, car_resized, style)
            
            # Resize back to original size
            if process_size != original_size:
                enhanced = enhanced.resize(original_size, Image.Resampling.LANCZOS)
            
            return enhanced
            
        except Exception as e:
            print(f"   ControlNet enhancement failed: {e}")
            return self._traditional_enhance(car_rgb, enhancement_level, style)
    
    def _refine_with_img2img(self, ai_enhanced, original, style):
        """Refine AI result with img2img for better consistency"""
        try:
            prompt = f"professional car photography, {style} paint finish, studio lighting, high quality, detailed"
            negative_prompt = "blurry, distorted, artifacts, low quality"
            
            with torch.autocast(self.device.type):
                refined = self.img2img_pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=ai_enhanced,
                    strength=0.3,  # Light refinement
                    num_inference_steps=20,
                    guidance_scale=7.5,
                    generator=torch.Generator(device=self.device).manual_seed(42)
                ).images[0]
            
            # Blend with original for natural look
            return Image.blend(original, refined, 0.6)
            
        except Exception as e:
            print(f"   Img2img refinement failed: {e}")
            return ai_enhanced
    
    def _get_enhancement_prompt(self, style, level):
        """Generate prompt based on style and enhancement level"""
        base_prompt = "professional automotive photography, luxury car, studio lighting, high resolution, detailed"
        
        style_prompts = {
            "glossy": "glossy paint finish, mirror-like reflections, wet look, shiny surface",
            "matte": "matte paint finish, smooth surface, elegant, sophisticated",
            "metallic": "metallic paint, pearl finish, color-shifting, premium",
            "luxury": "luxury car finish, premium paint, showroom quality, pristine condition"
        }
        
        level_modifiers = {
            "light": "subtle enhancement, natural look",
            "medium": "enhanced beauty, improved finish",
            "strong": "dramatic enhancement, perfect finish, flawless"
        }
        
        return f"{base_prompt}, {style_prompts.get(style, style_prompts['glossy'])}, {level_modifiers.get(level, level_modifiers['medium'])}"
    
    def _get_negative_prompt(self):
        """Get negative prompt to avoid unwanted artifacts"""
        return "blurry, distorted, artifacts, scratches, dents, rust, dirt, low quality, bad anatomy, deformed, ugly, pixelated, noise"
    
    def _get_inference_steps(self, level):
        """Get number of inference steps based on enhancement level"""
        steps = {
            "light": 15,
            "medium": 25,
            "strong": 35
        }
        return steps.get(level, 25)
    
    def _get_guidance_scale(self, level):
        """Get guidance scale based on enhancement level"""
        scales = {
            "light": 6.0,
            "medium": 7.5,
            "strong": 9.0
        }
        return scales.get(level, 7.5)
    
    def _get_optimal_size(self, original_size):
        """Get optimal processing size (multiple of 64 for Stable Diffusion)"""
        width, height = original_size
        
        # Target around 512-768 pixels for best quality/speed balance
        max_dim = 768
        
        if max(width, height) > max_dim:
            if width > height:
                new_width = max_dim
                new_height = int(height * max_dim / width)
            else:
                new_height = max_dim
                new_width = int(width * max_dim / height)
        else:
            new_width, new_height = width, height
        
        # Round to nearest multiple of 64
        new_width = ((new_width + 63) // 64) * 64
        new_height = ((new_height + 63) // 64) * 64
        
        return (new_width, new_height)
    
    def _traditional_enhance(self, car_rgb, enhancement_level, style):
        """Traditional enhancement fallback when AI models are not available"""
        print("   Using traditional enhancement...")
        
        enhanced = car_rgb.copy()
        
        # Enhancement based on level
        if enhancement_level == "light":
            contrast_factor = 1.1
            brightness_factor = 1.05
            saturation_factor = 1.1
        elif enhancement_level == "medium":
            contrast_factor = 1.2
            brightness_factor = 1.1
            saturation_factor = 1.15
        else:  # strong
            contrast_factor = 1.3
            brightness_factor = 1.15
            saturation_factor = 1.2
        
        # Apply enhancements
        enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = enhancer.enhance(contrast_factor)
        
        enhancer = ImageEnhance.Brightness(enhanced)
        enhanced = enhancer.enhance(brightness_factor)
        
        enhancer = ImageEnhance.Color(enhanced)
        enhanced = enhancer.enhance(saturation_factor)
        
        # Style-specific enhancements
        if style == "glossy":
            enhanced = self._add_glossy_effect(enhanced)
        elif style == "metallic":
            enhanced = self._add_metallic_effect(enhanced)
        
        return enhanced
    
    def _add_glossy_effect(self, image):
        """Add glossy paint effect"""
        # Enhance highlights
        img_np = np.array(image)
        
        # Create highlight mask
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        _, highlights = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        
        # Enhance bright areas
        mask = highlights > 0
        img_np[mask] = np.clip(img_np[mask] * 1.2, 0, 255)
        
        return Image.fromarray(img_np.astype(np.uint8))
    
    def _add_metallic_effect(self, image):
        """Add metallic paint effect"""
        # Add subtle color variation
        img_np = np.array(image).astype(np.float32)
        
        # Create subtle color shifts
        noise = np.random.normal(0, 3, img_np.shape)
        img_np += noise
        
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        return Image.fromarray(img_np)
    
    def _post_process_enhancement(self, enhanced_image, style):
        """Apply final post-processing touches"""
        # Subtle sharpening
        enhanced = enhanced_image.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
        
        # Style-specific post-processing
        if style == "glossy":
            # Add slight blur to create depth
            blurred = enhanced.filter(ImageFilter.GaussianBlur(radius=0.5))
            enhanced = Image.blend(enhanced, blurred, 0.1)
        
        elif style == "luxury":
            # Enhance overall quality
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(1.1)
        
        return enhanced
    
    def create_paint_variations(self, car_image, colors=None):
        """
        Creează variații de culoare pentru vopsea folosind AI
        
        Args:
            car_image: PIL Image - imaginea mașinii
            colors: list - lista de culori dorite ["red", "blue", "black", etc.]
        
        Returns:
            dict - dicționar cu variațiile de culoare
        """
        if colors is None:
            colors = ["red", "blue", "black", "white", "silver", "gold"]
        
        variations = {}
        
        for color in colors:
            try:
                print(f"🎨 Creating {color} paint variation...")
                
                if self.pipe is not None:
                    variation = self._create_color_variation_ai(car_image, color)
                else:
                    variation = self._create_color_variation_traditional(car_image, color)
                
                variations[color] = variation
                
            except Exception as e:
                print(f"   Failed to create {color} variation: {e}")
                variations[color] = car_image
        
        return variations
    
    def _create_color_variation_ai(self, car_image, color):
        """Create color variation using AI"""
        # This would use ControlNet with color-specific prompts
        # For now, fallback to traditional method
        return self._create_color_variation_traditional(car_image, color)
    
    def _create_color_variation_traditional(self, car_image, color):
        """Create color variation using traditional methods"""
        # Convert to HSV for color manipulation
        if car_image.mode == 'RGBA':
            alpha = car_image.split()[-1]
            rgb_image = Image.new('RGB', car_image.size, (255, 255, 255))
            rgb_image.paste(car_image, mask=alpha)
        else:
            rgb_image = car_image.convert('RGB')
            alpha = None
        
        # Color mapping
        color_shifts = {
            "red": (0, 1.3, 1.1),      # H, S, V multipliers
            "blue": (240/360, 1.2, 1.0),
            "black": (0, 0.3, 0.3),
            "white": (0, 0.1, 1.8),
            "silver": (0, 0.2, 1.4),
            "gold": (45/360, 1.1, 1.2)
        }
        
        if color in color_shifts:
            h_shift, s_mult, v_mult = color_shifts[color]
            
            # Convert to HSV
            hsv_image = rgb_image.convert('HSV')
            h, s, v = hsv_image.split()
            
            # Apply color transformation
            h_np = np.array(h).astype(np.float32)
            s_np = np.array(s).astype(np.float32)
            v_np = np.array(v).astype(np.float32)
            
            # Shift hue
            if h_shift > 0:
                h_np = (h_np + h_shift * 255) % 255
            
            # Adjust saturation and value
            s_np = np.clip(s_np * s_mult, 0, 255)
            v_np = np.clip(v_np * v_mult, 0, 255)
            
            # Convert back
            h_new = Image.fromarray(h_np.astype(np.uint8))
            s_new = Image.fromarray(s_np.astype(np.uint8))
            v_new = Image.fromarray(v_np.astype(np.uint8))
            
            colored_image = Image.merge('HSV', (h_new, s_new, v_new)).convert('RGB')
        else:
            colored_image = rgb_image
        
        # Restore alpha if it existed
        if alpha:
            colored_rgba = colored_image.convert('RGBA')
            colored_rgba.putalpha(alpha)
            return colored_rgba
        
        return colored_image
    
    def cleanup(self):
        """Cleanup GPU memory"""
        if hasattr(self, 'pipe') and self.pipe is not None:
            del self.pipe
        if hasattr(self, 'img2img_pipe') and self.img2img_pipe is not None:
            del self.img2img_pipe
        if hasattr(self, 'controlnet') and self.controlnet is not None:
            del self.controlnet
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("🧹 AI Beautifier memory cleaned up")
