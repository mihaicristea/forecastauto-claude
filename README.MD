# 🚗 Professional Car Image Editor with AI Beautifier v2.0

Editează profesional imagini cu mașini folosind AI avansată și procesare GPU optimizată.

## 🚀 Features

- ✅ **Advanced AI Segmentation** - U2Net pentru extragere profesională
- ✅ **AI Beautifier cu ControlNet + Stable Diffusion**
  - 🎨 Vopsea lucioasă cu reflexii mirror-like
  - 🧹 Eliminare automată pete și imperfecțiuni
  - ✨ Reflexii soft și efecte profesionale
  - 🎭 Multiple stiluri: glossy, matte, metallic, luxury
  - 🌈 Generare variații de culoare automată
- ✅ **Professional Backgrounds** - showroom, studio, gradient, urban
- ✅ **Smart Enhancement Levels** - light, medium, strong
- ✅ **GPU Optimization** - Memory management și fallback automat
- ✅ **Professional Compositing** - umbre realiste și reflexii
- ✅ **Branding Integration** - logo overlay personalizabil

## 📋 Prerequisites

- Docker
- NVIDIA GPU cu driver instalat (opțional, funcționează și pe CPU)
- nvidia-docker2 sau Docker 19.03+ cu suport GPU

## 🛠️ Installation

1. Clonează sau dezarhivează proiectul:
```bash
unzip car-image-editor.zip
cd car-image-editor
```

2. Rulează setup:
```bash
chmod +x setup.sh
./setup.sh
```

## 🎮 Usage

### Quick Start:
```bash
./scripts/run.sh
```

### Advanced Usage:
```bash
# Professional glossy finish
./scripts/run.sh -i input/car.jpg -b showroom --ai-style glossy --ai-level medium

# Luxury metallic with studio background
./scripts/run.sh -i input/car.jpg -b studio --ai-style luxury --ai-level strong

# Generate color variations
./scripts/run.sh --colors red,blue,black,white,silver,gold

# Disable AI for traditional processing
./scripts/run.sh --disable-ai --enhance

# Custom background
./scripts/run.sh -b custom --custom-bg my_background.jpg
```

### Command Line Options:
```bash
./scripts/run.sh [OPTIONS]

Options:
  -i, --input FILE        Input car image (default: input/car.jpg)
  -o, --output FILE       Output image path (default: output/edited.jpg)
  -b, --background TYPE   Background: showroom, studio, gradient, urban, custom
  --custom-bg FILE        Custom background image
  --ai-style STYLE        AI style: glossy, matte, metallic, luxury
  --ai-level LEVEL        AI level: light, medium, strong
  --disable-ai            Disable AI beautification
  --colors COLORS         Generate color variations (comma-separated)
  --logo-text TEXT        Logo text (default: 'Forecast AUTO')
  --no-logo               Disable logo overlay
  --enhance               Apply additional enhancement
  --quality N             JPEG quality 70-100 (default: 95)
  -v, --verbose           Verbose output
  --build                 Force rebuild Docker image
  -h, --help              Show help
```

### Direct Python Usage:
```bash
# Basic processing
python3 src/main.py --input input/car.jpg

# Advanced AI processing
python3 src/main.py \
    --input input/car.jpg \
    --output output/enhanced.jpg \
    --background showroom \
    --ai-style luxury \
    --ai-level strong \
    --color-variations red,blue,black,white
```

## 🤖 AI Beautifier Features

### 🎨 Stiluri Disponibile:
- **Glossy** - Vopsea lucioasă cu reflexii mirror-like
- **Matte** - Finisaj mat elegant și sofisticat  
- **Metallic** - Vopsea metalică cu efecte pearl
- **Luxury** - Finisaj premium de calitate showroom

### 📊 Nivele de Enhancement:
- **Light** - Îmbunătățiri subtile, aspect natural
- **Medium** - Enhancement echilibrat (recomandat)
- **Strong** - Îmbunătățiri dramatice, finisaj perfect

### 🌈 Color Variations:
Generează automat variații de culoare pentru aceeași mașină:
- Red, Blue, Black, White, Silver, Gold
- Menține calitatea și stilul original
- Salvează fiecare variație separat

### 🎯 Caracteristici AI:
- 🔧 **Fallback automat** la metode tradiționale dacă AI nu e disponibil
- 🧹 **Memory management** inteligent pentru GPU
- 🎨 **ControlNet integration** pentru control precis
- ✨ **Professional effects** - eliminare pete, reflexii soft, vopsea lucioasă
- 🚀 **GPU optimization** cu XFormers și attention slicing

## 🎨 Background Options

- **showroom** - Fundal showroom profesional cu reflexii pe podea
- **studio** - Fundal studio neutru cu iluminare profesională
- **gradient** - Fundal gradient radial elegant
- **urban** - Fundal urban modern (în dezvoltare)
- **custom** - Folosește propriul fundal cu `--custom-bg path/to/bg.jpg`

## 🧪 Testing & Demo

### Test AI Beautifier:
```bash
python3 test_ai_beautifier.py
```

### Comprehensive Demo:
```bash
python3 demo_comprehensive.py
```

### Quick Tests:
```bash
# Test different styles
./scripts/test_ai_beautifier.sh

# Run all tests
./scripts/run_test.sh
```

## 📁 Project Structure

```
├── src/                    # Core source code
│   ├── main.py            # Enhanced main interface
│   ├── car_editor.py      # Professional car editor
│   ├── ai_beautifier.py   # AI beautification engine
│   └── ...
├── input/                 # Input images
├── output/                # Generated results
├── scripts/               # Enhanced scripts
│   ├── run.sh            # Main execution script
│   ├── test_ai_beautifier.sh
│   └── ...
├── demo_comprehensive.py  # Full feature demo
├── test_ai_beautifier.py # AI testing suite
└── README.MD             # This file
```

## ⚙️ Configuration

Editează `config.yaml` pentru:
- Parametri AI Beautifier
- Setări GPU și memory management
- Configurare enhancement
- Parametri de procesare

## 🔧 Advanced Features

### Memory Management:
- Automatic GPU memory cleanup
- CPU fallback pentru sisteme fără GPU
- Optimized model loading și unloading

### Professional Compositing:
- Realistic shadow generation
- Floor reflections pentru showroom
- Professional lighting effects
- Film grain și color grading

### Quality Control:
- Multiple quality levels (70-100)
- Professional sharpening
- Color balance adjustment
- Edge refinement

## 🐛 Troubleshooting

### Quick Diagnosis:
```bash
# Run the debug script to identify issues
./scripts/debug_run.sh

# Test basic functionality without Docker
python3 test_simple.py
```

### Common Issues:

1. **Script execution errors:**
   ```bash
   # Make scripts executable
   chmod +x scripts/*.sh
   
   # Clear environment variables
   unset INPUT_FILE OUTPUT_FILE BACKGROUND
   
   # Run from project root directory
   cd /path/to/claude-forecastauto
   ./scripts/run.sh
   ```

2. **Missing input image:**
   ```bash
   # Add a car image to input directory
   cp your_car_image.jpg input/car.jpg
   
   # Or specify custom input
   ./scripts/run.sh -i path/to/your/car.jpg
   ```

3. **Docker issues:**
   ```bash
   # Check Docker installation
   docker --version
   docker compose version
   
   # Build the image
   docker compose build car-editor
   
   # Check GPU support
   nvidia-smi
   docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
   ```

4. **Out of memory:**
   ```bash
   # Use CPU-only mode
   ./scripts/run.sh --disable-ai
   
   # Or reduce enhancement level
   ./scripts/run.sh --ai-level light
   ```

5. **AI models not loading:**
   ```bash
   # Check internet connection for model download
   # Or use traditional enhancement
   ./scripts/run.sh --disable-ai --enhance
   ```

6. **Permission denied:**
   ```bash
   chmod +x scripts/*.sh
   chmod -R 755 input output
   ```

### Debug Tools:
- `./scripts/debug_run.sh` - Comprehensive system check
- `python3 test_simple.py` - Quick functionality test
- `./scripts/run.sh -h` - Show all available options
- `./scripts/run.sh -v` - Verbose output for debugging

## 📊 Performance

- **GPU Processing**: 15-30s per image (depending on complexity)
- **CPU Fallback**: 45-90s per image
- **Memory Usage**: 4-8GB VRAM (GPU) / 8-16GB RAM (CPU)
- **Quality**: Professional automotive photography standard

## 🎯 Use Cases

- **Automotive Dealerships** - Professional car photography
- **Online Marketplaces** - Enhanced vehicle listings
- **Marketing Materials** - High-quality promotional images
- **Social Media** - Eye-catching car content
- **Personal Projects** - Beautify your car photos

## 📝 License

MIT License

## 🤝 Contributing

Contributions welcome! Please read the contributing guidelines and submit pull requests.

## 📞 Support

For support and questions:
- Check the troubleshooting section
- Review the demo scripts
- Submit issues on GitHub

---
Made with ❤️ for Forecast AUTO - Professional Car Image Enhancement
