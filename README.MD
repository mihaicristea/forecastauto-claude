# 🚗 Professional Car Image Editor with GPU Support

Editează profesional imagini cu mașini folosind AI și procesare GPU.

## 🚀 Features

- ✅ Background removal cu AI (rembg)
- ✅ Înlocuire fundal profesional
- ✅ Detectare și înlocuire plăcuță înmatriculare
- ✅ Text overlay pentru logo
- ✅ Enhancement automat (sharpen, contrast, color)
- ✅ Suport GPU NVIDIA pentru procesare rapidă

## 📋 Prerequisites

- Docker
- NVIDIA GPU cu driver instalat
- nvidia-docker2 sau Docker 19.03+ cu suport GPU

## 🛠️ Installation

1. Clonează sau dezarhivează proiectul:
```bash
unzip car-image-editor.zip
cd car-image-editor
```

2. Rulează setup:
```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

## 🎮 Usage

### Quick Start:
```bash
chmod +x scripts/run.sh
./scripts/run.sh --input input/car.jpg --output output/edited.jpg
```

### Full Options:
```bash
./scripts/run.sh \
    --input input/car.jpg \
    --output output/edited.jpg \
    --background urban \
    --logo-text "Forecast AUTO" \
    --enhance
```

### Direct Docker:
```bash
docker run --gpus all \
    -v $(pwd)/input:/app/input \
    -v $(pwd)/output:/app/output \
    car-image-editor \
    python3 src/main.py --input input/car.jpg
```

## 🎨 Background Options

- `showroom` - Fundal showroom profesional
- `urban` - Fundal urban modern
- `studio` - Fundal studio neutru
- `custom` - Folosește propriul fundal cu `--custom-bg path/to/bg.jpg`

## 📁 Project Structure

```
input/          # Pune imaginile aici
output/         # Rezultatele vor fi salvate aici
backgrounds/    # Fundaluri predefinite
configs/        # Configurări
```

## ⚙️ Configuration

Editează `configs/config.yaml` pentru:
- Parametri de procesare
- Setări enhancement
- Configurare logo

## 🐛 Troubleshooting

1. **GPU not detected:**
   ```bash
   nvidia-smi  # Verifică driver NVIDIA
   docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
   ```

2. **Out of memory:**
   - Redu dimensiunea imaginii
   - Dezactivează SAM în config

3. **Permission denied:**
   ```bash
   chmod -R 777 input output
   ```

## 📝 License

MIT License

---
Made with ❤️ for Forecast AUTO