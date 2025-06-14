# 🐳 Docker Guide - AI Beautifier pentru Mașini

Ghid complet pentru rularea AI Beautifier cu ControlNet + Stable Diffusion în Docker.

## 📋 Prerequisite

1. **Docker** instalat
2. **NVIDIA GPU** cu driver instalat
3. **nvidia-docker2** sau **Docker 19.03+** cu suport GPU
4. **Imagine cu mașină** în directorul `input/car.jpg`

## 🚀 Comenzi Docker pentru AI Beautifier

### 1. Generarea Imaginii Docker

```bash
# Construiește imaginea Docker cu toate dependențele AI
docker build -t forecast-auto-ai .
```

**Această comandă va:**
- Instala CUDA 11.8 și cuDNN
- Instala Python 3.10 și toate dependențele
- Descărca modelele AI (ControlNet, Stable Diffusion)
- Configura mediul pentru GPU

### 2. Rularea cu Docker Compose (RECOMANDAT)

#### A. Test Complet AI Beautifier
```bash
# Testează toate funcționalitățile AI Beautifier
docker-compose run ai-beautifier-test
```

#### B. Demo AI Beautifier
```bash
# Creează demo grid cu toate stilurile
docker-compose run ai-beautifier-demo
```

#### C. Procesare Standard cu AI
```bash
# Procesează o imagine cu AI Beautifier integrat
docker-compose run car-editor
```

### 3. Rularea Directă cu Docker

#### A. Test AI Beautifier
```bash
docker run --gpus all \
    -v $(pwd)/input:/app/input \
    -v $(pwd)/output:/app/output \
    forecast-auto-ai \
    python3 test_ai_beautifier.py
```

#### B. Procesare cu AI Beautifier
```bash
docker run --gpus all \
    -v $(pwd)/input:/app/input \
    -v $(pwd)/output:/app/output \
    forecast-auto-ai \
    python3 src/main.py --input input/car.jpg --output output/ai_enhanced.jpg
```

#### C. Demo Complet
```bash
docker run --gpus all \
    -v $(pwd)/input:/app/input \
    -v $(pwd)/output:/app/output \
    forecast-auto-ai \
    python3 demo_ai_beautifier.py
```

## 🎨 Opțiuni Avansate

### Rulare cu Stiluri Specifice
```bash
# Stil glossy cu enhancement medium
docker run --gpus all \
    -v $(pwd)/input:/app/input \
    -v $(pwd)/output:/app/output \
    -e AI_STYLE=glossy \
    -e AI_LEVEL=medium \
    forecast-auto-ai \
    python3 src/main.py --input input/car.jpg
```

### Rulare cu Background Specific
```bash
# Background showroom cu AI enhancement
docker run --gpus all \
    -v $(pwd)/input:/app/input \
    -v $(pwd)/output:/app/output \
    -v $(pwd)/backgrounds:/app/backgrounds \
    forecast-auto-ai \
    python3 src/main.py --input input/car.jpg --background showroom
```

## 📁 Structura Directoarelor

```
project/
├── input/           # Pune imaginile aici
│   └── car.jpg     # Imaginea ta cu mașina
├── output/         # Rezultatele vor fi aici
│   ├── ai_beautified_*.jpg
│   ├── ai_color_*.jpg
│   ├── ai_integrated_*.jpg
│   └── ai_beautifier_demo.jpg
└── backgrounds/    # Fundaluri custom (opțional)
```

## 🎯 Rezultate Generate

După rulare, vei găsi în `output/`:

### AI Beautifier Results:
- `ai_beautified_light_glossy.jpg` - Enhancement ușor, stil lucios
- `ai_beautified_medium_glossy.jpg` - Enhancement mediu, stil lucios
- `ai_beautified_strong_glossy.jpg` - Enhancement puternic, stil lucios
- `ai_beautified_medium_metallic.jpg` - Stil metalic
- `ai_beautified_medium_luxury.jpg` - Stil luxury
- `ai_beautified_medium_matte.jpg` - Stil mat

### Color Variations:
- `ai_color_red.jpg` - Variația roșie
- `ai_color_blue.jpg` - Variația albastră
- `ai_color_black.jpg` - Variația neagră
- `ai_color_white.jpg` - Variația albă
- `ai_color_silver.jpg` - Variația argintie

### Integration Results:
- `ai_integrated_showroom.jpg` - Pipeline complet cu background showroom
- `ai_integrated_studio.jpg` - Pipeline complet cu background studio
- `ai_integrated_gradient.jpg` - Pipeline complet cu background gradient

### Demo Files:
- `ai_beautifier_comparison.jpg` - Comparație before/after
- `ai_beautifier_demo.jpg` - Grilă completă cu toate stilurile

## 🔧 Troubleshooting

### 1. GPU nu este detectat
```bash
# Verifică driver NVIDIA
nvidia-smi

# Testează Docker cu GPU
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### 2. Out of Memory
```bash
# Rulează cu memorie limitată
docker run --gpus all --memory=8g \
    -v $(pwd)/input:/app/input \
    -v $(pwd)/output:/app/output \
    forecast-auto-ai \
    python3 test_ai_beautifier.py
```

### 3. Modele nu se descarcă
```bash
# Descarcă manual modelele
docker run --gpus all \
    -v $(pwd)/models:/app/models \
    forecast-auto-ai \
    python3 models/download_models.py
```

### 4. Permission Denied
```bash
# Fixează permisiunile
sudo chmod -R 777 input output
```

## ⚡ Optimizări Performanță

### Pentru GPU-uri cu VRAM limitat:
```bash
# Rulează cu optimizări de memorie
docker run --gpus all \
    -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
    -v $(pwd)/input:/app/input \
    -v $(pwd)/output:/app/output \
    forecast-auto-ai \
    python3 test_ai_beautifier.py
```

### Pentru procesare batch:
```bash
# Procesează multiple imagini
for img in input/*.jpg; do
    docker run --gpus all \
        -v $(pwd)/input:/app/input \
        -v $(pwd)/output:/app/output \
        forecast-auto-ai \
        python3 src/main.py --input "$img"
done
```

## 🎉 Quick Start

1. **Pune imaginea** în `input/car.jpg`
2. **Construiește imaginea**: `docker build -t forecast-auto-ai .`
3. **Rulează testul**: `docker-compose run ai-beautifier-test`
4. **Verifică rezultatele** în `output/`

## 📊 Monitorizare

### Verifică utilizarea GPU:
```bash
# În timpul rulării
watch -n 1 nvidia-smi
```

### Verifică logurile:
```bash
# Loguri Docker Compose
docker-compose logs ai-beautifier-test

# Loguri container direct
docker logs <container_id>
```

---

🤖 **AI Beautifier** este gata să transforme imaginile tale de mașini în fotografii profesionale de showroom!
