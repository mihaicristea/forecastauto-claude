#!/bin/bash

# Professional Car Image Editor with AI Beautifier v2.0
# Enhanced script with support for all new features

set -e

# Clear any existing variables that might cause conflicts
unset INPUT_FILE OUTPUT_FILE BACKGROUND AI_STYLE AI_LEVEL LOGO_TEXT QUALITY VERBOSE EXTRA_ARGS

# Default values
INPUT_FILE="input/car.jpg"
OUTPUT_FILE="output/edited.jpg"
BACKGROUND="showroom"
AI_STYLE="glossy"
AI_LEVEL="medium"
LOGO_TEXT="Forecast AUTO"
QUALITY=95
VERBOSE=""
EXTRA_ARGS=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to show usage
show_usage() {
    echo -e "${CYAN}🚗 Professional Car Image Editor with AI Beautifier v2.0${NC}"
    echo -e "${CYAN}================================================================${NC}"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -i, --input FILE        Input car image (default: input/car.jpg)"
    echo "  -o, --output FILE       Output image path (default: output/edited.jpg)"
    echo "  -b, --background TYPE   Background type: showroom, studio, gradient, urban, custom"
    echo "  --custom-bg FILE        Custom background image (use with -b custom)"
    echo "  --ai-style STYLE        AI style: glossy, matte, metallic, luxury (default: glossy)"
    echo "  --ai-level LEVEL        AI level: light, medium, strong (default: medium)"
    echo "  --disable-ai            Disable AI beautification"
    echo "  --colors COLORS         Generate color variations (comma-separated)"
    echo "  --logo-text TEXT        Logo text (default: 'Forecast AUTO')"
    echo "  --no-logo               Disable logo overlay"
    echo "  --enhance               Apply additional enhancement"
    echo "  --quality N             JPEG quality 70-100 (default: 95)"
    echo "  -v, --verbose           Verbose output"
    echo "  --build                 Force rebuild Docker image"
    echo "  -h, --help              Show this help"
    echo ""
    echo "Examples:"
    echo "  $0                                          # Basic processing"
    echo "  $0 -i my_car.jpg -b studio --ai-style luxury"
    echo "  $0 --colors red,blue,black,white --ai-level strong"
    echo "  $0 --disable-ai --enhance -v"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            INPUT_FILE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -b|--background)
            BACKGROUND="$2"
            shift 2
            ;;
        --custom-bg)
            CUSTOM_BG="$2"
            EXTRA_ARGS="$EXTRA_ARGS --custom-bg $2"
            shift 2
            ;;
        --ai-style)
            AI_STYLE="$2"
            shift 2
            ;;
        --ai-level)
            AI_LEVEL="$2"
            shift 2
            ;;
        --disable-ai)
            EXTRA_ARGS="$EXTRA_ARGS --disable-ai"
            shift
            ;;
        --colors)
            EXTRA_ARGS="$EXTRA_ARGS --color-variations $2"
            shift 2
            ;;
        --logo-text)
            LOGO_TEXT="$2"
            shift 2
            ;;
        --no-logo)
            EXTRA_ARGS="$EXTRA_ARGS --no-logo"
            shift
            ;;
        --enhance)
            EXTRA_ARGS="$EXTRA_ARGS --enhance"
            shift
            ;;
        --quality)
            QUALITY="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE="--verbose"
            shift
            ;;
        --build)
            BUILD_FORCE="true"
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo -e "${RED}❌ Unknown option: $1${NC}"
            show_usage
            exit 1
            ;;
    esac
done

# Header
echo -e "${CYAN}🚗 Professional Car Image Editor with AI Beautifier v2.0${NC}"
echo -e "${CYAN}================================================================${NC}"
echo ""

# Check AI models cache status
echo -e "${BLUE}🔍 Checking AI models cache...${NC}"
if [ -d "models/cache/controlnet-canny" ] && [ -d "models/cache/stable-diffusion-v1-5" ]; then
    echo -e "${GREEN}✅ AI models are cached - processing will be fast${NC}"
    CACHE_SIZE=$(du -sh models/cache/ 2>/dev/null | cut -f1 || echo "unknown")
    echo -e "${GREEN}   Cache size: ${CACHE_SIZE}${NC}"
else
    echo -e "${YELLOW}⚠️  AI models not cached - first run will download ~3GB${NC}"
    echo -e "${YELLOW}   To avoid re-downloading: ./scripts/setup_cache.sh${NC}"
fi
echo ""

# Check if input file exists
if [[ ! -f "$INPUT_FILE" ]]; then
    echo -e "${RED}❌ Input file not found: $INPUT_FILE${NC}"
    echo -e "${YELLOW}💡 Please place your car image in the input/ directory${NC}"
    exit 1
fi

# Show configuration
echo -e "${BLUE}🔧 Configuration:${NC}"
echo -e "   Input: ${GREEN}$INPUT_FILE${NC}"
echo -e "   Output: ${GREEN}$OUTPUT_FILE${NC}"
echo -e "   Background: ${GREEN}$BACKGROUND${NC}"
echo -e "   AI Style: ${GREEN}$AI_STYLE${NC}"
echo -e "   AI Level: ${GREEN}$AI_LEVEL${NC}"
echo -e "   Logo: ${GREEN}$LOGO_TEXT${NC}"
echo -e "   Quality: ${GREEN}$QUALITY${NC}"
if [[ -n "$EXTRA_ARGS" ]]; then
    echo -e "   Extra: ${GREEN}$EXTRA_ARGS${NC}"
fi
echo ""

# Build Docker image if needed
if [[ "$BUILD_FORCE" == "true" ]] || ! docker images | grep -q "claude-forecastauto-car-editor"; then
    echo -e "${YELLOW}🔨 Building Docker image...${NC}"
    docker compose build car-editor
    echo ""
fi

# Create output directory
mkdir -p output

# Run the car editor
echo -e "${PURPLE}🎨 Processing car image...${NC}"
echo ""

# Build the command
CMD="python3 src/main.py --input $INPUT_FILE --output $OUTPUT_FILE --background $BACKGROUND --ai-style $AI_STYLE --ai-level $AI_LEVEL --logo-text \"$LOGO_TEXT\" --quality $QUALITY $VERBOSE $EXTRA_ARGS"

# Run with Docker
docker compose run --rm car-editor bash -c "$CMD"

# Check if processing was successful
if [[ $? -eq 0 ]]; then
    echo ""
    echo -e "${GREEN}✅ Processing completed successfully!${NC}"
    echo -e "${GREEN}📁 Results saved to: $OUTPUT_FILE${NC}"
    
    # Show additional outputs if color variations were generated
    if [[ "$EXTRA_ARGS" == *"--color-variations"* ]]; then
        echo -e "${GREEN}🎨 Color variations saved with suffix '_color_[colorname]'${NC}"
    fi
    
    echo ""
    echo -e "${CYAN}🎉 Professional car image editing complete!${NC}"
    echo -e "${YELLOW}💡 Check the output/ directory for your enhanced images${NC}"
else
    echo ""
    echo -e "${RED}❌ Processing failed. Check the error messages above.${NC}"
    exit 1
fi
