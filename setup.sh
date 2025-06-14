echo "🚀 Setting up Car Image Editor..."

# Create directories
mkdir -p input output backgrounds models

# Download sample backgrounds
echo "📥 Downloading sample backgrounds..."
# Add wget commands for sample backgrounds here

# Build Docker image
echo "🐳 Building Docker image..."
docker build -t car-image-editor .

echo "✅ Setup complete!"