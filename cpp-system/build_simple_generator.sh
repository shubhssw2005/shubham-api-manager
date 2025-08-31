#!/bin/bash

echo "🚀 Building Simple High-Performance Data Generator"
echo "================================================"

# Check if we're on macOS and install dependencies
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "📦 Installing dependencies on macOS..."
    
    if ! command -v brew &> /dev/null; then
        echo "❌ Homebrew not found. Please install Homebrew first."
        exit 1
    fi
    
    # Install dependencies
    brew install curl nlohmann-json || echo "Dependencies already installed"
    
    echo "✅ Dependencies installed"
fi

# Build the simple generator
echo "🔨 Building simple data generator..."

g++ -std=c++17 -O3 -march=native -mtune=native \
    simple_data_generator.cpp \
    -lcurl \
    -o simple_data_generator \
    -I/opt/homebrew/include \
    -L/opt/homebrew/lib

if [ $? -eq 0 ]; then
    echo "✅ Build completed successfully!"
    echo ""
    echo "🚀 To run the high-performance data generator:"
    echo "   ./simple_data_generator"
    echo ""
    echo "📊 Expected performance:"
    echo "   - 76,000 posts in ~60-120 seconds"
    echo "   - 5-20x faster than JavaScript"
    echo "   - Multi-threaded batch processing"
    echo "   - REST API based (no MongoDB driver needed)"
    echo ""
    echo "⚠️  Make sure your Next.js API server is running on localhost:3000"
else
    echo "❌ Build failed!"
    echo "Try installing dependencies manually:"
    echo "   brew install curl nlohmann-json"
    exit 1
fi