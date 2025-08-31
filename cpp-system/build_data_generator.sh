#!/bin/bash

echo "🚀 Building High-Performance Data Generator"
echo "=========================================="

# Check if we're on macOS and install dependencies
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "📦 Installing MongoDB C++ driver on macOS..."
    
    # Install MongoDB C++ driver using Homebrew
    if ! command -v brew &> /dev/null; then
        echo "❌ Homebrew not found. Please install Homebrew first."
        exit 1
    fi
    
    # Install dependencies
    brew install mongo-cxx-driver || echo "MongoDB C++ driver already installed"
    brew install pkg-config || echo "pkg-config already installed"
    
    echo "✅ Dependencies installed"
fi

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "🔧 Configuring build..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_CXX_FLAGS="-O3 -march=native -mtune=native -flto" \
    -DENABLE_PERFORMANCE_MONITORING=ON

# Build the project
echo "🔨 Building..."
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu) massive_data_generator

if [ $? -eq 0 ]; then
    echo "✅ Build completed successfully!"
    echo ""
    echo "🚀 To run the high-performance data generator:"
    echo "   cd build && ./src/data-generator/massive_data_generator"
    echo ""
    echo "📊 Expected performance:"
    echo "   - 76,000 posts in ~30-60 seconds"
    echo "   - 10-50x faster than JavaScript"
    echo "   - Multi-threaded batch processing"
    echo "   - Memory-optimized operations"
else
    echo "❌ Build failed!"
    exit 1
fi