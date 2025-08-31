#!/bin/bash
# Build script for Ultra Low-Latency C++ System

set -e

BUILD_TYPE=${1:-Release}
CLEAN=${2:-false}

echo "Building Ultra Low-Latency C++ System (${BUILD_TYPE})..."

# Clean build if requested
if [[ "$CLEAN" == "clean" ]]; then
    echo "Cleaning previous build..."
    rm -rf build/
fi

# Install dependencies
echo "Installing dependencies..."
conan install . --build=missing -s build_type=${BUILD_TYPE}

# Configure CMake
echo "Configuring CMake..."
if [[ "$BUILD_TYPE" == "Debug" ]]; then
    cmake --preset conan-debug
else
    cmake --preset conan-release
fi

# Build
echo "Building..."
cmake --build --preset conan-${BUILD_TYPE,,} -j$(nproc)

# Run tests
echo "Running tests..."
cd build/${BUILD_TYPE}
ctest --output-on-failure

echo "Build complete!"
echo "Binaries are in build/${BUILD_TYPE}/bin/"