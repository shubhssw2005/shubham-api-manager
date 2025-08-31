#!/bin/bash
# Simple build script for C++ system without Conan dependencies

set -e

echo "Building C++ system with simple configuration..."

# Create build directory
mkdir -p build
cd build

# Simple compilation for test program
echo "Compiling test program..."
g++ -std=c++17 -O2 -o test_real_data ../simple_test.cpp -pthread

echo "Build complete!"
echo "Binary: build/test_real_data"