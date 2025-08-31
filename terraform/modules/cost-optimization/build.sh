#!/bin/bash

# Build script for C++ Lambda function
set -e

echo "Building Cost Optimizer Lambda function..."

# Check if Docker is available for cross-compilation
if command -v docker &> /dev/null; then
    echo "Using Docker for cross-compilation..."
    
    # Build using Amazon Linux 2 container
    docker run --rm -v "$PWD":/workspace -w /workspace amazonlinux:2 bash -c "
        yum update -y && \
        yum install -y gcc-c++ cmake3 make git curl-devel openssl-devel && \
        
        # Install AWS SDK dependencies
        curl -L https://github.com/aws/aws-lambda-cpp/archive/v0.2.7.tar.gz | tar -xz && \
        cd aws-lambda-cpp-0.2.7 && \
        mkdir build && cd build && \
        cmake3 .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local && \
        make -j\$(nproc) && make install && \
        cd /workspace && \
        
        # Install AWS SDK for C++
        curl -L https://github.com/aws/aws-sdk-cpp/archive/1.9.379.tar.gz | tar -xz && \
        cd aws-sdk-cpp-1.9.379 && \
        mkdir build && cd build && \
        cmake3 .. -DCMAKE_BUILD_TYPE=Release \
                 -DBUILD_ONLY='ce;sns;s3;ec2' \
                 -DENABLE_TESTING=OFF \
                 -DCMAKE_INSTALL_PREFIX=/usr/local && \
        make -j\$(nproc) && make install && \
        cd /workspace && \
        
        # Install jsoncpp
        yum install -y jsoncpp-devel && \
        
        # Build the cost optimizer
        mkdir -p build && cd build && \
        cmake3 .. -DCMAKE_BUILD_TYPE=Release && \
        make -j\$(nproc) && \
        cp cost_optimizer ../bootstrap
    "
    
    echo "Docker build completed!"
else
    echo "Docker not available, attempting local build..."
    
    # Create build directory
    mkdir -p build
    cd build

    # Configure with cmake
    cmake .. -DCMAKE_BUILD_TYPE=Release

    # Build the project
    make -j$(nproc)

    # Copy the executable
    cp cost_optimizer ../bootstrap

    cd ..
fi

# Make it executable
chmod +x bootstrap

# Create deployment package
zip -r cost_optimizer.zip bootstrap

echo "Build completed successfully!"
echo "Deployment package created: cost_optimizer.zip"