#!/bin/bash
# Development environment setup script for Ultra Low-Latency C++ System

set -e

echo "Setting up Ultra Low-Latency C++ development environment..."

# Check if running as root for system-level configurations
if [[ $EUID -eq 0 ]]; then
    echo "Setting up system-level configurations..."
    
    # Configure huge pages
    echo 'vm.nr_hugepages=1024' >> /etc/sysctl.conf
    echo 'vm.hugetlb_shm_group=1000' >> /etc/sysctl.conf
    sysctl -p
    
    # Create huge pages mount point
    mkdir -p /dev/hugepages
    mount -t hugetlbfs nodev /dev/hugepages
    
    # Add to fstab for persistence
    echo 'nodev /dev/hugepages hugetlbfs defaults 0 0' >> /etc/fstab
    
    echo "System-level setup complete. Please run this script again as regular user."
    exit 0
fi

# Install Conan if not present
if ! command -v conan &> /dev/null; then
    echo "Installing Conan package manager..."
    pip3 install --user conan==2.0.13
    export PATH="$HOME/.local/bin:$PATH"
fi

# Create Conan profile
echo "Setting up Conan profile..."
conan profile detect --force

# Install dependencies
echo "Installing C++ dependencies with Conan..."
conan install . --build=missing -s build_type=Debug
conan install . --build=missing -s build_type=Release

# Set up CMake presets
echo "Configuring CMake..."
cmake --preset conan-debug
cmake --preset conan-release

echo "Development environment setup complete!"
echo ""
echo "To build the project:"
echo "  Debug:   cmake --build --preset conan-debug"
echo "  Release: cmake --build --preset conan-release"
echo ""
echo "To run tests:"
echo "  cd build/Debug && ctest"