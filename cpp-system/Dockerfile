# Multi-stage Docker build for ultra-low-latency C++ system
FROM ubuntu:22.04 as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    wget \
    curl \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install DPDK dependencies
RUN apt-get update && apt-get install -y \
    libdpdk-dev \
    libnuma-dev \
    libpcap-dev \
    && rm -rf /var/lib/apt/lists/*

# Install CUDA toolkit
FROM base as cuda-base
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb \
    && apt-get update \
    && apt-get install -y cuda-toolkit-12-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Conan package manager
FROM cuda-base as conan-base
RUN pip3 install conan==2.0.13
RUN conan profile detect --force

# Build stage
FROM conan-base as builder
WORKDIR /src
COPY conanfile.py .
RUN conan install . --build=missing -s build_type=Release

COPY . .
RUN cmake --preset conan-release
RUN cmake --build --preset conan-release

# Runtime stage
FROM ubuntu:22.04 as runtime
RUN apt-get update && apt-get install -y \
    libdpdk22 \
    libnuma1 \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Copy binaries and configuration
COPY --from=builder /src/build/Release/bin/* /usr/local/bin/
COPY --from=builder /src/config/* /etc/ultra-cpp/

# Create non-root user
RUN useradd -r -s /bin/false ultra-cpp

# Set up huge pages
RUN mkdir -p /dev/hugepages
RUN echo 'vm.nr_hugepages=1024' >> /etc/sysctl.conf

EXPOSE 8080 9090
USER ultra-cpp

CMD ["/usr/local/bin/ultra-api-gateway"]