# GPU Compute Integration

This module provides ultra-low latency GPU acceleration for the C++ system, including ML inference, image processing, and cryptographic operations.

## Features

### 1. CUDA Development Environment
- **CUDA Runtime Integration**: Full CUDA runtime support with error handling
- **TensorRT Support**: ML inference acceleration with TensorRT (when available)
- **cuBLAS Integration**: High-performance linear algebra operations
- **cuRAND Integration**: GPU-accelerated random number generation
- **Multi-GPU Support**: Configurable device selection and management

### 2. GPU Memory Pool
- **Efficient Allocation**: Custom memory pool with O(1) allocation/deallocation
- **Alignment Support**: Configurable memory alignment for optimal performance
- **Fragmentation Management**: Automatic block merging and splitting
- **Thread Safety**: Lock-free operations where possible
- **Memory Statistics**: Real-time memory usage tracking

### 3. ML Inference Pipeline
- **Model Caching**: Efficient TensorRT model loading and caching
- **Batch Processing**: Optimized batch inference for throughput
- **Dynamic Shapes**: Support for variable input sizes
- **Multiple Precisions**: FP32, FP16, and INT8 inference support
- **Kernel Fusion**: Automatic optimization of inference graphs

### 4. Image Processing
- **Resize Operations**: High-performance bilinear interpolation
- **Filter Operations**: Gaussian blur, edge detection, and custom filters
- **Color Space Conversion**: RGB ↔ YUV conversion with SIMD optimization
- **Batch Processing**: Process multiple images simultaneously
- **Memory Optimization**: Zero-copy operations where possible

### 5. Cryptographic Acceleration
- **Hash Functions**: GPU-accelerated SHA-256 batch processing
- **Encryption**: AES encryption with batch operations
- **Random Generation**: Hardware-accelerated random number generation
- **Constant Time**: Side-channel resistant implementations

## Requirements

### System Requirements
- NVIDIA GPU with Compute Capability 7.5+ (RTX 20xx series or newer)
- CUDA Toolkit 12.0 or later
- CMake 3.20 or later
- C++20 compatible compiler (GCC 10+, Clang 12+)

### Dependencies
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install nvidia-cuda-toolkit cmake build-essential

# CentOS/RHEL
sudo yum install cuda cmake gcc-c++

# macOS (if CUDA support available)
brew install cmake
# Install CUDA from NVIDIA website
```

### Optional Dependencies
- TensorRT 8.0+ (for ML inference acceleration)
- Google Test (for running tests)
- Google Benchmark (for performance testing)

## Building

### Basic Build
```bash
cd cpp-system
mkdir build
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

### Build with TensorRT
```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_TENSORRT=ON \
    -DTensorRT_ROOT=/path/to/tensorrt
cmake --build build --parallel
```

### Build Options
- `ENABLE_TENSORRT`: Enable TensorRT support (default: AUTO)
- `CUDA_ARCHITECTURES`: Target GPU architectures (default: 75;80;86;89)
- `ENABLE_GPU_TESTS`: Build GPU compute tests (default: ON)
- `ENABLE_GPU_EXAMPLES`: Build GPU examples (default: ON)

## Usage

### Basic GPU Engine Setup
```cpp
#include "gpu-compute/gpu_compute_engine.hpp"

using namespace ultra::gpu;

// Configure GPU engine
GPUComputeEngine::Config config;
config.device_id = 0;
config.memory_pool_size = 512 * 1024 * 1024; // 512MB
config.enable_tensorrt = true;

// Create and initialize
GPUComputeEngine engine(config);
if (!engine.initialize()) {
    throw std::runtime_error("Failed to initialize GPU engine");
}
```

### ML Inference
```cpp
// Load a model
MLModelCache model_cache("/path/to/models");
model_cache.load_model("my_model", "/path/to/model.trt");

// Single inference
std::vector<float> input(1024, 1.0f);
auto result = engine.infer("my_model", input);

// Batch inference
std::vector<std::vector<float>> batch_inputs = {input1, input2, input3};
auto batch_results = engine.infer_batch("my_model", batch_inputs);
```

### Image Processing
```cpp
// Prepare image data
ImageData input_image;
input_image.width = 1920;
input_image.height = 1080;
input_image.channels = 3;
input_image.data = /* GPU memory pointer */;

std::vector<ImageData> inputs = {input_image};
std::vector<ImageData> outputs;

// Resize images
engine.resize_image_batch(inputs, outputs, 512, 512);

// Apply filters
engine.apply_filter_batch(inputs, outputs, "gaussian_blur");
```

### Cryptographic Operations
```cpp
// Hash multiple data blocks
std::vector<std::vector<uint8_t>> data = {
    {0x01, 0x02, 0x03},
    {0x04, 0x05, 0x06}
};

auto hashes = engine.compute_hash_batch(data);

// Encrypt data
std::vector<uint8_t> key(32, 0x42);
std::vector<std::vector<uint8_t>> ciphertexts;
engine.encrypt_batch(data, ciphertexts, key);
```

### Memory Management
```cpp
// Allocate GPU memory
void* gpu_ptr = engine.allocate_gpu_memory(1024 * 1024); // 1MB

// Use the memory...

// Free GPU memory
engine.free_gpu_memory(gpu_ptr);
```

## Performance Optimization

### Memory Pool Tuning
```cpp
// Configure memory pool for your workload
config.memory_pool_size = 2 * 1024 * 1024 * 1024; // 2GB for large workloads

// Use appropriate alignment for your data
void* aligned_ptr = memory_pool.allocate(size, 512); // 512-byte alignment
```

### Batch Size Optimization
```cpp
// Optimize batch size for your GPU
config.max_batch_size = 64; // Larger batches for high-end GPUs

// Process in optimal batch sizes
const int optimal_batch = 32;
for (size_t i = 0; i < inputs.size(); i += optimal_batch) {
    auto batch_end = std::min(i + optimal_batch, inputs.size());
    std::vector<std::vector<float>> batch(inputs.begin() + i, inputs.begin() + batch_end);
    auto results = engine.infer_batch("model", batch);
}
```

### Stream Optimization
```cpp
// Use multiple CUDA streams for overlapping operations
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// Overlap memory transfers and computation
// (Implementation details in the engine)
```

## Monitoring and Debugging

### Statistics Collection
```cpp
// Get performance statistics
auto stats = engine.get_stats();
std::cout << "Operations completed: " << stats.operations_completed << std::endl;
std::cout << "Average processing time: " 
          << stats.total_processing_time_ns / stats.operations_completed << " ns" << std::endl;

// Get device information
auto device_info = engine.get_device_info();
std::cout << "GPU: " << device_info.name << std::endl;
std::cout << "Free memory: " << device_info.free_memory / (1024*1024) << " MB" << std::endl;
```

### Error Handling
```cpp
try {
    auto result = engine.infer("model", input);
} catch (const std::runtime_error& e) {
    LOG_ERROR("GPU operation failed: {}", e.what());
    // Implement fallback logic
}
```

## Testing

### Running Tests
```bash
# Build and run all tests
cmake --build build --target test_gpu_compute
./build/tests/test_gpu_compute

# Run specific test categories
./build/tests/test_gpu_compute --gtest_filter="GPUComputeTest.*"
./build/tests/test_gpu_compute --gtest_filter="CUDAMemoryPoolTest.*"
```

### Performance Benchmarks
```bash
# Run GPU compute demo
./build/examples/gpu_compute_demo

# Run performance benchmarks
./build/tests/benchmark_gpu_compute
```

## Configuration

### Runtime Configuration
The GPU compute engine can be configured via the `gpu-compute.conf` file:

```ini
[gpu]
device_id = 0
memory_pool_size = 536870912  # 512MB
enable_tensorrt = true

[performance]
enable_profiling = true
collect_hardware_counters = true
```

### Environment Variables
- `CUDA_VISIBLE_DEVICES`: Limit visible GPU devices
- `CUDA_CACHE_PATH`: CUDA kernel cache directory
- `TENSORRT_VERBOSE`: Enable TensorRT verbose logging

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```cpp
   // Reduce memory pool size or batch size
   config.memory_pool_size = 256 * 1024 * 1024; // 256MB
   config.max_batch_size = 16;
   ```

2. **TensorRT Model Loading Fails**
   ```cpp
   // Check model compatibility and paths
   if (!model_cache.load_model("model", "/correct/path/model.trt")) {
       LOG_ERROR("Failed to load TensorRT model");
   }
   ```

3. **Performance Issues**
   ```cpp
   // Enable profiling to identify bottlenecks
   config.enable_profiling = true;
   
   // Check GPU utilization
   auto stats = engine.get_stats();
   if (stats.gpu_utilization_percent < 80) {
       // Increase batch size or optimize kernels
   }
   ```

### Debug Build
```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug -DENABLE_GPU_DEBUG=ON
```

## Integration with Node.js System

The GPU compute engine integrates with the existing Node.js system through:

1. **HTTP API**: RESTful endpoints for GPU operations
2. **Shared Memory**: Zero-copy data sharing where possible
3. **Message Queues**: Asynchronous operation requests
4. **Metrics Export**: Prometheus-compatible metrics

See the main system documentation for integration details.

## Performance Characteristics

### Expected Performance (RTX 4090)
- **ML Inference**: 10,000+ inferences/second (small models)
- **Image Resize**: 1,000+ images/second (1080p → 512x512)
- **SHA-256 Hashing**: 100+ GB/s throughput
- **Memory Allocation**: Sub-microsecond allocation/deallocation

### Latency Targets
- **Memory Operations**: < 1μs
- **Small ML Inference**: < 100μs
- **Image Processing**: < 500μs
- **Crypto Operations**: < 10μs per operation

## Future Enhancements

- [ ] Multi-GPU support with automatic load balancing
- [ ] RDMA integration for cluster computing
- [ ] Custom CUDA kernel compilation at runtime
- [ ] Integration with NVIDIA Triton Inference Server
- [ ] Support for AMD ROCm and Intel oneAPI