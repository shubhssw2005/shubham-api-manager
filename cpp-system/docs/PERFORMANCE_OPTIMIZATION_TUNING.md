# Performance Optimization and Tuning

This document describes the comprehensive performance optimization and tuning system implemented in the ultra-low-latency C++ system. The system provides multiple layers of optimization including CPU affinity management, cache optimization, compiler optimization profiles, and runtime adaptive tuning.

## Overview

The performance optimization system consists of four main components:

1. **CPU Affinity and NUMA Optimization** - Hardware-level thread and memory placement
2. **Cache-Friendly Data Structures** - Memory layout optimization for cache efficiency
3. **Compiler Optimization Profiles** - Build-time optimization with PGO and BOLT support
4. **Runtime Adaptive Tuning** - Dynamic performance parameter adjustment

## CPU Affinity and NUMA Optimization

### Features

- **Automatic topology discovery** - Detects CPU cores, NUMA nodes, and hyperthread siblings
- **Multiple affinity strategies** - Single core, NUMA-local, performance cores, isolated cores
- **NUMA memory policies** - Local allocation, interleaving, and strict binding
- **Thread pool optimization** - Optimal CPU assignment for worker threads

### Usage Example

```cpp
#include "performance-monitor/cpu_affinity.hpp"

CPUAffinityManager manager;
manager.initialize();

// Set NUMA-local affinity
CPUAffinityManager::AffinityConfig config;
config.strategy = CPUAffinityManager::AffinityStrategy::NUMA_LOCAL;
config.avoid_hyperthread_siblings = true;

manager.set_thread_affinity(config);

// Launch performance-aware thread
auto thread = PerformanceThread::launch_with_affinity(
    config,
    []() {
        // High-performance workload
    }
);
```

### Affinity Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `NONE` | No affinity setting | Default behavior |
| `SINGLE_CORE` | Pin to single CPU core | Latency-critical tasks |
| `NUMA_LOCAL` | Pin to cores in same NUMA node | Memory-intensive workloads |
| `PERFORMANCE_CORES` | Avoid hyperthread siblings | CPU-intensive tasks |
| `ISOLATED_CORES` | Use isolated CPUs if available | Real-time applications |
| `CUSTOM` | User-defined CPU set | Specialized requirements |

## Cache-Friendly Data Structures

### Features

- **Cache-aligned allocators** - Automatic alignment to cache line boundaries
- **Prefetching support** - Hardware prefetch hints for better cache utilization
- **Structure of Arrays (SoA)** - Improved cache locality for bulk operations
- **Lock-free hash tables** - High-performance concurrent data structures

### Cache-Friendly Vector

```cpp
#include "performance-monitor/cache_optimization.hpp"

CacheFriendlyVector<int> vec;

// Automatic cache line alignment and prefetching
for (size_t i = 0; i < 1000000; ++i) {
    vec.push_back(i);
}

// Optimized access with prefetching
for (size_t i = 0; i < vec.size(); ++i) {
    process(vec[i]); // Automatic prefetch of next cache line
}
```

### Structure of Arrays

```cpp
// Instead of Array of Structures (AoS)
struct Particle { float x, y, z, mass; };
std::vector<Particle> particles;

// Use Structure of Arrays (SoA) for better cache locality
StructureOfArrays<float, float, float, float> particles_soa;

// Bulk operations on single component (better cache utilization)
auto& x_array = particles_soa.get_array<0>();
for (size_t i = 0; i < x_array.size(); ++i) {
    x_array[i] += velocity_x[i] * dt;
}
```

### Cache Performance Analysis

```cpp
CacheAnalyzer analyzer;
analyzer.start_monitoring();

// Run workload
perform_workload();

analyzer.stop_monitoring();
auto stats = analyzer.get_stats();

std::cout << "L1 hit rate: " << stats.l1_hit_rate() * 100 << "%\n";
std::cout << "L2 hit rate: " << stats.l2_hit_rate() * 100 << "%\n";
std::cout << "L3 hit rate: " << stats.l3_hit_rate() * 100 << "%\n";
```

## Compiler Optimization Profiles

### Features

- **Multiple optimization levels** - Debug, standard, aggressive, size, fast
- **Profile-Guided Optimization (PGO)** - Training-based optimization
- **Link Time Optimization (LTO)** - Cross-module optimization
- **BOLT support** - Binary layout optimization
- **Architecture targeting** - Native, Haswell, Skylake, Zen2, Zen3, ARM

### Built-in Profiles

| Profile | Optimization | Features | Use Case |
|---------|-------------|----------|----------|
| `debug` | `-O0 -g` | Debug symbols, no optimization | Development |
| `release` | `-O3 -march=native -flto` | Aggressive + LTO | Production |
| `performance` | `-O3 -march=native -flto -fprofile-use` | Aggressive + LTO + PGO | High performance |
| `size` | `-Os -flto` | Size optimization + LTO | Memory-constrained |
| `ultra` | `-Ofast -march=native -flto -fprofile-use -bolt` | All optimizations | Maximum performance |

### PGO Workflow

```cpp
CompilerOptimizationManager manager;

// 1. Generate profile
CompilerOptimizationManager::PGOWorkflow workflow;
workflow.profile_name = "my_app";
workflow.source_dir = "/path/to/source";
workflow.build_dir = "/path/to/build";
workflow.benchmark_executable = "./benchmark";
workflow.profile_output_dir = "/path/to/profiles";

manager.generate_pgo_profile(workflow);

// 2. Build with profile
manager.build_with_pgo(workflow);
```

### BOLT Optimization

```cpp
BOLTHelper bolt_helper;

// Collect performance profile
bolt_helper.collect_perf_profile(
    "./my_binary",
    "./run_workload.sh",
    "perf.data"
);

// Optimize binary layout
bolt_helper.optimize_binary(
    "./my_binary",
    "perf.data",
    "./my_binary_optimized"
);
```

## Runtime Adaptive Tuning

### Features

- **Multiple tuning algorithms** - Gradient descent, simulated annealing, genetic algorithm, Bayesian optimization
- **Performance objectives** - Throughput, latency, CPU usage, memory usage, cache hit rate
- **Parameter management** - Registration, bounds checking, sensitivity weighting
- **Metrics collection** - Real-time performance monitoring

### Basic Usage

```cpp
AdaptiveTuningSystem::TuningConfig config;
config.objective = AdaptiveTuningSystem::OptimizationObjective::MAXIMIZE_THROUGHPUT;
config.tuning_interval = std::chrono::seconds(10);

AdaptiveTuningSystem tuning_system(config);

// Register tunable parameters
AdaptiveTuningSystem::TunableParameter thread_count_param;
thread_count_param.name = "thread_count";
thread_count_param.current_value = 4.0;
thread_count_param.min_value = 1.0;
thread_count_param.max_value = 16.0;
thread_count_param.setter = [](double value) { 
    set_worker_thread_count(static_cast<int>(value)); 
};

tuning_system.register_parameter(thread_count_param);

// Start auto-tuning
tuning_system.start_auto_tuning();

// Update metrics periodically
AdaptiveTuningSystem::PerformanceMetrics metrics;
metrics.throughput.store(current_throughput);
metrics.latency_p95.store(current_p95_latency);
metrics.cpu_utilization.store(current_cpu_usage);

tuning_system.update_metrics(metrics);
```

### Tuning Algorithms

| Algorithm | Description | Best For |
|-----------|-------------|----------|
| `GRADIENT_DESCENT` | Local optimization using gradients | Smooth parameter spaces |
| `SIMULATED_ANNEALING` | Global optimization with cooling | Noisy or multi-modal spaces |
| `GENETIC_ALGORITHM` | Population-based evolution | Complex parameter interactions |
| `BAYESIAN_OPTIMIZATION` | Model-based optimization | Expensive function evaluations |
| `REINFORCEMENT_LEARNING` | Learning-based adaptation | Dynamic environments |

### Adaptive Components

#### Memory Allocation Tuner

```cpp
MemoryAllocationTuner tuner;

// Record allocations
void* ptr = malloc(size);
tuner.record_allocation(size, ptr);

// Record deallocations
tuner.record_deallocation(ptr);
free(ptr);

// Enable adaptive pool sizing
tuner.enable_adaptive_pools(true);
tuner.tune_pool_sizes();
```

#### Cache Tuner

```cpp
CacheTuner tuner;

// Register cache configuration
CacheTuner::CacheConfig config;
config.capacity = 1000000;
config.shard_count = 64;
config.eviction_threshold = 0.8;

tuner.register_cache("my_cache", config);

// Record cache accesses
tuner.record_cache_access("my_cache", "key1", true);  // hit
tuner.record_cache_access("my_cache", "key2", false); // miss

// Enable adaptive sizing
tuner.enable_adaptive_sizing("my_cache", true);
```

#### Network Tuner

```cpp
NetworkTuner tuner;

// Configure network parameters
NetworkTuner::NetworkConfig config;
config.buffer_size = 65536;
config.connection_pool_size = 100;
config.connection_timeout = std::chrono::seconds(5);

tuner.set_network_config(config);

// Enable adaptive tuning
tuner.enable_adaptive_tuning(true);
```

## Integrated Performance Management

### Performance Tuning Manager

The `PerformanceTuningManager` provides unified access to all tuning components:

```cpp
PerformanceTuningManager manager;

// Access individual components
auto& adaptive_tuning = manager.get_adaptive_tuning_system();
auto& scheduler = manager.get_adaptive_scheduler();
auto& memory_tuner = manager.get_memory_tuner();
auto& cache_tuner = manager.get_cache_tuner();
auto& network_tuner = manager.get_network_tuner();

// Start all tuning systems
manager.start_all_tuning();

// Get system-wide metrics
auto metrics = manager.get_system_metrics();

// Save/load performance profiles
manager.save_performance_profile("high_throughput");
manager.load_performance_profile("low_latency");
```

## Performance Measurement and Analysis

### Benchmarking

```cpp
// Cache performance benchmarks
std::vector<int> data(1000000);
std::iota(data.begin(), data.end(), 0);

double seq_time = CacheAnalyzer::benchmark_sequential_access(data, 100);
double rand_time = CacheAnalyzer::benchmark_random_access(data, 100);
double strided_time = CacheAnalyzer::benchmark_strided_access(data, 8, 100);

std::cout << "Sequential: " << seq_time << " ns/element\n";
std::cout << "Random: " << rand_time << " ns/element\n";
std::cout << "Strided: " << strided_time << " ns/element\n";
```

### Hardware Performance Counters

The system integrates with hardware performance counters (PMU) on Linux systems to provide detailed performance insights:

- L1/L2/L3 cache hits and misses
- TLB misses
- Branch mispredictions
- Instructions retired
- CPU cycles

## Best Practices

### CPU Affinity

1. **Use NUMA-local affinity** for memory-intensive workloads
2. **Avoid hyperthread siblings** for CPU-intensive tasks
3. **Pin interrupt handlers** to separate cores when possible
4. **Use isolated cores** for real-time applications

### Cache Optimization

1. **Align data structures** to cache line boundaries
2. **Use Structure of Arrays** for bulk operations
3. **Implement prefetching** for predictable access patterns
4. **Minimize false sharing** between threads

### Compiler Optimization

1. **Use PGO** for production builds with representative workloads
2. **Enable LTO** for cross-module optimization
3. **Target specific architectures** when deployment is known
4. **Use BOLT** for final binary optimization

### Adaptive Tuning

1. **Start with conservative parameters** and let the system adapt
2. **Monitor multiple metrics** to avoid local optima
3. **Use appropriate algorithms** for your parameter space
4. **Validate tuning results** with A/B testing

## Configuration Examples

### High-Throughput Configuration

```cpp
// CPU affinity for maximum parallelism
CPUAffinityManager::AffinityConfig cpu_config;
cpu_config.strategy = CPUAffinityManager::AffinityStrategy::PERFORMANCE_CORES;
cpu_config.avoid_hyperthread_siblings = false; // Use all logical cores

// Adaptive tuning for throughput
AdaptiveTuningSystem::TuningConfig tuning_config;
tuning_config.objective = AdaptiveTuningSystem::OptimizationObjective::MAXIMIZE_THROUGHPUT;
tuning_config.learning_rate = 0.2; // Aggressive learning

// Large cache for high hit rates
CacheTuner::CacheConfig cache_config;
cache_config.capacity = 10000000;
cache_config.shard_count = 128;
```

### Low-Latency Configuration

```cpp
// CPU affinity for latency
CPUAffinityManager::AffinityConfig cpu_config;
cpu_config.strategy = CPUAffinityManager::AffinityStrategy::ISOLATED_CORES;
cpu_config.avoid_hyperthread_siblings = true;

// Adaptive tuning for latency
AdaptiveTuningSystem::TuningConfig tuning_config;
tuning_config.objective = AdaptiveTuningSystem::OptimizationObjective::MINIMIZE_LATENCY;
tuning_config.learning_rate = 0.05; // Conservative learning

// Small, fast cache
CacheTuner::CacheConfig cache_config;
cache_config.capacity = 100000;
cache_config.shard_count = 16;
cache_config.enable_prefetching = true;
```

## Monitoring and Debugging

### Performance Metrics

The system provides comprehensive metrics for monitoring:

```cpp
// Adaptive tuning metrics
auto tuning_metrics = adaptive_tuning.get_current_metrics();
std::cout << "Throughput: " << tuning_metrics.throughput.load() << " ops/sec\n";
std::cout << "P95 Latency: " << tuning_metrics.latency_p95.load() << " ns\n";

// Cache metrics
auto cache_metrics = cache_tuner.get_cache_metrics("my_cache");
std::cout << "Hit rate: " << cache_metrics.hit_rate.load() * 100 << "%\n";

// Memory metrics
auto memory_stats = memory_tuner.get_stats();
std::cout << "Allocations: " << memory_stats.total_allocations.load() << "\n";
```

### Debugging Tools

1. **Cache analyzer** - Identify cache performance issues
2. **Tuning history** - Track parameter changes and their effects
3. **Performance counters** - Hardware-level performance insights
4. **Compiler reports** - PGO and LTO optimization feedback

## Platform Support

### Linux

- Full support for all features
- NUMA topology detection
- Hardware performance counters
- CPU isolation and affinity
- perf integration for BOLT

### Other Platforms

- Basic CPU affinity support
- Cache-friendly data structures
- Compiler optimization profiles
- Limited adaptive tuning

## Dependencies

### Required

- C++20 compiler (GCC 10+, Clang 12+)
- CMake 3.20+
- Threads library

### Optional

- libnuma (NUMA support)
- Linux perf tools (BOLT support)
- CUDA toolkit (GPU optimization)
- Google Test (testing)

## Building with Optimization

```bash
# Configure with performance optimizations
cmake -B build -DCMAKE_BUILD_TYPE=Release \
               -DCMAKE_CXX_FLAGS="-O3 -march=native -flto" \
               -DENABLE_PGO=ON \
               -DENABLE_BOLT=ON

# Build
cmake --build build -j$(nproc)

# Run PGO training
./build/examples/performance_optimization_demo

# Rebuild with PGO
cmake --build build -j$(nproc)
```

This comprehensive performance optimization and tuning system provides the foundation for achieving ultra-low-latency performance in C++ applications through hardware-aware optimization, intelligent caching, advanced compiler techniques, and runtime adaptation.