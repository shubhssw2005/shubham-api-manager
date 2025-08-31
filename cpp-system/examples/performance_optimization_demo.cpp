#include "performance-monitor/cpu_affinity.hpp"
#include "performance-monitor/cache_optimization.hpp"
#include "performance-monitor/compiler_optimization.hpp"
#include "performance-monitor/adaptive_tuning.hpp"
#include "common/logger.hpp"

#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <random>
#include <algorithm>

using namespace ultra_cpp::performance;

// Example workload for demonstration
class WorkloadSimulator {
public:
    WorkloadSimulator(size_t data_size = 1000000) : data_size_(data_size) {
        // Initialize test data
        regular_data_.resize(data_size_);
        std::iota(regular_data_.begin(), regular_data_.end(), 0);
        
        for (size_t i = 0; i < data_size_; ++i) {
            cache_friendly_data_.push_back(static_cast<int>(i));
        }
        
        // Create random access pattern
        random_indices_.resize(data_size_);
        std::iota(random_indices_.begin(), random_indices_.end(), 0);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(random_indices_.begin(), random_indices_.end(), gen);
    }
    
    // CPU-intensive workload
    double cpu_intensive_workload(size_t iterations = 1000) {
        auto start = std::chrono::high_resolution_clock::now();
        
        volatile double result = 0.0;
        for (size_t i = 0; i < iterations; ++i) {
            for (size_t j = 0; j < data_size_ / 1000; ++j) {
                result += std::sin(static_cast<double>(j)) * std::cos(static_cast<double>(i));
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        
        return static_cast<double>(duration.count()) / iterations;
    }
    
    // Memory-intensive workload with regular vector
    double memory_workload_regular() {
        auto start = std::chrono::high_resolution_clock::now();
        
        volatile long long sum = 0;
        for (size_t idx : random_indices_) {
            sum += regular_data_[idx];
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        
        return static_cast<double>(duration.count());
    }
    
    // Memory-intensive workload with cache-friendly vector
    double memory_workload_cache_friendly() {
        auto start = std::chrono::high_resolution_clock::now();
        
        volatile long long sum = 0;
        for (size_t idx : random_indices_) {
            sum += cache_friendly_data_[idx];
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        
        return static_cast<double>(duration.count());
    }
    
    // Sequential access workload
    double sequential_access_workload() {
        auto start = std::chrono::high_resolution_clock::now();
        
        volatile long long sum = 0;
        for (size_t i = 0; i < cache_friendly_data_.size(); ++i) {
            sum += cache_friendly_data_[i];
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        
        return static_cast<double>(duration.count());
    }

private:
    size_t data_size_;
    std::vector<int> regular_data_;
    CacheFriendlyVector<int> cache_friendly_data_;
    std::vector<size_t> random_indices_;
};

void demonstrate_cpu_affinity() {
    std::cout << "\n=== CPU Affinity Demonstration ===\n";
    
    CPUAffinityManager manager;
    if (!manager.initialize()) {
        std::cout << "Failed to initialize CPU affinity manager\n";
        return;
    }
    
    const auto& topology = manager.get_topology();
    std::cout << "System topology:\n";
    std::cout << "  Logical CPUs: " << topology.total_logical_cpus << "\n";
    std::cout << "  Physical cores: " << topology.total_physical_cores << "\n";
    std::cout << "  NUMA nodes: " << topology.total_numa_nodes << "\n";
    
    if (manager.is_numa_available()) {
        std::cout << "  NUMA support: Available\n";
        std::cout << "  Current CPU: " << manager.get_current_cpu() << "\n";
        std::cout << "  Current NUMA node: " << manager.get_current_numa_node() << "\n";
    } else {
        std::cout << "  NUMA support: Not available\n";
    }
    
    // Test different affinity strategies
    WorkloadSimulator workload;
    
    std::cout << "\nTesting CPU affinity impact on performance:\n";
    
    // Baseline (no affinity)
    double baseline_time = workload.cpu_intensive_workload(100);
    std::cout << "  Baseline (no affinity): " << baseline_time << " ns/iteration\n";
    
    // Single core affinity
    CPUAffinityManager::AffinityConfig single_core_config;
    single_core_config.strategy = CPUAffinityManager::AffinityStrategy::SINGLE_CORE;
    
    if (manager.set_thread_affinity(single_core_config)) {
        double single_core_time = workload.cpu_intensive_workload(100);
        std::cout << "  Single core affinity: " << single_core_time << " ns/iteration\n";
        std::cout << "  Performance change: " << 
                     ((single_core_time - baseline_time) / baseline_time * 100) << "%\n";
    }
    
    // Performance cores affinity
    CPUAffinityManager::AffinityConfig perf_config;
    perf_config.strategy = CPUAffinityManager::AffinityStrategy::PERFORMANCE_CORES;
    perf_config.avoid_hyperthread_siblings = true;
    
    if (manager.set_thread_affinity(perf_config)) {
        double perf_time = workload.cpu_intensive_workload(100);
        std::cout << "  Performance cores: " << perf_time << " ns/iteration\n";
        std::cout << "  Performance change: " << 
                     ((perf_time - baseline_time) / baseline_time * 100) << "%\n";
    }
}

void demonstrate_cache_optimization() {
    std::cout << "\n=== Cache Optimization Demonstration ===\n";
    
    WorkloadSimulator workload;
    
    // Compare regular vs cache-friendly data structures
    std::cout << "Comparing memory access patterns:\n";
    
    double regular_time = workload.memory_workload_regular();
    double cache_friendly_time = workload.memory_workload_cache_friendly();
    double sequential_time = workload.sequential_access_workload();
    
    std::cout << "  Regular vector (random access): " << regular_time << " ns\n";
    std::cout << "  Cache-friendly vector (random access): " << cache_friendly_time << " ns\n";
    std::cout << "  Cache-friendly vector (sequential): " << sequential_time << " ns\n";
    
    double improvement = (regular_time - cache_friendly_time) / regular_time * 100;
    std::cout << "  Cache-friendly improvement: " << improvement << "%\n";
    
    double seq_improvement = (regular_time - sequential_time) / regular_time * 100;
    std::cout << "  Sequential access improvement: " << seq_improvement << "%\n";
    
    // Demonstrate cache analysis
    CacheAnalyzer analyzer;
    analyzer.start_monitoring();
    
    std::vector<int> test_data(100000);
    std::iota(test_data.begin(), test_data.end(), 0);
    
    double seq_bench = CacheAnalyzer::benchmark_sequential_access(test_data, 10);
    double rand_bench = CacheAnalyzer::benchmark_random_access(test_data, 10);
    double strided_bench = CacheAnalyzer::benchmark_strided_access(test_data, 8, 10);
    
    std::cout << "\nCache benchmark results (ns per element):\n";
    std::cout << "  Sequential access: " << seq_bench << "\n";
    std::cout << "  Random access: " << rand_bench << "\n";
    std::cout << "  Strided access (stride=8): " << strided_bench << "\n";
    
    analyzer.stop_monitoring();
    
    auto cache_stats = analyzer.get_stats();
    std::cout << "\nCache statistics:\n";
    std::cout << "  L1 hit rate: " << cache_stats.l1_hit_rate() * 100 << "%\n";
    std::cout << "  L2 hit rate: " << cache_stats.l2_hit_rate() * 100 << "%\n";
    std::cout << "  L3 hit rate: " << cache_stats.l3_hit_rate() * 100 << "%\n";
    
    // Demonstrate Structure of Arrays
    std::cout << "\nStructure of Arrays demonstration:\n";
    
    StructureOfArrays<int, double, float> soa;
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10000; ++i) {
        soa.push_back(i, static_cast<double>(i) * 1.5, static_cast<float>(i) * 2.0f);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto soa_insert_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    
    start = std::chrono::high_resolution_clock::now();
    volatile double sum = 0.0;
    for (size_t i = 0; i < soa.size(); ++i) {
        sum += soa.get<1>(i); // Access only double array
    }
    end = std::chrono::high_resolution_clock::now();
    auto soa_access_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    
    std::cout << "  SoA insertion time: " << soa_insert_time.count() << " ns\n";
    std::cout << "  SoA access time: " << soa_access_time.count() << " ns\n";
}

void demonstrate_compiler_optimization() {
    std::cout << "\n=== Compiler Optimization Demonstration ===\n";
    
    CompilerOptimizationManager manager;
    
    // Display compiler information
    auto compiler_info = manager.detect_compiler();
    std::cout << "Detected compiler: ";
    switch (compiler_info.type) {
        case CompilerOptimizationManager::CompilerType::GCC:
            std::cout << "GCC";
            break;
        case CompilerOptimizationManager::CompilerType::CLANG:
            std::cout << "Clang";
            break;
        case CompilerOptimizationManager::CompilerType::ICC:
            std::cout << "Intel C++";
            break;
        default:
            std::cout << "Unknown";
            break;
    }
    std::cout << " version " << compiler_info.version << "\n";
    
    std::cout << "Compiler capabilities:\n";
    std::cout << "  PGO support: " << (compiler_info.supports_pgo ? "Yes" : "No") << "\n";
    std::cout << "  LTO support: " << (compiler_info.supports_lto ? "Yes" : "No") << "\n";
    std::cout << "  ThinLTO support: " << (compiler_info.supports_thinlto ? "Yes" : "No") << "\n";
    std::cout << "  BOLT support: " << (compiler_info.supports_bolt ? "Yes" : "No") << "\n";
    
    // List available optimization profiles
    auto profiles = manager.list_profiles();
    std::cout << "\nAvailable optimization profiles:\n";
    for (const auto& profile_name : profiles) {
        auto profile = manager.get_profile(profile_name);
        std::cout << "  " << profile_name << ": ";
        
        switch (profile.level) {
            case CompilerOptimizationManager::OptimizationLevel::DEBUG:
                std::cout << "Debug (-O0)";
                break;
            case CompilerOptimizationManager::OptimizationLevel::STANDARD:
                std::cout << "Standard (-O2)";
                break;
            case CompilerOptimizationManager::OptimizationLevel::AGGRESSIVE:
                std::cout << "Aggressive (-O3)";
                break;
            case CompilerOptimizationManager::OptimizationLevel::FAST:
                std::cout << "Fast (-Ofast)";
                break;
            default:
                std::cout << "Other";
                break;
        }
        
        if (profile.enable_lto) std::cout << " + LTO";
        if (profile.enable_pgo) std::cout << " + PGO";
        std::cout << "\n";
    }
    
    // Generate CMake flags for different profiles
    std::cout << "\nGenerated CMake flags:\n";
    auto release_profile = manager.get_profile("release");
    std::string cmake_flags = manager.generate_cmake_flags(release_profile);
    std::cout << "  Release profile: " << cmake_flags << "\n";
    
    auto ultra_profile = manager.get_profile("ultra");
    cmake_flags = manager.generate_cmake_flags(ultra_profile);
    std::cout << "  Ultra profile: " << cmake_flags << "\n";
}

void demonstrate_adaptive_tuning() {
    std::cout << "\n=== Adaptive Tuning Demonstration ===\n";
    
    // Create tuning system with manual control
    AdaptiveTuningSystem::TuningConfig config;
    config.enable_auto_tuning = false;
    config.objective = AdaptiveTuningSystem::OptimizationObjective::MAXIMIZE_THROUGHPUT;
    
    AdaptiveTuningSystem tuning_system(config);
    
    // Register some tunable parameters
    double thread_count = 4.0;
    double batch_size = 100.0;
    double cache_size = 1000.0;
    
    AdaptiveTuningSystem::TunableParameter thread_param;
    thread_param.name = "thread_count";
    thread_param.current_value = thread_count;
    thread_param.min_value = 1.0;
    thread_param.max_value = 16.0;
    thread_param.step_size = 1.0;
    thread_param.setter = [&thread_count](double value) { thread_count = value; };
    thread_param.getter = [&thread_count]() { return thread_count; };
    
    AdaptiveTuningSystem::TunableParameter batch_param;
    batch_param.name = "batch_size";
    batch_param.current_value = batch_size;
    batch_param.min_value = 10.0;
    batch_param.max_value = 1000.0;
    batch_param.step_size = 10.0;
    batch_param.setter = [&batch_size](double value) { batch_size = value; };
    batch_param.getter = [&batch_size]() { return batch_size; };
    
    tuning_system.register_parameter(thread_param);
    tuning_system.register_parameter(batch_param);
    
    std::cout << "Registered tunable parameters:\n";
    auto params = tuning_system.list_parameters();
    for (const auto& param : params) {
        std::cout << "  " << param << ": " << tuning_system.get_parameter(param) << "\n";
    }
    
    // Simulate performance measurements and tuning
    WorkloadSimulator workload;
    
    std::cout << "\nSimulating adaptive tuning:\n";
    
    for (int iteration = 0; iteration < 5; ++iteration) {
        // Simulate workload with current parameters
        double workload_time = workload.cpu_intensive_workload(50);
        double throughput = 50.0 * 1000000000.0 / workload_time; // ops/sec
        
        // Create performance metrics
        AdaptiveTuningSystem::PerformanceMetrics metrics;
        metrics.throughput.store(throughput);
        metrics.latency_p95.store(workload_time);
        metrics.cpu_utilization.store(75.0 + (iteration * 2.0)); // Simulate increasing CPU usage
        metrics.cache_hit_rate.store(90.0 - (iteration * 1.0));  // Simulate decreasing hit rate
        
        tuning_system.update_metrics(metrics);
        
        std::cout << "  Iteration " << iteration + 1 << ":\n";
        std::cout << "    Throughput: " << throughput << " ops/sec\n";
        std::cout << "    Latency: " << workload_time << " ns\n";
        std::cout << "    Thread count: " << thread_count << "\n";
        std::cout << "    Batch size: " << batch_size << "\n";
        
        // Manual parameter adjustment (simulating adaptive algorithm)
        if (iteration < 4) {
            if (throughput < 100000) {
                tuning_system.set_parameter("thread_count", thread_count + 1);
            }
            if (workload_time > 20000) {
                tuning_system.set_parameter("batch_size", batch_size + 50);
            }
        }
    }
    
    // Demonstrate other tuning components
    std::cout << "\nOther adaptive tuning components:\n";
    
    MemoryAllocationTuner memory_tuner;
    void* test_ptr = malloc(1024);
    memory_tuner.record_allocation(1024, test_ptr);
    memory_tuner.record_deallocation(test_ptr);
    free(test_ptr);
    
    auto mem_stats = memory_tuner.get_stats();
    std::cout << "  Memory allocations: " << mem_stats.total_allocations.load() << "\n";
    std::cout << "  Memory deallocations: " << mem_stats.total_deallocations.load() << "\n";
    
    CacheTuner cache_tuner;
    cache_tuner.record_cache_access("demo_cache", "key1", true);
    cache_tuner.record_cache_access("demo_cache", "key2", false);
    cache_tuner.record_cache_access("demo_cache", "key3", true);
    
    auto cache_metrics = cache_tuner.get_cache_metrics("demo_cache");
    std::cout << "  Cache hit rate: " << cache_metrics.hit_rate.load() * 100 << "%\n";
    
    NetworkTuner network_tuner;
    NetworkTuner::NetworkConfig net_config;
    net_config.buffer_size = 65536;
    net_config.connection_pool_size = 100;
    network_tuner.set_network_config(net_config);
    
    auto net_metrics = network_tuner.get_metrics();
    std::cout << "  Network connections: " << net_metrics.connections_established.load() << "\n";
}

void demonstrate_integrated_tuning() {
    std::cout << "\n=== Integrated Performance Tuning ===\n";
    
    PerformanceTuningManager manager;
    
    // Get system metrics
    auto system_metrics = manager.get_system_metrics();
    
    std::cout << "System-wide performance metrics:\n";
    std::cout << "  Adaptive tuning throughput: " << 
                 system_metrics.adaptive_metrics.throughput.load() << " ops/sec\n";
    std::cout << "  Scheduler tasks completed: " << 
                 system_metrics.scheduler_metrics.tasks_completed.load() << "\n";
    std::cout << "  Memory allocations: " << 
                 system_metrics.memory_stats.total_allocations.load() << "\n";
    
    std::cout << "\nIntegrated tuning manager provides unified access to:\n";
    std::cout << "  - Adaptive tuning system\n";
    std::cout << "  - Adaptive scheduler\n";
    std::cout << "  - Memory allocation tuner\n";
    std::cout << "  - Cache tuner\n";
    std::cout << "  - Network tuner\n";
    
    // Demonstrate saving/loading performance profiles
    std::cout << "\nPerformance profile management:\n";
    std::cout << "  Profiles can be saved and loaded for different workloads\n";
    std::cout << "  This allows quick switching between optimized configurations\n";
}

int main() {
    std::cout << "Ultra Low-Latency C++ Performance Optimization Demo\n";
    std::cout << "==================================================\n";
    
    try {
        demonstrate_cpu_affinity();
        demonstrate_cache_optimization();
        demonstrate_compiler_optimization();
        demonstrate_adaptive_tuning();
        demonstrate_integrated_tuning();
        
        std::cout << "\n=== Demo Complete ===\n";
        std::cout << "This demonstration showed the key performance optimization features:\n";
        std::cout << "1. CPU affinity and NUMA topology optimization\n";
        std::cout << "2. Cache-friendly data structures and memory layout optimization\n";
        std::cout << "3. Compiler optimization profiles with PGO support\n";
        std::cout << "4. Runtime adaptive tuning with multiple algorithms\n";
        std::cout << "5. Integrated performance tuning management\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Demo failed with exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}