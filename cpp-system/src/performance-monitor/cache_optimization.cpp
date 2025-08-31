#include "performance-monitor/cache_optimization.hpp"
#include "common/logger.hpp"

#include <chrono>
#include <random>
#include <algorithm>

#ifdef __linux__
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <sys/ioctl.h>
#endif

namespace ultra_cpp {
namespace performance {

#ifdef __linux__
struct CacheAnalyzer::Impl {
    struct PerfCounter {
        int fd = -1;
        uint64_t config;
        std::string name;
    };
    
    std::vector<PerfCounter> counters;
    bool monitoring = false;
    
    Impl() {
        // Initialize performance counters
        counters = {
            {-1, PERF_COUNT_HW_CACHE_L1D | (PERF_COUNT_HW_CACHE_OP_READ << 8) | 
                 (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16), "L1D_READ_ACCESS"},
            {-1, PERF_COUNT_HW_CACHE_L1D | (PERF_COUNT_HW_CACHE_OP_READ << 8) | 
                 (PERF_COUNT_HW_CACHE_RESULT_MISS << 16), "L1D_READ_MISS"},
            {-1, PERF_COUNT_HW_CACHE_LL | (PERF_COUNT_HW_CACHE_OP_READ << 8) | 
                 (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16), "LL_READ_ACCESS"},
            {-1, PERF_COUNT_HW_CACHE_LL | (PERF_COUNT_HW_CACHE_OP_READ << 8) | 
                 (PERF_COUNT_HW_CACHE_RESULT_MISS << 16), "LL_READ_MISS"},
            {-1, PERF_COUNT_HW_CACHE_DTLB | (PERF_COUNT_HW_CACHE_OP_READ << 8) | 
                 (PERF_COUNT_HW_CACHE_RESULT_MISS << 16), "DTLB_READ_MISS"},
            {-1, PERF_COUNT_HW_BRANCH_MISSES, "BRANCH_MISSES"}
        };
    }
    
    ~Impl() {
        stop_monitoring();
    }
    
    bool start_monitoring() {
        if (monitoring) return true;
        
        for (auto& counter : counters) {
            struct perf_event_attr pe = {};
            pe.type = PERF_TYPE_HW_CACHE;
            pe.size = sizeof(struct perf_event_attr);
            pe.config = counter.config;
            pe.disabled = 1;
            pe.exclude_kernel = 1;
            pe.exclude_hv = 1;
            
            if (counter.name == "BRANCH_MISSES") {
                pe.type = PERF_TYPE_HARDWARE;
            }
            
            counter.fd = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
            if (counter.fd == -1) {
                LOG_WARN("Failed to open perf counter for {}: {}", counter.name, strerror(errno));
                continue;
            }
            
            ioctl(counter.fd, PERF_EVENT_IOC_RESET, 0);
            ioctl(counter.fd, PERF_EVENT_IOC_ENABLE, 0);
        }
        
        monitoring = true;
        return true;
    }
    
    void stop_monitoring() {
        if (!monitoring) return;
        
        for (auto& counter : counters) {
            if (counter.fd != -1) {
                ioctl(counter.fd, PERF_EVENT_IOC_DISABLE, 0);
                close(counter.fd);
                counter.fd = -1;
            }
        }
        
        monitoring = false;
    }
    
    CacheStats get_stats() const {
        CacheStats stats;
        
        for (const auto& counter : counters) {
            if (counter.fd == -1) continue;
            
            uint64_t count;
            if (read(counter.fd, &count, sizeof(count)) == sizeof(count)) {
                if (counter.name == "L1D_READ_ACCESS") {
                    stats.l1_hits = count;
                } else if (counter.name == "L1D_READ_MISS") {
                    stats.l1_misses = count;
                    if (stats.l1_hits > count) {
                        stats.l1_hits -= count;
                    }
                } else if (counter.name == "LL_READ_ACCESS") {
                    stats.l3_hits = count;
                } else if (counter.name == "LL_READ_MISS") {
                    stats.l3_misses = count;
                    if (stats.l3_hits > count) {
                        stats.l3_hits -= count;
                    }
                } else if (counter.name == "DTLB_READ_MISS") {
                    stats.tlb_misses = count;
                } else if (counter.name == "BRANCH_MISSES") {
                    stats.branch_mispredictions = count;
                }
            }
        }
        
        return stats;
    }
    
    void reset_stats() {
        for (const auto& counter : counters) {
            if (counter.fd != -1) {
                ioctl(counter.fd, PERF_EVENT_IOC_RESET, 0);
            }
        }
    }
};
#else
struct CacheAnalyzer::Impl {
    bool start_monitoring() { return false; }
    void stop_monitoring() {}
    CacheStats get_stats() const { return CacheStats{}; }
    void reset_stats() {}
};
#endif

CacheAnalyzer::CacheAnalyzer() : impl_(std::make_unique<Impl>()) {}

CacheAnalyzer::~CacheAnalyzer() = default;

void CacheAnalyzer::start_monitoring() {
    impl_->start_monitoring();
}

void CacheAnalyzer::stop_monitoring() {
    impl_->stop_monitoring();
}

CacheAnalyzer::CacheStats CacheAnalyzer::get_stats() const {
    return impl_->get_stats();
}

void CacheAnalyzer::reset_stats() {
    impl_->reset_stats();
}

template<typename T>
double CacheAnalyzer::benchmark_sequential_access(const std::vector<T>& data, size_t iterations) {
    if (data.empty()) return 0.0;
    
    volatile T sum = T{};
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t iter = 0; iter < iterations; ++iter) {
        for (size_t i = 0; i < data.size(); ++i) {
            sum += data[i];
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    
    // Prevent optimization
    volatile T dummy = sum;
    (void)dummy;
    
    return static_cast<double>(duration.count()) / (iterations * data.size());
}

template<typename T>
double CacheAnalyzer::benchmark_random_access(const std::vector<T>& data, size_t iterations) {
    if (data.empty()) return 0.0;
    
    // Generate random indices
    std::vector<size_t> indices(data.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(indices.begin(), indices.end(), gen);
    
    volatile T sum = T{};
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t iter = 0; iter < iterations; ++iter) {
        for (size_t idx : indices) {
            sum += data[idx];
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    
    // Prevent optimization
    volatile T dummy = sum;
    (void)dummy;
    
    return static_cast<double>(duration.count()) / (iterations * data.size());
}

template<typename T>
double CacheAnalyzer::benchmark_strided_access(const std::vector<T>& data, size_t stride, size_t iterations) {
    if (data.empty() || stride == 0) return 0.0;
    
    volatile T sum = T{};
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t iter = 0; iter < iterations; ++iter) {
        for (size_t i = 0; i < data.size(); i += stride) {
            sum += data[i];
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    
    // Prevent optimization
    volatile T dummy = sum;
    (void)dummy;
    
    size_t elements_accessed = (data.size() + stride - 1) / stride;
    return static_cast<double>(duration.count()) / (iterations * elements_accessed);
}

// Explicit template instantiations for common types
template double CacheAnalyzer::benchmark_sequential_access<int>(const std::vector<int>&, size_t);
template double CacheAnalyzer::benchmark_sequential_access<double>(const std::vector<double>&, size_t);
template double CacheAnalyzer::benchmark_sequential_access<uint64_t>(const std::vector<uint64_t>&, size_t);

template double CacheAnalyzer::benchmark_random_access<int>(const std::vector<int>&, size_t);
template double CacheAnalyzer::benchmark_random_access<double>(const std::vector<double>&, size_t);
template double CacheAnalyzer::benchmark_random_access<uint64_t>(const std::vector<uint64_t>&, size_t);

template double CacheAnalyzer::benchmark_strided_access<int>(const std::vector<int>&, size_t, size_t);
template double CacheAnalyzer::benchmark_strided_access<double>(const std::vector<double>&, size_t, size_t);
template double CacheAnalyzer::benchmark_strided_access<uint64_t>(const std::vector<uint64_t>&, size_t, size_t);

} // namespace performance
} // namespace ultra_cpp