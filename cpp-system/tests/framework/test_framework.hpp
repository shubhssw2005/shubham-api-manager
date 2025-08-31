#pragma once

#include <gtest/gtest.h>
#include <benchmark/benchmark.h>
#include <chrono>
#include <memory>
#include <vector>
#include <functional>
#include <random>
#include <thread>
#include <atomic>
#include <string>
#include <map>

namespace ultra::testing {

// Performance test utilities
class PerformanceTimer {
public:
    PerformanceTimer() : start_(std::chrono::high_resolution_clock::now()) {}
    
    uint64_t elapsed_ns() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_).count();
    }
    
    double elapsed_ms() const {
        return elapsed_ns() / 1000000.0;
    }
    
    void reset() {
        start_ = std::chrono::high_resolution_clock::now();
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
};

// Latency measurement utilities
class LatencyMeasurement {
public:
    void record_latency(uint64_t latency_ns) {
        latencies_.push_back(latency_ns);
    }
    
    struct Stats {
        uint64_t min_ns;
        uint64_t max_ns;
        uint64_t p50_ns;
        uint64_t p95_ns;
        uint64_t p99_ns;
        uint64_t p999_ns;
        double mean_ns;
        double stddev_ns;
        size_t count;
    };
    
    Stats calculate_stats() const;
    void reset() { latencies_.clear(); }

private:
    std::vector<uint64_t> latencies_;
};

// Load testing utilities
class LoadGenerator {
public:
    struct Config {
        size_t thread_count = 1;
        size_t requests_per_second = 1000;
        std::chrono::seconds duration{10};
        std::function<void()> request_func;
    };
    
    explicit LoadGenerator(const Config& config);
    void run();
    
    struct Results {
        size_t total_requests;
        size_t successful_requests;
        size_t failed_requests;
        LatencyMeasurement::Stats latency_stats;
        double actual_rps;
    };
    
    Results get_results() const { return results_; }

private:
    Config config_;
    Results results_;
    std::atomic<bool> running_{false};
    std::atomic<size_t> completed_requests_{0};
    std::atomic<size_t> failed_requests_{0};
    LatencyMeasurement latency_measurement_;
    
    void worker_thread();
};

// Chaos testing utilities
class ChaosInjector {
public:
    enum class FailureType {
        MEMORY_ALLOCATION_FAILURE,
        NETWORK_TIMEOUT,
        DISK_IO_ERROR,
        CPU_SPIKE,
        THREAD_STARVATION
    };
    
    struct Config {
        FailureType type;
        double probability = 0.1; // 10% chance
        std::chrono::milliseconds duration{100};
        std::function<void()> failure_action;
    };
    
    void register_failure(const Config& config);
    void enable_chaos();
    void disable_chaos();
    bool should_inject_failure(FailureType type);
    void inject_failure(FailureType type);

private:
    std::map<FailureType, Config> failure_configs_;
    std::atomic<bool> chaos_enabled_{false};
    std::random_device rd_;
    std::mt19937 gen_{rd_()};
    std::uniform_real_distribution<> dis_{0.0, 1.0};
};

// Memory leak detection
class MemoryTracker {
public:
    static MemoryTracker& instance();
    
    void track_allocation(void* ptr, size_t size);
    void track_deallocation(void* ptr);
    
    struct Stats {
        size_t total_allocations;
        size_t total_deallocations;
        size_t current_allocations;
        size_t peak_memory_usage;
        size_t current_memory_usage;
    };
    
    Stats get_stats() const;
    void reset_stats();
    bool has_leaks() const;

private:
    MemoryTracker() = default;
    
    mutable std::mutex mutex_;
    std::map<void*, size_t> allocations_;
    Stats stats_{};
};

// Test fixtures for common scenarios
class UltraTestFixture : public ::testing::Test {
protected:
    void SetUp() override;
    void TearDown() override;
    
    // Helper methods for common test patterns
    void expect_latency_under(std::function<void()> func, uint64_t max_latency_ns);
    void expect_throughput_over(std::function<void()> func, size_t min_ops_per_second);
    void run_concurrent_test(std::function<void()> func, size_t thread_count, 
                           std::chrono::seconds duration);

protected:
    std::unique_ptr<ChaosInjector> chaos_injector_;
    std::unique_ptr<MemoryTracker> memory_tracker_;
};

// Benchmark fixtures
class UltraBenchmarkFixture : public benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State& state) override;
    void TearDown(const ::benchmark::State& state) override;

protected:
    // Helper methods for benchmark setup
    void setup_memory_pools();
    void setup_test_data(size_t size);
    void cleanup_resources();

protected:
    std::vector<uint8_t> test_data_;
    size_t data_size_{0};
};

// Macros for common test patterns
#define ULTRA_EXPECT_LATENCY_UNDER(func, max_ns) \
    do { \
        PerformanceTimer timer; \
        func; \
        EXPECT_LT(timer.elapsed_ns(), max_ns) \
            << "Operation took " << timer.elapsed_ns() << "ns, expected under " << max_ns << "ns"; \
    } while(0)

#define ULTRA_BENCHMARK_LATENCY(func) \
    do { \
        PerformanceTimer timer; \
        func; \
        state.SetIterationTime(timer.elapsed_ns() / 1e9); \
    } while(0)

#define ULTRA_CHAOS_TEST(test_name, failure_type) \
    TEST_F(UltraTestFixture, test_name##_WithChaos) { \
        chaos_injector_->register_failure({failure_type, 0.2, std::chrono::milliseconds{50}}); \
        chaos_injector_->enable_chaos(); \
        test_name(); \
        chaos_injector_->disable_chaos(); \
    }

} // namespace ultra::testing