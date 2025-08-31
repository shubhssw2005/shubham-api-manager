#include "test_framework.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <future>

namespace ultra::testing {

// LatencyMeasurement implementation
LatencyMeasurement::Stats LatencyMeasurement::calculate_stats() const {
    if (latencies_.empty()) {
        return {};
    }
    
    auto sorted = latencies_;
    std::sort(sorted.begin(), sorted.end());
    
    Stats stats;
    stats.count = sorted.size();
    stats.min_ns = sorted.front();
    stats.max_ns = sorted.back();
    
    // Calculate percentiles
    auto percentile = [&](double p) -> uint64_t {
        size_t index = static_cast<size_t>(p * (sorted.size() - 1));
        return sorted[index];
    };
    
    stats.p50_ns = percentile(0.50);
    stats.p95_ns = percentile(0.95);
    stats.p99_ns = percentile(0.99);
    stats.p999_ns = percentile(0.999);
    
    // Calculate mean
    uint64_t sum = std::accumulate(sorted.begin(), sorted.end(), 0ULL);
    stats.mean_ns = static_cast<double>(sum) / sorted.size();
    
    // Calculate standard deviation
    double variance = 0.0;
    for (uint64_t latency : sorted) {
        double diff = static_cast<double>(latency) - stats.mean_ns;
        variance += diff * diff;
    }
    variance /= sorted.size();
    stats.stddev_ns = std::sqrt(variance);
    
    return stats;
}

// LoadGenerator implementation
LoadGenerator::LoadGenerator(const Config& config) : config_(config) {}

void LoadGenerator::run() {
    running_ = true;
    completed_requests_ = 0;
    failed_requests_ = 0;
    latency_measurement_.reset();
    
    std::vector<std::thread> workers;
    workers.reserve(config_.thread_count);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Start worker threads
    for (size_t i = 0; i < config_.thread_count; ++i) {
        workers.emplace_back(&LoadGenerator::worker_thread, this);
    }
    
    // Wait for duration
    std::this_thread::sleep_for(config_.duration);
    running_ = false;
    
    // Wait for all workers to finish
    for (auto& worker : workers) {
        worker.join();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto actual_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count() / 1000.0;
    
    // Calculate results
    results_.total_requests = completed_requests_ + failed_requests_;
    results_.successful_requests = completed_requests_;
    results_.failed_requests = failed_requests_;
    results_.latency_stats = latency_measurement_.calculate_stats();
    results_.actual_rps = results_.total_requests / actual_duration;
}

void LoadGenerator::worker_thread() {
    auto target_interval = std::chrono::nanoseconds(
        static_cast<uint64_t>(1e9 / (config_.requests_per_second / config_.thread_count)));
    
    auto next_request_time = std::chrono::high_resolution_clock::now();
    
    while (running_) {
        auto now = std::chrono::high_resolution_clock::now();
        if (now >= next_request_time) {
            PerformanceTimer timer;
            
            try {
                config_.request_func();
                completed_requests_++;
                latency_measurement_.record_latency(timer.elapsed_ns());
            } catch (...) {
                failed_requests_++;
            }
            
            next_request_time += target_interval;
        } else {
            // Sleep for a short time to avoid busy waiting
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    }
}

// ChaosInjector implementation
void ChaosInjector::register_failure(const Config& config) {
    failure_configs_[config.type] = config;
}

void ChaosInjector::enable_chaos() {
    chaos_enabled_ = true;
}

void ChaosInjector::disable_chaos() {
    chaos_enabled_ = false;
}

bool ChaosInjector::should_inject_failure(FailureType type) {
    if (!chaos_enabled_) return false;
    
    auto it = failure_configs_.find(type);
    if (it == failure_configs_.end()) return false;
    
    return dis_(gen_) < it->second.probability;
}

void ChaosInjector::inject_failure(FailureType type) {
    auto it = failure_configs_.find(type);
    if (it == failure_configs_.end()) return;
    
    const auto& config = it->second;
    
    switch (type) {
        case FailureType::MEMORY_ALLOCATION_FAILURE:
            // Simulate memory allocation failure
            throw std::bad_alloc();
            
        case FailureType::NETWORK_TIMEOUT:
            // Simulate network timeout
            std::this_thread::sleep_for(config.duration);
            throw std::runtime_error("Network timeout");
            
        case FailureType::DISK_IO_ERROR:
            // Simulate disk I/O error
            throw std::runtime_error("Disk I/O error");
            
        case FailureType::CPU_SPIKE:
            // Simulate CPU spike
            {
                auto end_time = std::chrono::high_resolution_clock::now() + config.duration;
                while (std::chrono::high_resolution_clock::now() < end_time) {
                    // Busy wait to consume CPU
                    volatile int dummy = 0;
                    for (int i = 0; i < 1000; ++i) {
                        dummy += i;
                    }
                }
            }
            break;
            
        case FailureType::THREAD_STARVATION:
            // Simulate thread starvation
            std::this_thread::sleep_for(config.duration);
            break;
    }
    
    if (config.failure_action) {
        config.failure_action();
    }
}

// MemoryTracker implementation
MemoryTracker& MemoryTracker::instance() {
    static MemoryTracker instance;
    return instance;
}

void MemoryTracker::track_allocation(void* ptr, size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    allocations_[ptr] = size;
    stats_.total_allocations++;
    stats_.current_allocations++;
    stats_.current_memory_usage += size;
    stats_.peak_memory_usage = std::max(stats_.peak_memory_usage, stats_.current_memory_usage);
}

void MemoryTracker::track_deallocation(void* ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = allocations_.find(ptr);
    if (it != allocations_.end()) {
        stats_.total_deallocations++;
        stats_.current_allocations--;
        stats_.current_memory_usage -= it->second;
        allocations_.erase(it);
    }
}

MemoryTracker::Stats MemoryTracker::get_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

void MemoryTracker::reset_stats() {
    std::lock_guard<std::mutex> lock(mutex_);
    allocations_.clear();
    stats_ = {};
}

bool MemoryTracker::has_leaks() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return !allocations_.empty();
}

// UltraTestFixture implementation
void UltraTestFixture::SetUp() {
    chaos_injector_ = std::make_unique<ChaosInjector>();
    memory_tracker_ = &MemoryTracker::instance();
    memory_tracker_->reset_stats();
}

void UltraTestFixture::TearDown() {
    // Check for memory leaks
    if (memory_tracker_->has_leaks()) {
        auto stats = memory_tracker_->get_stats();
        FAIL() << "Memory leaks detected: " << stats.current_allocations 
               << " allocations not freed (" << stats.current_memory_usage << " bytes)";
    }
}

void UltraTestFixture::expect_latency_under(std::function<void()> func, uint64_t max_latency_ns) {
    PerformanceTimer timer;
    func();
    uint64_t actual_latency = timer.elapsed_ns();
    EXPECT_LT(actual_latency, max_latency_ns) 
        << "Operation took " << actual_latency << "ns, expected under " << max_latency_ns << "ns";
}

void UltraTestFixture::expect_throughput_over(std::function<void()> func, size_t min_ops_per_second) {
    const auto test_duration = std::chrono::seconds(1);
    size_t operations = 0;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    auto end_time = start_time + test_duration;
    
    while (std::chrono::high_resolution_clock::now() < end_time) {
        func();
        operations++;
    }
    
    EXPECT_GE(operations, min_ops_per_second)
        << "Achieved " << operations << " ops/sec, expected at least " << min_ops_per_second;
}

void UltraTestFixture::run_concurrent_test(std::function<void()> func, size_t thread_count, 
                                         std::chrono::seconds duration) {
    std::vector<std::future<void>> futures;
    std::atomic<bool> running{true};
    
    // Start worker threads
    for (size_t i = 0; i < thread_count; ++i) {
        futures.push_back(std::async(std::launch::async, [&]() {
            while (running) {
                func();
            }
        }));
    }
    
    // Run for specified duration
    std::this_thread::sleep_for(duration);
    running = false;
    
    // Wait for all threads to complete
    for (auto& future : futures) {
        future.wait();
    }
}

// UltraBenchmarkFixture implementation
void UltraBenchmarkFixture::SetUp(const ::benchmark::State& state) {
    setup_memory_pools();
    setup_test_data(state.range(0));
}

void UltraBenchmarkFixture::TearDown(const ::benchmark::State& state) {
    cleanup_resources();
}

void UltraBenchmarkFixture::setup_memory_pools() {
    // Initialize memory pools for benchmarking
    // This would be implemented based on the actual memory pool system
}

void UltraBenchmarkFixture::setup_test_data(size_t size) {
    data_size_ = size;
    test_data_.resize(size);
    
    // Fill with random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint8_t> dis(0, 255);
    
    for (size_t i = 0; i < size; ++i) {
        test_data_[i] = dis(gen);
    }
}

void UltraBenchmarkFixture::cleanup_resources() {
    test_data_.clear();
    data_size_ = 0;
}

} // namespace ultra::testing