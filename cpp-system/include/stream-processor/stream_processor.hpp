#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <thread>
#include <vector>
#include <immintrin.h>
#include "../lockfree/mpmc_ring_buffer.hpp"
#include "../common/types.hpp"
#include <unordered_map>

namespace ultra_cpp {
namespace stream {

/**
 * High-performance event structure optimized for cache efficiency
 * Aligned to cache line boundary for optimal SIMD processing
 */
struct alignas(64) StreamEvent {
    uint64_t timestamp_ns;
    uint32_t event_type;
    uint32_t tenant_id;
    uint32_t user_id;
    uint32_t data_size;
    uint32_t sequence_id;
    uint32_t reserved;  // Padding for alignment
    
    // Variable length data follows
    char data[];
    
    // Helper methods for SIMD processing
    static constexpr size_t HEADER_SIZE = 32;  // Size without data
    static constexpr size_t CACHE_LINE_SIZE = 64;
    
    size_t total_size() const noexcept {
        return HEADER_SIZE + data_size;
    }
    
    // Get aligned size for efficient memory operations
    size_t aligned_size() const noexcept {
        return (total_size() + CACHE_LINE_SIZE - 1) & ~(CACHE_LINE_SIZE - 1);
    }
};

/**
 * Event handler function type
 */
using EventHandler = std::function<void(const StreamEvent&)>;

/**
 * Batch processing function for SIMD operations
 */
using BatchHandler = std::function<void(const std::vector<const StreamEvent*>&)>;

/**
 * Performance metrics for stream processing
 */
struct StreamMetrics {
    std::atomic<uint64_t> events_processed{0};
    std::atomic<uint64_t> events_dropped{0};
    std::atomic<uint64_t> processing_latency_ns{0};
    std::atomic<uint64_t> queue_depth{0};
    std::atomic<uint64_t> batch_size_sum{0};
    std::atomic<uint64_t> batch_count{0};
    
    // Hardware performance counters
    std::atomic<uint64_t> cache_misses{0};
    std::atomic<uint64_t> cpu_cycles{0};
    std::atomic<uint64_t> instructions{0};
    
    double get_average_latency_ns() const {
        uint64_t processed = events_processed.load();
        return processed > 0 ? static_cast<double>(processing_latency_ns.load()) / processed : 0.0;
    }
    
    double get_average_batch_size() const {
        uint64_t batches = batch_count.load();
        return batches > 0 ? static_cast<double>(batch_size_sum.load()) / batches : 0.0;
    }
};

/**
 * Configuration for stream processor
 */
struct StreamConfig {
    size_t ring_buffer_size = 1024 * 1024;  // 1M events
    size_t worker_threads = std::thread::hardware_concurrency();
    size_t batch_size = 256;  // Optimal for SIMD processing
    std::chrono::microseconds batch_timeout{100};
    bool enable_simd = true;
    bool enable_hardware_counters = true;
    size_t memory_pool_size = 512 * 1024 * 1024;  // 512MB
};

/**
 * Ultra-high performance stream processor with lock-free queues,
 * SIMD acceleration, and microsecond latency processing
 */
class StreamProcessor {
public:
    explicit StreamProcessor(const StreamConfig& config = {});
    ~StreamProcessor();
    
    // Non-copyable, non-movable
    StreamProcessor(const StreamProcessor&) = delete;
    StreamProcessor& operator=(const StreamProcessor&) = delete;
    StreamProcessor(StreamProcessor&&) = delete;
    StreamProcessor& operator=(StreamProcessor&&) = delete;
    
    /**
     * Start the stream processing engine
     */
    void start();
    
    /**
     * Stop the stream processing engine gracefully
     */
    void stop();
    
    /**
     * Subscribe to events of a specific type
     */
    void subscribe(uint32_t event_type, EventHandler handler);
    
    /**
     * Subscribe to batch processing for SIMD operations
     */
    void subscribe_batch(uint32_t event_type, BatchHandler handler);
    
    /**
     * Publish an event to the stream (zero-copy when possible)
     */
    bool publish(const StreamEvent& event);
    
    /**
     * Publish event with data (will copy data)
     */
    bool publish(uint32_t event_type, uint32_t tenant_id, uint32_t user_id,
                const void* data, size_t data_size);
    
    /**
     * Get current performance metrics
     */
    StreamMetrics get_metrics() const;
    
    /**
     * Reset performance metrics
     */
    void reset_metrics();
    
    /**
     * Check if processor is running
     */
    bool is_running() const noexcept { return running_.load(); }
    
private:
    StreamConfig config_;
    std::atomic<bool> running_{false};
    std::atomic<uint32_t> sequence_counter_{0};
    
    // Lock-free ring buffer for event ingestion
    std::unique_ptr<lockfree::MPMCRingBuffer<StreamEvent*>> event_queue_;
    
    // Memory pool for event allocation
    std::unique_ptr<uint8_t[]> memory_pool_;
    std::atomic<size_t> memory_offset_{0};
    
    // Worker threads for processing
    std::vector<std::thread> worker_threads_;
    
    // Event handlers
    std::unordered_map<uint32_t, std::vector<EventHandler>> event_handlers_;
    std::unordered_map<uint32_t, std::vector<BatchHandler>> batch_handlers_;
    
    // Performance metrics
    mutable StreamMetrics metrics_;
    
    // Private methods
    void worker_loop(int worker_id);
    void process_event_batch(const std::vector<const StreamEvent*>& batch);
    void process_single_event(const StreamEvent& event);
    StreamEvent* allocate_event(size_t total_size);
    void deallocate_event(StreamEvent* event);
    
    // SIMD-accelerated processing
    void process_events_simd(const std::vector<const StreamEvent*>& events);
    void update_hardware_counters();
};

} // namespace stream
} // namespace ultra_cpp