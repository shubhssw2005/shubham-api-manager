#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>
#include <immintrin.h>
#include <string>
#include "../lockfree/mpmc_ring_buffer.hpp"
#include "stream_processor.hpp"

namespace ultra_cpp {
namespace stream {

/**
 * Event ingestion configuration
 */
struct IngestionConfig {
    size_t batch_size = 256;  // Optimal for SIMD processing
    std::chrono::microseconds batch_timeout{100};  // Maximum wait time for batch
    size_t ring_buffer_size = 1024 * 1024;  // 1M events
    size_t worker_threads = std::thread::hardware_concurrency();
    bool enable_zero_copy = true;
    bool enable_batching = true;
    bool enable_compression = false;  // For network ingestion
    size_t memory_pool_size = 256 * 1024 * 1024;  // 256MB
};

/**
 * Ingestion statistics
 */
struct IngestionStats {
    std::atomic<uint64_t> events_ingested{0};
    std::atomic<uint64_t> events_dropped{0};
    std::atomic<uint64_t> batches_processed{0};
    std::atomic<uint64_t> ingestion_latency_ns{0};
    std::atomic<uint64_t> queue_depth{0};
    std::atomic<uint64_t> memory_used{0};
    
    // Throughput metrics
    std::atomic<uint64_t> bytes_per_second{0};
    std::atomic<uint64_t> events_per_second{0};
    
    double get_average_latency_ns() const {
        uint64_t ingested = events_ingested.load();
        return ingested > 0 ? static_cast<double>(ingestion_latency_ns.load()) / ingested : 0.0;
    }
    
    double get_drop_rate() const {
        uint64_t total = events_ingested.load() + events_dropped.load();
        return total > 0 ? static_cast<double>(events_dropped.load()) / total : 0.0;
    }
};

/**
 * Event batch for efficient processing
 */
struct alignas(64) EventBatch {
    static constexpr size_t MAX_BATCH_SIZE = 512;
    
    std::atomic<size_t> count{0};
    uint64_t batch_id;
    uint64_t timestamp_ns;
    StreamEvent* events[MAX_BATCH_SIZE];
    
    EventBatch() : batch_id(0), timestamp_ns(0) {
        for (size_t i = 0; i < MAX_BATCH_SIZE; ++i) {
            events[i] = nullptr;
        }
    }
    
    bool add_event(StreamEvent* event) {
        size_t current_count = count.load();
        if (current_count >= MAX_BATCH_SIZE) {
            return false;
        }
        
        events[current_count] = event;
        count.store(current_count + 1);
        return true;
    }
    
    void reset() {
        count.store(0);
        batch_id = 0;
        timestamp_ns = 0;
        for (size_t i = 0; i < MAX_BATCH_SIZE; ++i) {
            events[i] = nullptr;
        }
    }
    
    size_t get_count() const { return count.load(); }
    
    std::vector<StreamEvent*> get_events() const {
        size_t cnt = count.load();
        std::vector<StreamEvent*> result;
        result.reserve(cnt);
        for (size_t i = 0; i < cnt; ++i) {
            result.push_back(events[i]);
        }
        return result;
    }
};

/**
 * Callback for processed batches
 */
using BatchCallback = std::function<void(const EventBatch&)>;

/**
 * High-performance event ingestion engine with lock-free queues and batching
 */
class EventIngestion {
public:
    explicit EventIngestion(const IngestionConfig& config = {});
    ~EventIngestion();
    
    // Non-copyable, non-movable
    EventIngestion(const EventIngestion&) = delete;
    EventIngestion& operator=(const EventIngestion&) = delete;
    
    /**
     * Start the ingestion engine
     */
    void start();
    
    /**
     * Stop the ingestion engine gracefully
     */
    void stop();
    
    /**
     * Ingest a single event (zero-copy when possible)
     */
    bool ingest_event(StreamEvent* event);
    
    /**
     * Ingest event with data copying
     */
    bool ingest_event(uint32_t event_type, uint32_t tenant_id, uint32_t user_id,
                     const void* data, size_t data_size);
    
    /**
     * Ingest a batch of events
     */
    bool ingest_batch(const std::vector<StreamEvent*>& events);
    
    /**
     * Register callback for processed batches
     */
    void register_batch_callback(BatchCallback callback);
    
    /**
     * Get current ingestion statistics
     */
    IngestionStats get_stats() const;
    
    /**
     * Reset statistics
     */
    void reset_stats();
    
    /**
     * Force flush of current batch
     */
    void flush_batch();
    
    /**
     * Check if ingestion is running
     */
    bool is_running() const { return running_.load(); }
    
    /**
     * Get current queue depth
     */
    size_t get_queue_depth() const;
    
private:
    IngestionConfig config_;
    std::atomic<bool> running_{false};
    std::atomic<uint64_t> batch_id_counter_{0};
    
    // Lock-free ring buffer for events
    std::unique_ptr<lockfree::MPMCRingBuffer<StreamEvent*>> event_queue_;
    
    // Memory pool for event allocation
    std::unique_ptr<uint8_t[]> memory_pool_;
    std::atomic<size_t> memory_offset_{0};
    
    // Batch processing
    std::unique_ptr<EventBatch> current_batch_;
    std::atomic<uint64_t> last_batch_time_ns_{0};
    
    // Worker threads
    std::vector<std::thread> worker_threads_;
    std::thread batch_manager_thread_;
    
    // Batch callbacks
    std::vector<BatchCallback> batch_callbacks_;
    
    // Statistics
    mutable IngestionStats stats_;
    std::atomic<uint64_t> last_stats_update_ns_{0};
    
    // Private methods
    void worker_loop(int worker_id);
    void batch_manager_loop();
    void process_current_batch();
    StreamEvent* allocate_event(size_t total_size);
    void deallocate_event(StreamEvent* event);
    
    // SIMD-accelerated batch processing
    void simd_process_batch(const EventBatch& batch);
    void simd_validate_events(const std::vector<StreamEvent*>& events);
    
    // Performance optimization
    void prefetch_events(const std::vector<StreamEvent*>& events);
    void update_throughput_stats();
    
    // Utility functions
    uint64_t get_current_time_ns() const;
    bool should_flush_batch() const;
    size_t calculate_event_size(size_t data_size) const;
};

/**
 * Network event ingestion for distributed systems
 */
class NetworkEventIngestion {
public:
    struct NetworkConfig {
        uint16_t port = 8090;
        size_t max_connections = 1000;
        size_t buffer_size = 64 * 1024;  // 64KB per connection
        bool enable_compression = true;
        std::string bind_address = "0.0.0.0";
    };
    
    explicit NetworkEventIngestion(const NetworkConfig& config, 
                                  EventIngestion& local_ingestion);
    ~NetworkEventIngestion();
    
    /**
     * Start network ingestion server
     */
    void start();
    
    /**
     * Stop network ingestion server
     */
    void stop();
    
    /**
     * Get network statistics
     */
    struct NetworkStats {
        std::atomic<uint64_t> connections_active{0};
        std::atomic<uint64_t> bytes_received{0};
        std::atomic<uint64_t> packets_received{0};
        std::atomic<uint64_t> packets_dropped{0};
    };
    
    NetworkStats get_network_stats() const;
    
private:
    NetworkConfig config_;
    EventIngestion& local_ingestion_;
    std::atomic<bool> running_{false};
    
    // Network handling
    std::thread server_thread_;
    mutable NetworkStats network_stats_;
    
    void server_loop();
    void handle_connection(int client_socket);
    bool parse_network_event(const uint8_t* buffer, size_t size, StreamEvent*& event);
};

} // namespace stream
} // namespace ultra_cpp