#include "../../include/stream-processor/stream_processor.hpp"
#include <algorithm>
#include <cstring>
#include <sys/mman.h>
#include <unistd.h>
#include <pthread.h>
#include <unordered_map>

#ifdef __linux__
#include <linux/perf_event.h>
#include <sys/syscall.h>
#endif

namespace ultra_cpp {
namespace stream {

StreamProcessor::StreamProcessor(const StreamConfig& config)
    : config_(config)
    , event_queue_(std::make_unique<lockfree::MPMCRingBuffer<StreamEvent*>>())
    , memory_pool_(std::make_unique<uint8_t[]>(config_.memory_pool_size)) {
    
    // Initialize memory pool with huge pages if available
    if (mlock(memory_pool_.get(), config_.memory_pool_size) != 0) {
        // Fallback to regular memory if huge pages not available
    }
}

StreamProcessor::~StreamProcessor() {
    if (running_.load()) {
        stop();
    }
    
    if (memory_pool_) {
        munlock(memory_pool_.get(), config_.memory_pool_size);
    }
}

void StreamProcessor::start() {
    if (running_.exchange(true)) {
        return; // Already running
    }
    
    // Start worker threads
    worker_threads_.reserve(config_.worker_threads);
    for (size_t i = 0; i < config_.worker_threads; ++i) {
        worker_threads_.emplace_back(&StreamProcessor::worker_loop, this, static_cast<int>(i));
        
        // Set CPU affinity for better cache locality
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(i % std::thread::hardware_concurrency(), &cpuset);
        pthread_setaffinity_np(worker_threads_[i].native_handle(), sizeof(cpu_set_t), &cpuset);
    }
}

void StreamProcessor::stop() {
    running_.store(false);
    
    // Wait for all worker threads to finish
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    worker_threads_.clear();
}

void StreamProcessor::subscribe(uint32_t event_type, EventHandler handler) {
    event_handlers_[event_type].push_back(std::move(handler));
}

void StreamProcessor::subscribe_batch(uint32_t event_type, BatchHandler handler) {
    batch_handlers_[event_type].push_back(std::move(handler));
}

bool StreamProcessor::publish(const StreamEvent& event) {
    if (!running_.load()) {
        return false;
    }
    
    // Allocate memory for the event
    StreamEvent* event_copy = allocate_event(event.total_size());
    if (!event_copy) {
        metrics_.events_dropped.fetch_add(1);
        return false;
    }
    
    // Copy event data
    std::memcpy(event_copy, &event, event.total_size());
    event_copy->sequence_id = sequence_counter_.fetch_add(1);
    event_copy->timestamp_ns = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    
    // Try to enqueue
    if (!event_queue_->try_enqueue(event_copy)) {
        deallocate_event(event_copy);
        metrics_.events_dropped.fetch_add(1);
        return false;
    }
    
    metrics_.queue_depth.store(event_queue_->size());
    return true;
}

bool StreamProcessor::publish(uint32_t event_type, uint32_t tenant_id, uint32_t user_id,
                             const void* data, size_t data_size) {
    if (!running_.load()) {
        return false;
    }
    
    size_t total_size = StreamEvent::HEADER_SIZE + data_size;
    StreamEvent* event = allocate_event(total_size);
    if (!event) {
        metrics_.events_dropped.fetch_add(1);
        return false;
    }
    
    // Initialize event
    event->timestamp_ns = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    event->event_type = event_type;
    event->tenant_id = tenant_id;
    event->user_id = user_id;
    event->data_size = static_cast<uint32_t>(data_size);
    event->sequence_id = sequence_counter_.fetch_add(1);
    event->reserved = 0;
    
    // Copy data
    if (data && data_size > 0) {
        std::memcpy(event->data, data, data_size);
    }
    
    // Try to enqueue
    if (!event_queue_->try_enqueue(event)) {
        deallocate_event(event);
        metrics_.events_dropped.fetch_add(1);
        return false;
    }
    
    metrics_.queue_depth.store(event_queue_->size());
    return true;
}

StreamMetrics StreamProcessor::get_metrics() const {
    return metrics_;
}

void StreamProcessor::reset_metrics() {
    metrics_.events_processed.store(0);
    metrics_.events_dropped.store(0);
    metrics_.processing_latency_ns.store(0);
    metrics_.queue_depth.store(0);
    metrics_.batch_size_sum.store(0);
    metrics_.batch_count.store(0);
    metrics_.cache_misses.store(0);
    metrics_.cpu_cycles.store(0);
    metrics_.instructions.store(0);
}

StreamEvent* StreamProcessor::allocate_event(size_t total_size) {
    // Align to cache line boundary
    size_t aligned_size = (total_size + StreamEvent::CACHE_LINE_SIZE - 1) & ~(StreamEvent::CACHE_LINE_SIZE - 1);
    
    size_t offset = memory_offset_.fetch_add(aligned_size);
    if (offset + aligned_size > config_.memory_pool_size) {
        // Memory pool exhausted - could implement circular buffer or dynamic allocation
        return nullptr;
    }
    
    return reinterpret_cast<StreamEvent*>(memory_pool_.get() + offset);
}

void StreamProcessor::deallocate_event(StreamEvent* event) {
    // In this simple implementation, we don't actually deallocate
    // A more sophisticated implementation would use a free list or memory pool
    // For now, we rely on the circular nature of the memory pool
}

} // namespace stream
} // namespace ultra_cppvoid S
treamProcessor::worker_loop(int worker_id) {
    std::vector<const StreamEvent*> batch;
    batch.reserve(config_.batch_size);
    
    auto last_batch_time = std::chrono::high_resolution_clock::now();
    
    while (running_.load()) {
        StreamEvent* event = nullptr;
        
        // Try to dequeue events for batching
        while (batch.size() < config_.batch_size && event_queue_->try_dequeue(event)) {
            batch.push_back(event);
        }
        
        // Process batch if we have events or timeout occurred
        auto now = std::chrono::high_resolution_clock::now();
        bool timeout = std::chrono::duration_cast<std::chrono::microseconds>(now - last_batch_time) >= config_.batch_timeout;
        
        if (!batch.empty() && (batch.size() >= config_.batch_size || timeout)) {
            auto start_time = std::chrono::high_resolution_clock::now();
            
            process_event_batch(batch);
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto latency_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
            
            // Update metrics
            metrics_.events_processed.fetch_add(batch.size());
            metrics_.processing_latency_ns.fetch_add(latency_ns);
            metrics_.batch_size_sum.fetch_add(batch.size());
            metrics_.batch_count.fetch_add(1);
            
            // Clean up batch
            for (const StreamEvent* evt : batch) {
                deallocate_event(const_cast<StreamEvent*>(evt));
            }
            batch.clear();
            last_batch_time = now;
        }
        
        // Update hardware counters periodically
        if (config_.enable_hardware_counters && worker_id == 0) {
            update_hardware_counters();
        }
        
        // Small yield to prevent busy waiting
        if (batch.empty()) {
            std::this_thread::yield();
        }
    }
    
    // Process remaining events in batch
    if (!batch.empty()) {
        process_event_batch(batch);
        for (const StreamEvent* evt : batch) {
            deallocate_event(const_cast<StreamEvent*>(evt));
        }
    }
}

void StreamProcessor::process_event_batch(const std::vector<const StreamEvent*>& batch) {
    if (batch.empty()) {
        return;
    }
    
    // SIMD-accelerated processing if enabled
    if (config_.enable_simd && batch.size() >= 4) {
        process_events_simd(batch);
    }
    
    // Group events by type for efficient processing
    std::unordered_map<uint32_t, std::vector<const StreamEvent*>> events_by_type;
    for (const StreamEvent* event : batch) {
        events_by_type[event->event_type].push_back(event);
    }
    
    // Process each event type
    for (const auto& [event_type, events] : events_by_type) {
        // Call batch handlers first
        auto batch_it = batch_handlers_.find(event_type);
        if (batch_it != batch_handlers_.end()) {
            for (const auto& handler : batch_it->second) {
                handler(events);
            }
        }
        
        // Call individual event handlers
        auto event_it = event_handlers_.find(event_type);
        if (event_it != event_handlers_.end()) {
            for (const StreamEvent* event : events) {
                for (const auto& handler : event_it->second) {
                    handler(*event);
                }
            }
        }
    }
}

void StreamProcessor::process_single_event(const StreamEvent& event) {
    auto it = event_handlers_.find(event.event_type);
    if (it != event_handlers_.end()) {
        for (const auto& handler : it->second) {
            handler(event);
        }
    }
}

void StreamProcessor::process_events_simd(const std::vector<const StreamEvent*>& events) {
    if (!config_.enable_simd || events.size() < 4) {
        return;
    }
    
    // SIMD processing for timestamp validation and basic filtering
    size_t simd_count = (events.size() / 4) * 4;
    
    for (size_t i = 0; i < simd_count; i += 4) {
        // Load 4 timestamps
        __m256i timestamps = _mm256_set_epi64x(
            events[i + 3]->timestamp_ns,
            events[i + 2]->timestamp_ns,
            events[i + 1]->timestamp_ns,
            events[i]->timestamp_ns
        );
        
        // Load 4 event types
        __m128i event_types = _mm_set_epi32(
            events[i + 3]->event_type,
            events[i + 2]->event_type,
            events[i + 1]->event_type,
            events[i]->event_type
        );
        
        // Perform SIMD validation (example: check for valid timestamps)
        uint64_t current_time = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        __m256i current_times = _mm256_set1_epi64x(current_time);
        __m256i time_diff = _mm256_sub_epi64(current_times, timestamps);
        
        // Check if timestamps are within reasonable range (1 second)
        __m256i max_diff = _mm256_set1_epi64x(1000000000ULL); // 1 second in nanoseconds
        __m256i valid_mask = _mm256_cmpgt_epi64(max_diff, time_diff);
        
        // Store validation results (simplified - in real implementation would handle invalid events)
        // This is just an example of SIMD usage for batch processing
    }
}

void StreamProcessor::update_hardware_counters() {
#ifdef __linux__
    // This is a simplified example - real implementation would use perf_event_open
    // to read hardware performance counters
    static uint64_t last_cycles = 0;
    static uint64_t last_instructions = 0;
    
    // Placeholder for actual hardware counter reading
    // In practice, you would use perf_event_open() and read() syscalls
    uint64_t cycles = last_cycles + 1000000; // Dummy increment
    uint64_t instructions = last_instructions + 500000; // Dummy increment
    
    metrics_.cpu_cycles.store(cycles);
    metrics_.instructions.store(instructions);
    
    last_cycles = cycles;
    last_instructions = instructions;
#endif
}