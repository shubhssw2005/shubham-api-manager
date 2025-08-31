#include "../../include/stream-processor/event_ingestion.hpp"
#include <algorithm>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <random>

namespace ultra_cpp {
namespace stream {

EventIngestion::EventIngestion(const IngestionConfig& config)
    : config_(config)
    , event_queue_(std::make_unique<lockfree::MPMCRingBuffer<StreamEvent*>>())
    , memory_pool_(std::make_unique<uint8_t[]>(config_.memory_pool_size))
    , current_batch_(std::make_unique<EventBatch>()) {
    
    // Initialize batch
    current_batch_->batch_id = batch_id_counter_.fetch_add(1);
    current_batch_->timestamp_ns = get_current_time_ns();
}

EventIngestion::~EventIngestion() {
    if (running_.load()) {
        stop();
    }
}

void EventIngestion::start() {
    if (running_.exchange(true)) {
        return; // Already running
    }
    
    last_batch_time_ns_.store(get_current_time_ns());
    
    // Start worker threads
    worker_threads_.reserve(config_.worker_threads);
    for (size_t i = 0; i < config_.worker_threads; ++i) {
        worker_threads_.emplace_back(&EventIngestion::worker_loop, this, static_cast<int>(i));
    }
    
    // Start batch manager thread
    batch_manager_thread_ = std::thread(&EventIngestion::batch_manager_loop, this);
}

void EventIngestion::stop() {
    running_.store(false);
    
    // Process final batch
    flush_batch();
    
    // Wait for all threads to finish
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    worker_threads_.clear();
    
    if (batch_manager_thread_.joinable()) {
        batch_manager_thread_.join();
    }
}

bool EventIngestion::ingest_event(StreamEvent* event) {
    if (!running_.load() || !event) {
        return false;
    }
    
    auto start_time = get_current_time_ns();
    
    // Try to enqueue the event
    if (!event_queue_->try_enqueue(event)) {
        stats_.events_dropped.fetch_add(1);
        return false;
    }
    
    // Update statistics
    stats_.events_ingested.fetch_add(1);
    stats_.queue_depth.store(event_queue_->size());
    
    auto end_time = get_current_time_ns();
    stats_.ingestion_latency_ns.fetch_add(end_time - start_time);
    
    return true;
}

bool EventIngestion::ingest_event(uint32_t event_type, uint32_t tenant_id, uint32_t user_id,
                                 const void* data, size_t data_size) {
    if (!running_.load()) {
        return false;
    }
    
    // Allocate event
    size_t total_size = StreamEvent::HEADER_SIZE + data_size;
    StreamEvent* event = allocate_event(total_size);
    if (!event) {
        stats_.events_dropped.fetch_add(1);
        return false;
    }
    
    // Initialize event
    event->timestamp_ns = get_current_time_ns();
    event->event_type = event_type;
    event->tenant_id = tenant_id;
    event->user_id = user_id;
    event->data_size = static_cast<uint32_t>(data_size);
    event->sequence_id = 0; // Will be set by stream processor
    event->reserved = 0;
    
    // Copy data
    if (data && data_size > 0) {
        std::memcpy(event->data, data, data_size);
    }
    
    return ingest_event(event);
}

bool EventIngestion::ingest_batch(const std::vector<StreamEvent*>& events) {
    if (!running_.load() || events.empty()) {
        return false;
    }
    
    auto start_time = get_current_time_ns();
    size_t ingested_count = 0;
    
    for (StreamEvent* event : events) {
        if (event_queue_->try_enqueue(event)) {
            ++ingested_count;
        } else {
            stats_.events_dropped.fetch_add(1);
        }
    }
    
    // Update statistics
    stats_.events_ingested.fetch_add(ingested_count);
    stats_.queue_depth.store(event_queue_->size());
    
    auto end_time = get_current_time_ns();
    stats_.ingestion_latency_ns.fetch_add(end_time - start_time);
    
    return ingested_count == events.size();
}

void EventIngestion::register_batch_callback(BatchCallback callback) {
    batch_callbacks_.push_back(std::move(callback));
}

IngestionStats EventIngestion::get_stats() const {
    update_throughput_stats();
    return stats_;
}

void EventIngestion::reset_stats() {
    stats_.events_ingested.store(0);
    stats_.events_dropped.store(0);
    stats_.batches_processed.store(0);
    stats_.ingestion_latency_ns.store(0);
    stats_.queue_depth.store(0);
    stats_.memory_used.store(0);
    stats_.bytes_per_second.store(0);
    stats_.events_per_second.store(0);
}

void EventIngestion::flush_batch() {
    if (current_batch_->get_count() > 0) {
        process_current_batch();
    }
}

size_t EventIngestion::get_queue_depth() const {
    return event_queue_->size();
}

void EventIngestion::worker_loop(int worker_id) {
    std::vector<StreamEvent*> local_batch;
    local_batch.reserve(config_.batch_size);
    
    while (running_.load()) {
        StreamEvent* event = nullptr;
        
        // Dequeue events into local batch
        while (local_batch.size() < config_.batch_size && event_queue_->try_dequeue(event)) {
            local_batch.push_back(event);
        }
        
        // Process local batch if we have events
        if (!local_batch.empty()) {
            // SIMD validation if enabled
            if (config_.enable_simd) {
                simd_validate_events(local_batch);
            }
            
            // Prefetch for better cache performance
            prefetch_events(local_batch);
            
            // Add events to current batch (thread-safe)
            for (StreamEvent* evt : local_batch) {
                if (!current_batch_->add_event(evt)) {
                    // Current batch is full, process it
                    process_current_batch();
                    
                    // Create new batch and add event
                    current_batch_ = std::make_unique<EventBatch>();
                    current_batch_->batch_id = batch_id_counter_.fetch_add(1);
                    current_batch_->timestamp_ns = get_current_time_ns();
                    current_batch_->add_event(evt);
                }
            }
            
            local_batch.clear();
        } else {
            // No events available, yield CPU
            std::this_thread::yield();
        }
    }
}

void EventIngestion::batch_manager_loop() {
    while (running_.load()) {
        // Check if we should flush the current batch due to timeout
        if (should_flush_batch()) {
            flush_batch();
        }
        
        // Sleep for a short time to avoid busy waiting
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
}

void EventIngestion::process_current_batch() {
    if (current_batch_->get_count() == 0) {
        return;
    }
    
    // SIMD processing if enabled and batch is large enough
    if (config_.enable_simd && current_batch_->get_count() >= 4) {
        simd_process_batch(*current_batch_);
    }
    
    // Call all registered batch callbacks
    for (const auto& callback : batch_callbacks_) {
        callback(*current_batch_);
    }
    
    // Update statistics
    stats_.batches_processed.fetch_add(1);
    
    // Reset current batch
    current_batch_->reset();
    current_batch_->batch_id = batch_id_counter_.fetch_add(1);
    current_batch_->timestamp_ns = get_current_time_ns();
    last_batch_time_ns_.store(get_current_time_ns());
}

StreamEvent* EventIngestion::allocate_event(size_t total_size) {
    // Align to cache line boundary
    size_t aligned_size = (total_size + StreamEvent::CACHE_LINE_SIZE - 1) & ~(StreamEvent::CACHE_LINE_SIZE - 1);
    
    size_t offset = memory_offset_.fetch_add(aligned_size);
    if (offset + aligned_size > config_.memory_pool_size) {
        // Memory pool exhausted
        return nullptr;
    }
    
    stats_.memory_used.store(offset + aligned_size);
    return reinterpret_cast<StreamEvent*>(memory_pool_.get() + offset);
}

void EventIngestion::deallocate_event(StreamEvent* event) {
    // Simple implementation - in practice would use a free list
}

void EventIngestion::simd_process_batch(const EventBatch& batch) {
    auto events = batch.get_events();
    if (events.size() < 4) {
        return;
    }
    
    // SIMD processing for batch validation and preprocessing
    size_t simd_count = (events.size() / 4) * 4;
    
    for (size_t i = 0; i < simd_count; i += 4) {
        // Load 4 event timestamps for validation
        __m256i timestamps = _mm256_set_epi64x(
            events[i + 3]->timestamp_ns,
            events[i + 2]->timestamp_ns,
            events[i + 1]->timestamp_ns,
            events[i]->timestamp_ns
        );
        
        // Validate timestamp ordering (events should be roughly in order)
        uint64_t current_time = get_current_time_ns();
        __m256i current_times = _mm256_set1_epi64x(current_time);
        __m256i time_diff = _mm256_sub_epi64(current_times, timestamps);
        
        // Check for reasonable timestamp range (within last hour)
        __m256i max_age = _mm256_set1_epi64x(3600ULL * 1000000000ULL); // 1 hour in nanoseconds
        __m256i valid_mask = _mm256_cmpgt_epi64(max_age, time_diff);
        
        // In a real implementation, would handle invalid events based on mask
    }
}

void EventIngestion::simd_validate_events(const std::vector<StreamEvent*>& events) {
    if (events.size() < 4) {
        return;
    }
    
    // SIMD validation for event structure integrity
    size_t simd_count = (events.size() / 4) * 4;
    
    for (size_t i = 0; i < simd_count; i += 4) {
        // Load event sizes for validation
        __m128i sizes = _mm_set_epi32(
            events[i + 3]->data_size,
            events[i + 2]->data_size,
            events[i + 1]->data_size,
            events[i]->data_size
        );
        
        // Check for reasonable data sizes (< 1MB per event)
        __m128i max_size = _mm_set1_epi32(1024 * 1024);
        __m128i valid_sizes = _mm_cmplt_epi32(sizes, max_size);
        
        // In practice, would handle validation failures
    }
}

void EventIngestion::prefetch_events(const std::vector<StreamEvent*>& events) {
    // Prefetch event data for better cache performance
    for (const StreamEvent* event : events) {
        __builtin_prefetch(event, 0, 3); // Prefetch for read, high temporal locality
        if (event->data_size > 0) {
            __builtin_prefetch(event->data, 0, 3);
        }
    }
}

void EventIngestion::update_throughput_stats() const {
    static uint64_t last_update_time = 0;
    static uint64_t last_events_count = 0;
    static uint64_t last_bytes_count = 0;
    
    uint64_t current_time = get_current_time_ns();
    uint64_t current_events = stats_.events_ingested.load();
    uint64_t current_bytes = stats_.memory_used.load();
    
    if (last_update_time > 0) {
        uint64_t time_diff_ns = current_time - last_update_time;
        if (time_diff_ns > 1000000000ULL) { // Update every second
            uint64_t events_diff = current_events - last_events_count;
            uint64_t bytes_diff = current_bytes - last_bytes_count;
            
            double time_diff_s = static_cast<double>(time_diff_ns) / 1000000000.0;
            
            stats_.events_per_second.store(static_cast<uint64_t>(events_diff / time_diff_s));
            stats_.bytes_per_second.store(static_cast<uint64_t>(bytes_diff / time_diff_s));
            
            last_update_time = current_time;
            last_events_count = current_events;
            last_bytes_count = current_bytes;
        }
    } else {
        last_update_time = current_time;
        last_events_count = current_events;
        last_bytes_count = current_bytes;
    }
}

uint64_t EventIngestion::get_current_time_ns() const {
    return std::chrono::high_resolution_clock::now().time_since_epoch().count();
}

bool EventIngestion::should_flush_batch() const {
    uint64_t current_time = get_current_time_ns();
    uint64_t last_batch_time = last_batch_time_ns_.load();
    uint64_t timeout_ns = config_.batch_timeout.count() * 1000; // Convert microseconds to nanoseconds
    
    return (current_time - last_batch_time) >= timeout_ns;
}

size_t EventIngestion::calculate_event_size(size_t data_size) const {
    return StreamEvent::HEADER_SIZE + data_size;
}

} // namespace stream
} // namespace ultra_cpp