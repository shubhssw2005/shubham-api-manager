#pragma once

#include <atomic>
#include <memory>
#include <array>
#include <cstdint>

namespace ultra_cpp {
namespace lockfree {

/**
 * Multi-Producer Multi-Consumer lock-free ring buffer
 * High-performance circular buffer for inter-thread communication
 */
template<typename T, size_t Capacity = 1024>
class MPMCRingBuffer {
public:
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be power of 2");
    static_assert(Capacity >= 2, "Capacity must be at least 2");
    
    MPMCRingBuffer();
    ~MPMCRingBuffer() = default;
    
    // Non-copyable, non-movable for safety
    MPMCRingBuffer(const MPMCRingBuffer&) = delete;
    MPMCRingBuffer& operator=(const MPMCRingBuffer&) = delete;
    MPMCRingBuffer(MPMCRingBuffer&&) = delete;
    MPMCRingBuffer& operator=(MPMCRingBuffer&&) = delete;
    
    // Producer operations
    bool try_enqueue(const T& item) noexcept;
    bool try_enqueue(T&& item) noexcept;
    
    template<typename... Args>
    bool try_emplace(Args&&... args) noexcept;
    
    // Consumer operations
    bool try_dequeue(T& item) noexcept;
    
    // Status queries
    bool empty() const noexcept;
    bool full() const noexcept;
    size_t size() const noexcept;
    size_t capacity() const noexcept { return Capacity; }
    
    // Statistics
    struct Stats {
        std::atomic<uint64_t> enqueue_attempts{0};
        std::atomic<uint64_t> enqueue_successes{0};
        std::atomic<uint64_t> dequeue_attempts{0};
        std::atomic<uint64_t> dequeue_successes{0};
        std::atomic<uint64_t> contention_count{0};
    };
    
    const Stats& get_stats() const noexcept { return stats_; }
    void reset_stats() noexcept;
    
private:
    struct alignas(64) Slot {
        std::atomic<uint64_t> sequence{0};
        T data;
    };
    
    static constexpr size_t CACHE_LINE_SIZE = 64;
    
    // Pad to avoid false sharing
    alignas(CACHE_LINE_SIZE) std::atomic<uint64_t> enqueue_pos_{0};
    alignas(CACHE_LINE_SIZE) std::atomic<uint64_t> dequeue_pos_{0};
    alignas(CACHE_LINE_SIZE) std::array<Slot, Capacity> buffer_;
    alignas(CACHE_LINE_SIZE) mutable Stats stats_;
    
    static constexpr uint64_t MASK = Capacity - 1;
};

template<typename T, size_t Capacity>
MPMCRingBuffer<T, Capacity>::MPMCRingBuffer() {
    // Initialize all slots with their sequence numbers
    for (size_t i = 0; i < Capacity; ++i) {
        buffer_[i].sequence.store(i, std::memory_order_relaxed);
    }
}

template<typename T, size_t Capacity>
bool MPMCRingBuffer<T, Capacity>::try_enqueue(const T& item) noexcept {
    stats_.enqueue_attempts.fetch_add(1, std::memory_order_relaxed);
    
    uint64_t pos = enqueue_pos_.load(std::memory_order_relaxed);
    
    while (true) {
        Slot& slot = buffer_[pos & MASK];
        uint64_t seq = slot.sequence.load(std::memory_order_acquire);
        
        if (seq == pos) {
            // Slot is available for writing
            if (enqueue_pos_.compare_exchange_weak(
                    pos, pos + 1, std::memory_order_relaxed)) {
                
                // Successfully claimed the slot
                slot.data = item;
                slot.sequence.store(pos + 1, std::memory_order_release);
                
                stats_.enqueue_successes.fetch_add(1, std::memory_order_relaxed);
                return true;
            }
            // CAS failed, retry with updated pos
            stats_.contention_count.fetch_add(1, std::memory_order_relaxed);
        } else if (seq < pos) {
            // Buffer is full
            return false;
        } else {
            // Another thread is ahead, update pos and retry
            pos = enqueue_pos_.load(std::memory_order_relaxed);
        }
    }
}

template<typename T, size_t Capacity>
bool MPMCRingBuffer<T, Capacity>::try_enqueue(T&& item) noexcept {
    stats_.enqueue_attempts.fetch_add(1, std::memory_order_relaxed);
    
    uint64_t pos = enqueue_pos_.load(std::memory_order_relaxed);
    
    while (true) {
        Slot& slot = buffer_[pos & MASK];
        uint64_t seq = slot.sequence.load(std::memory_order_acquire);
        
        if (seq == pos) {
            if (enqueue_pos_.compare_exchange_weak(
                    pos, pos + 1, std::memory_order_relaxed)) {
                
                slot.data = std::move(item);
                slot.sequence.store(pos + 1, std::memory_order_release);
                
                stats_.enqueue_successes.fetch_add(1, std::memory_order_relaxed);
                return true;
            }
            stats_.contention_count.fetch_add(1, std::memory_order_relaxed);
        } else if (seq < pos) {
            return false;
        } else {
            pos = enqueue_pos_.load(std::memory_order_relaxed);
        }
    }
}

template<typename T, size_t Capacity>
template<typename... Args>
bool MPMCRingBuffer<T, Capacity>::try_emplace(Args&&... args) noexcept {
    stats_.enqueue_attempts.fetch_add(1, std::memory_order_relaxed);
    
    uint64_t pos = enqueue_pos_.load(std::memory_order_relaxed);
    
    while (true) {
        Slot& slot = buffer_[pos & MASK];
        uint64_t seq = slot.sequence.load(std::memory_order_acquire);
        
        if (seq == pos) {
            if (enqueue_pos_.compare_exchange_weak(
                    pos, pos + 1, std::memory_order_relaxed)) {
                
                new (&slot.data) T(std::forward<Args>(args)...);
                slot.sequence.store(pos + 1, std::memory_order_release);
                
                stats_.enqueue_successes.fetch_add(1, std::memory_order_relaxed);
                return true;
            }
            stats_.contention_count.fetch_add(1, std::memory_order_relaxed);
        } else if (seq < pos) {
            return false;
        } else {
            pos = enqueue_pos_.load(std::memory_order_relaxed);
        }
    }
}

template<typename T, size_t Capacity>
bool MPMCRingBuffer<T, Capacity>::try_dequeue(T& item) noexcept {
    stats_.dequeue_attempts.fetch_add(1, std::memory_order_relaxed);
    
    uint64_t pos = dequeue_pos_.load(std::memory_order_relaxed);
    
    while (true) {
        Slot& slot = buffer_[pos & MASK];
        uint64_t seq = slot.sequence.load(std::memory_order_acquire);
        
        if (seq == pos + 1) {
            // Slot has data ready for reading
            if (dequeue_pos_.compare_exchange_weak(
                    pos, pos + 1, std::memory_order_relaxed)) {
                
                // Successfully claimed the slot
                item = std::move(slot.data);
                slot.sequence.store(pos + Capacity, std::memory_order_release);
                
                stats_.dequeue_successes.fetch_add(1, std::memory_order_relaxed);
                return true;
            }
            // CAS failed, retry with updated pos
            stats_.contention_count.fetch_add(1, std::memory_order_relaxed);
        } else if (seq < pos + 1) {
            // Buffer is empty
            return false;
        } else {
            // Another thread is ahead, update pos and retry
            pos = dequeue_pos_.load(std::memory_order_relaxed);
        }
    }
}

template<typename T, size_t Capacity>
bool MPMCRingBuffer<T, Capacity>::empty() const noexcept {
    uint64_t enq_pos = enqueue_pos_.load(std::memory_order_acquire);
    uint64_t deq_pos = dequeue_pos_.load(std::memory_order_acquire);
    return enq_pos == deq_pos;
}

template<typename T, size_t Capacity>
bool MPMCRingBuffer<T, Capacity>::full() const noexcept {
    uint64_t enq_pos = enqueue_pos_.load(std::memory_order_acquire);
    uint64_t deq_pos = dequeue_pos_.load(std::memory_order_acquire);
    return (enq_pos - deq_pos) >= Capacity;
}

template<typename T, size_t Capacity>
size_t MPMCRingBuffer<T, Capacity>::size() const noexcept {
    uint64_t enq_pos = enqueue_pos_.load(std::memory_order_acquire);
    uint64_t deq_pos = dequeue_pos_.load(std::memory_order_acquire);
    return static_cast<size_t>(enq_pos - deq_pos);
}

template<typename T, size_t Capacity>
void MPMCRingBuffer<T, Capacity>::reset_stats() noexcept {
    stats_.enqueue_attempts.store(0, std::memory_order_relaxed);
    stats_.enqueue_successes.store(0, std::memory_order_relaxed);
    stats_.dequeue_attempts.store(0, std::memory_order_relaxed);
    stats_.dequeue_successes.store(0, std::memory_order_relaxed);
    stats_.contention_count.store(0, std::memory_order_relaxed);
}

} // namespace lockfree
} // namespace ultra_cpp