#pragma once

#include <atomic>
#include <memory>
#include <functional>
#include <thread>
#include <chrono>
#include <vector>

namespace ultra {
namespace memory {

/**
 * RCU (Read-Copy-Update) implementation for safe concurrent access
 * Provides lock-free read access with deferred memory reclamation
 */
class RcuManager {
public:
    using EpochType = uint64_t;
    using DeleterFunc = std::function<void()>;
    
    static constexpr EpochType INVALID_EPOCH = 0;
    
    RcuManager();
    ~RcuManager();
    
    // Non-copyable, non-movable
    RcuManager(const RcuManager&) = delete;
    RcuManager& operator=(const RcuManager&) = delete;
    
    /**
     * Enter read-side critical section
     * @return Current epoch for this read section
     */
    EpochType read_lock() noexcept;
    
    /**
     * Exit read-side critical section
     * @param epoch Epoch returned by read_lock()
     */
    void read_unlock(EpochType epoch) noexcept;
    
    /**
     * Schedule object for deferred deletion
     * @param deleter Function to call when safe to delete
     */
    void defer_delete(DeleterFunc deleter);
    
    /**
     * Force synchronization - wait for all current readers to finish
     */
    void synchronize();
    
    /**
     * Get current global epoch
     */
    EpochType get_current_epoch() const noexcept {
        return global_epoch_.load(std::memory_order_acquire);
    }
    
    /**
     * Get the singleton instance
     */
    static RcuManager& instance();
    
    /**
     * Statistics for RCU operations
     */
    struct Stats {
        std::atomic<uint64_t> read_sections{0};
        std::atomic<uint64_t> deferred_deletions{0};
        std::atomic<uint64_t> completed_deletions{0};
        std::atomic<uint64_t> synchronizations{0};
        std::atomic<uint64_t> active_readers{0};
    };
    
    const Stats& get_stats() const noexcept { return stats_; }
    
private:
    struct alignas(64) ReaderState {
        std::atomic<EpochType> epoch{INVALID_EPOCH};
        std::atomic<bool> active{false};
    };
    
    struct DeferredDeletion {
        DeleterFunc deleter;
        EpochType epoch;
        DeferredDeletion* next = nullptr;
    };
    
    std::atomic<EpochType> global_epoch_;
    std::vector<std::unique_ptr<ReaderState>> reader_states_;
    std::atomic<size_t> reader_count_{0};
    
    // Deferred deletion queue
    std::atomic<DeferredDeletion*> deletion_queue_head_{nullptr};
    std::atomic<DeferredDeletion*> deletion_queue_tail_{nullptr};
    
    Stats stats_;
    
    // Background thread for processing deletions
    std::atomic<bool> shutdown_{false};
    std::thread cleanup_thread_;
    
    static thread_local ReaderState* tl_reader_state_;
    
    ReaderState* get_reader_state();
    void cleanup_loop();
    void process_deletions();
    EpochType find_minimum_epoch() const noexcept;
};

/**
 * RAII wrapper for RCU read-side critical sections
 */
class RcuReadGuard {
public:
    explicit RcuReadGuard(RcuManager& manager = RcuManager::instance()) 
        : manager_(manager), epoch_(manager_.read_lock()) {}
    
    ~RcuReadGuard() {
        manager_.read_unlock(epoch_);
    }
    
    // Non-copyable, non-movable
    RcuReadGuard(const RcuReadGuard&) = delete;
    RcuReadGuard& operator=(const RcuReadGuard&) = delete;
    RcuReadGuard(RcuReadGuard&&) = delete;
    RcuReadGuard& operator=(RcuReadGuard&&) = delete;
    
    RcuManager::EpochType get_epoch() const noexcept { return epoch_; }
    
private:
    RcuManager& manager_;
    RcuManager::EpochType epoch_;
};

/**
 * RCU-protected smart pointer for safe concurrent access
 */
template<typename T>
class RcuPtr {
public:
    RcuPtr() : ptr_(nullptr) {}
    
    explicit RcuPtr(T* ptr) : ptr_(ptr) {}
    
    RcuPtr(const RcuPtr& other) : ptr_(other.ptr_.load()) {}
    
    RcuPtr& operator=(const RcuPtr& other) {
        store(other.load());
        return *this;
    }
    
    RcuPtr& operator=(T* ptr) {
        store(ptr);
        return *this;
    }
    
    /**
     * Load pointer (must be called within RCU read section)
     */
    T* load() const noexcept {
        return ptr_.load(std::memory_order_acquire);
    }
    
    /**
     * Store new pointer and schedule old one for deletion
     */
    void store(T* new_ptr) {
        T* old_ptr = ptr_.exchange(new_ptr, std::memory_order_acq_rel);
        if (old_ptr) {
            RcuManager::instance().defer_delete([old_ptr]() {
                delete old_ptr;
            });
        }
    }
    
    /**
     * Compare and swap
     */
    bool compare_exchange_weak(T*& expected, T* desired) noexcept {
        return ptr_.compare_exchange_weak(expected, desired, 
                                        std::memory_order_acq_rel,
                                        std::memory_order_acquire);
    }
    
    bool compare_exchange_strong(T*& expected, T* desired) noexcept {
        return ptr_.compare_exchange_strong(expected, desired,
                                          std::memory_order_acq_rel,
                                          std::memory_order_acquire);
    }
    
    /**
     * Dereference operators (must be called within RCU read section)
     */
    T& operator*() const {
        T* ptr = load();
        if (!ptr) throw std::runtime_error("Dereferencing null RcuPtr");
        return *ptr;
    }
    
    T* operator->() const {
        T* ptr = load();
        if (!ptr) throw std::runtime_error("Dereferencing null RcuPtr");
        return ptr;
    }
    
    /**
     * Check if pointer is null
     */
    bool is_null() const noexcept {
        return load() == nullptr;
    }
    
    explicit operator bool() const noexcept {
        return !is_null();
    }
    
private:
    std::atomic<T*> ptr_;
};

/**
 * RCU-protected shared pointer with reference counting
 */
template<typename T>
class RcuSharedPtr {
public:
    RcuSharedPtr() : control_block_(nullptr) {}
    
    explicit RcuSharedPtr(T* ptr) {
        if (ptr) {
            control_block_ = new ControlBlock(ptr);
        }
    }
    
    RcuSharedPtr(const RcuSharedPtr& other) {
        ControlBlock* cb = other.control_block_.load();
        if (cb && cb->try_increment_ref()) {
            control_block_.store(cb);
        }
    }
    
    RcuSharedPtr& operator=(const RcuSharedPtr& other) {
        if (this != &other) {
            reset();
            ControlBlock* cb = other.control_block_.load();
            if (cb && cb->try_increment_ref()) {
                control_block_.store(cb);
            }
        }
        return *this;
    }
    
    ~RcuSharedPtr() {
        reset();
    }
    
    /**
     * Get raw pointer (must be called within RCU read section)
     */
    T* get() const noexcept {
        ControlBlock* cb = control_block_.load(std::memory_order_acquire);
        return cb ? cb->ptr : nullptr;
    }
    
    /**
     * Reset to null
     */
    void reset() {
        ControlBlock* old_cb = control_block_.exchange(nullptr);
        if (old_cb) {
            old_cb->decrement_ref();
        }
    }
    
    /**
     * Reset with new pointer
     */
    void reset(T* ptr) {
        ControlBlock* new_cb = ptr ? new ControlBlock(ptr) : nullptr;
        ControlBlock* old_cb = control_block_.exchange(new_cb);
        if (old_cb) {
            old_cb->decrement_ref();
        }
    }
    
    /**
     * Get reference count
     */
    long use_count() const noexcept {
        ControlBlock* cb = control_block_.load();
        return cb ? cb->ref_count.load() : 0;
    }
    
    /**
     * Check if unique owner
     */
    bool unique() const noexcept {
        return use_count() == 1;
    }
    
    /**
     * Dereference operators (must be called within RCU read section)
     */
    T& operator*() const {
        T* ptr = get();
        if (!ptr) throw std::runtime_error("Dereferencing null RcuSharedPtr");
        return *ptr;
    }
    
    T* operator->() const {
        T* ptr = get();
        if (!ptr) throw std::runtime_error("Dereferencing null RcuSharedPtr");
        return ptr;
    }
    
    explicit operator bool() const noexcept {
        return get() != nullptr;
    }
    
private:
    struct ControlBlock {
        T* ptr;
        std::atomic<long> ref_count;
        
        explicit ControlBlock(T* p) : ptr(p), ref_count(1) {}
        
        bool try_increment_ref() noexcept {
            long count = ref_count.load();
            do {
                if (count == 0) return false;
            } while (!ref_count.compare_exchange_weak(count, count + 1));
            return true;
        }
        
        void decrement_ref() noexcept {
            if (ref_count.fetch_sub(1) == 1) {
                RcuManager::instance().defer_delete([this]() {
                    delete ptr;
                    delete this;
                });
            }
        }
    };
    
    std::atomic<ControlBlock*> control_block_;
};

/**
 * Factory function for RCU shared pointer
 */
template<typename T, typename... Args>
RcuSharedPtr<T> make_rcu_shared(Args&&... args) {
    return RcuSharedPtr<T>(new T(std::forward<Args>(args)...));
}

} // namespace memory
} // namespace ultra