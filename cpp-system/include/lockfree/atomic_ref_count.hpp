#pragma once

#include <atomic>
#include <memory>
#include <functional>
#include <type_traits>

namespace ultra_cpp {
namespace lockfree {

/**
 * Atomic reference counting for safe memory reclamation
 * Provides thread-safe reference counting with automatic cleanup
 */
template<typename T>
class AtomicRefCount {
public:
    explicit AtomicRefCount(T* ptr = nullptr) noexcept;
    ~AtomicRefCount() noexcept;
    
    // Copy constructor and assignment
    AtomicRefCount(const AtomicRefCount& other) noexcept;
    AtomicRefCount& operator=(const AtomicRefCount& other) noexcept;
    
    // Move constructor and assignment
    AtomicRefCount(AtomicRefCount&& other) noexcept;
    AtomicRefCount& operator=(AtomicRefCount&& other) noexcept;
    
    // Access operators
    T* operator->() const noexcept;
    T& operator*() const noexcept;
    T* get() const noexcept;
    
    // Reference management
    void reset(T* ptr = nullptr) noexcept;
    T* release() noexcept;
    
    // Status queries
    bool empty() const noexcept;
    explicit operator bool() const noexcept;
    
    // Reference count queries
    uint32_t use_count() const noexcept;
    bool unique() const noexcept;
    
    // Comparison operators
    bool operator==(const AtomicRefCount& other) const noexcept;
    bool operator!=(const AtomicRefCount& other) const noexcept;
    bool operator<(const AtomicRefCount& other) const noexcept;
    
private:
    struct ControlBlock {
        std::atomic<uint32_t> ref_count{1};
        std::atomic<T*> ptr{nullptr};
        std::function<void(T*)> deleter;
        
        explicit ControlBlock(T* p, std::function<void(T*)> del = std::default_delete<T>{})
            : ptr(p), deleter(std::move(del)) {}
    };
    
    std::atomic<ControlBlock*> control_block_{nullptr};
    
    void acquire() noexcept;
    void release_ref() noexcept;
    void acquire_control_block(ControlBlock* cb) noexcept;
    void release_control_block() noexcept;
};

/**
 * Hazard pointer system for lock-free memory reclamation
 * Provides epoch-based memory management for lock-free data structures
 */
class HazardPointerSystem {
public:
    static constexpr size_t MAX_HAZARD_POINTERS = 64;
    static constexpr size_t MAX_RETIRED_OBJECTS = 1024;
    
    struct HazardPointer {
        std::atomic<void*> ptr{nullptr};
        std::atomic<bool> active{false};
        alignas(64) char padding[64 - sizeof(std::atomic<void*>) - sizeof(std::atomic<bool>)];
    };
    
    static HazardPointerSystem& instance();
    
    // Hazard pointer management
    HazardPointer* acquire_hazard_pointer() noexcept;
    void release_hazard_pointer(HazardPointer* hp) noexcept;
    
    // Memory reclamation
    void retire_object(void* ptr, std::function<void(void*)> deleter) noexcept;
    void scan_and_reclaim() noexcept;
    
    // Statistics
    struct Stats {
        std::atomic<uint64_t> hazard_pointers_acquired{0};
        std::atomic<uint64_t> hazard_pointers_released{0};
        std::atomic<uint64_t> objects_retired{0};
        std::atomic<uint64_t> objects_reclaimed{0};
        std::atomic<uint64_t> scan_cycles{0};
    };
    
    const Stats& get_stats() const noexcept { return stats_; }
    
private:
    HazardPointerSystem() = default;
    ~HazardPointerSystem() = default;
    
    struct RetiredObject {
        void* ptr;
        std::function<void(void*)> deleter;
        uint64_t retire_epoch;
    };
    
    alignas(64) std::array<HazardPointer, MAX_HAZARD_POINTERS> hazard_pointers_;
    alignas(64) std::atomic<uint32_t> next_hp_index_{0};
    
    // Per-thread retired object lists
    thread_local static std::array<RetiredObject, MAX_RETIRED_OBJECTS> retired_objects_;
    thread_local static size_t retired_count_;
    
    alignas(64) std::atomic<uint64_t> global_epoch_{0};
    alignas(64) mutable Stats stats_;
    
    bool is_hazardous(void* ptr) const noexcept;
    void advance_epoch() noexcept;
};

/**
 * RAII wrapper for hazard pointers
 */
class HazardPointerGuard {
public:
    explicit HazardPointerGuard(void* ptr = nullptr) noexcept;
    ~HazardPointerGuard() noexcept;
    
    // Non-copyable, movable
    HazardPointerGuard(const HazardPointerGuard&) = delete;
    HazardPointerGuard& operator=(const HazardPointerGuard&) = delete;
    
    HazardPointerGuard(HazardPointerGuard&& other) noexcept;
    HazardPointerGuard& operator=(HazardPointerGuard&& other) noexcept;
    
    // Pointer management
    void protect(void* ptr) noexcept;
    void reset() noexcept;
    
    template<typename T>
    T* get() const noexcept {
        return static_cast<T*>(hp_ ? hp_->ptr.load(std::memory_order_acquire) : nullptr);
    }
    
private:
    HazardPointerSystem::HazardPointer* hp_{nullptr};
};

// Implementation

template<typename T>
AtomicRefCount<T>::AtomicRefCount(T* ptr) noexcept {
    if (ptr != nullptr) {
        control_block_.store(new ControlBlock(ptr), std::memory_order_release);
    }
}

template<typename T>
AtomicRefCount<T>::~AtomicRefCount() noexcept {
    release_control_block();
}

template<typename T>
AtomicRefCount<T>::AtomicRefCount(const AtomicRefCount& other) noexcept {
    ControlBlock* cb = other.control_block_.load(std::memory_order_acquire);
    if (cb != nullptr) {
        acquire_control_block(cb);
        control_block_.store(cb, std::memory_order_release);
    }
}

template<typename T>
AtomicRefCount<T>& AtomicRefCount<T>::operator=(const AtomicRefCount& other) noexcept {
    if (this != &other) {
        ControlBlock* new_cb = other.control_block_.load(std::memory_order_acquire);
        ControlBlock* old_cb = control_block_.exchange(new_cb, std::memory_order_acq_rel);
        
        if (new_cb != nullptr) {
            acquire_control_block(new_cb);
        }
        
        if (old_cb != nullptr) {
            release_control_block();
        }
    }
    return *this;
}

template<typename T>
AtomicRefCount<T>::AtomicRefCount(AtomicRefCount&& other) noexcept {
    control_block_.store(other.control_block_.exchange(nullptr, std::memory_order_acq_rel), 
                        std::memory_order_release);
}

template<typename T>
AtomicRefCount<T>& AtomicRefCount<T>::operator=(AtomicRefCount&& other) noexcept {
    if (this != &other) {
        ControlBlock* old_cb = control_block_.exchange(
            other.control_block_.exchange(nullptr, std::memory_order_acq_rel),
            std::memory_order_acq_rel);
        
        if (old_cb != nullptr) {
            release_control_block();
        }
    }
    return *this;
}

template<typename T>
T* AtomicRefCount<T>::operator->() const noexcept {
    ControlBlock* cb = control_block_.load(std::memory_order_acquire);
    return cb ? cb->ptr.load(std::memory_order_acquire) : nullptr;
}

template<typename T>
T& AtomicRefCount<T>::operator*() const noexcept {
    return *get();
}

template<typename T>
T* AtomicRefCount<T>::get() const noexcept {
    ControlBlock* cb = control_block_.load(std::memory_order_acquire);
    return cb ? cb->ptr.load(std::memory_order_acquire) : nullptr;
}

template<typename T>
void AtomicRefCount<T>::reset(T* ptr) noexcept {
    ControlBlock* new_cb = ptr ? new ControlBlock(ptr) : nullptr;
    ControlBlock* old_cb = control_block_.exchange(new_cb, std::memory_order_acq_rel);
    
    if (old_cb != nullptr) {
        release_control_block();
    }
}

template<typename T>
T* AtomicRefCount<T>::release() noexcept {
    ControlBlock* cb = control_block_.exchange(nullptr, std::memory_order_acq_rel);
    if (cb == nullptr) {
        return nullptr;
    }
    
    T* ptr = cb->ptr.exchange(nullptr, std::memory_order_acq_rel);
    release_control_block();
    return ptr;
}

template<typename T>
bool AtomicRefCount<T>::empty() const noexcept {
    return get() == nullptr;
}

template<typename T>
AtomicRefCount<T>::operator bool() const noexcept {
    return !empty();
}

template<typename T>
uint32_t AtomicRefCount<T>::use_count() const noexcept {
    ControlBlock* cb = control_block_.load(std::memory_order_acquire);
    return cb ? cb->ref_count.load(std::memory_order_acquire) : 0;
}

template<typename T>
bool AtomicRefCount<T>::unique() const noexcept {
    return use_count() == 1;
}

template<typename T>
void AtomicRefCount<T>::acquire_control_block(ControlBlock* cb) noexcept {
    if (cb != nullptr) {
        cb->ref_count.fetch_add(1, std::memory_order_acq_rel);
    }
}

template<typename T>
void AtomicRefCount<T>::release_control_block() noexcept {
    ControlBlock* cb = control_block_.load(std::memory_order_acquire);
    if (cb != nullptr && cb->ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        T* ptr = cb->ptr.load(std::memory_order_acquire);
        if (ptr != nullptr) {
            cb->deleter(ptr);
        }
        delete cb;
    }
}

} // namespace lockfree
} // namespace ultra_cpp