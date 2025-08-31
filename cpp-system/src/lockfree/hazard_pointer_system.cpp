#include "lockfree/atomic_ref_count.hpp"
#include <algorithm>
#include <thread>

namespace ultra_cpp {
namespace lockfree {

// Thread-local storage for retired objects
thread_local std::array<HazardPointerSystem::RetiredObject, HazardPointerSystem::MAX_RETIRED_OBJECTS> 
    HazardPointerSystem::retired_objects_;
thread_local size_t HazardPointerSystem::retired_count_ = 0;

HazardPointerSystem& HazardPointerSystem::instance() {
    static HazardPointerSystem instance;
    return instance;
}

HazardPointerSystem::HazardPointer* HazardPointerSystem::acquire_hazard_pointer() noexcept {
    // Try to find an inactive hazard pointer
    for (size_t i = 0; i < MAX_HAZARD_POINTERS; ++i) {
        HazardPointer& hp = hazard_pointers_[i];
        bool expected = false;
        if (hp.active.compare_exchange_weak(expected, true, std::memory_order_acq_rel)) {
            hp.ptr.store(nullptr, std::memory_order_release);
            stats_.hazard_pointers_acquired.fetch_add(1, std::memory_order_relaxed);
            return &hp;
        }
    }
    
    // No available hazard pointer
    return nullptr;
}

void HazardPointerSystem::release_hazard_pointer(HazardPointer* hp) noexcept {
    if (hp != nullptr) {
        hp->ptr.store(nullptr, std::memory_order_release);
        hp->active.store(false, std::memory_order_release);
        stats_.hazard_pointers_released.fetch_add(1, std::memory_order_relaxed);
    }
}

void HazardPointerSystem::retire_object(void* ptr, std::function<void(void*)> deleter) noexcept {
    if (ptr == nullptr) {
        return;
    }
    
    // Add to thread-local retired list
    if (retired_count_ < MAX_RETIRED_OBJECTS) {
        retired_objects_[retired_count_] = {
            ptr, 
            std::move(deleter), 
            global_epoch_.load(std::memory_order_acquire)
        };
        ++retired_count_;
        stats_.objects_retired.fetch_add(1, std::memory_order_relaxed);
    } else {
        // List is full, try to reclaim some objects
        scan_and_reclaim();
        
        // If still full, delete immediately (fallback)
        if (retired_count_ >= MAX_RETIRED_OBJECTS) {
            deleter(ptr);
            stats_.objects_reclaimed.fetch_add(1, std::memory_order_relaxed);
        } else {
            retired_objects_[retired_count_] = {ptr, std::move(deleter), global_epoch_.load(std::memory_order_acquire)};
            ++retired_count_;
            stats_.objects_retired.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

void HazardPointerSystem::scan_and_reclaim() noexcept {
    if (retired_count_ == 0) {
        return;
    }
    
    stats_.scan_cycles.fetch_add(1, std::memory_order_relaxed);
    
    // Advance global epoch
    advance_epoch();
    
    // Scan retired objects and reclaim safe ones
    size_t write_index = 0;
    for (size_t read_index = 0; read_index < retired_count_; ++read_index) {
        RetiredObject& obj = retired_objects_[read_index];
        
        if (!is_hazardous(obj.ptr)) {
            // Safe to reclaim
            obj.deleter(obj.ptr);
            stats_.objects_reclaimed.fetch_add(1, std::memory_order_relaxed);
        } else {
            // Still hazardous, keep in list
            if (write_index != read_index) {
                retired_objects_[write_index] = std::move(obj);
            }
            ++write_index;
        }
    }
    
    retired_count_ = write_index;
}

bool HazardPointerSystem::is_hazardous(void* ptr) const noexcept {
    for (const auto& hp : hazard_pointers_) {
        if (hp.active.load(std::memory_order_acquire) && 
            hp.ptr.load(std::memory_order_acquire) == ptr) {
            return true;
        }
    }
    return false;
}

void HazardPointerSystem::advance_epoch() noexcept {
    global_epoch_.fetch_add(1, std::memory_order_acq_rel);
}

// HazardPointerGuard implementation

HazardPointerGuard::HazardPointerGuard(void* ptr) noexcept 
    : hp_(HazardPointerSystem::instance().acquire_hazard_pointer()) {
    if (hp_ != nullptr && ptr != nullptr) {
        hp_->ptr.store(ptr, std::memory_order_release);
    }
}

HazardPointerGuard::~HazardPointerGuard() noexcept {
    if (hp_ != nullptr) {
        HazardPointerSystem::instance().release_hazard_pointer(hp_);
    }
}

HazardPointerGuard::HazardPointerGuard(HazardPointerGuard&& other) noexcept 
    : hp_(other.hp_) {
    other.hp_ = nullptr;
}

HazardPointerGuard& HazardPointerGuard::operator=(HazardPointerGuard&& other) noexcept {
    if (this != &other) {
        if (hp_ != nullptr) {
            HazardPointerSystem::instance().release_hazard_pointer(hp_);
        }
        hp_ = other.hp_;
        other.hp_ = nullptr;
    }
    return *this;
}

void HazardPointerGuard::protect(void* ptr) noexcept {
    if (hp_ != nullptr) {
        hp_->ptr.store(ptr, std::memory_order_release);
    }
}

void HazardPointerGuard::reset() noexcept {
    if (hp_ != nullptr) {
        hp_->ptr.store(nullptr, std::memory_order_release);
    }
}

} // namespace lockfree
} // namespace ultra_cpp