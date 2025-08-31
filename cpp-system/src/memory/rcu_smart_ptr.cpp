#include "memory/rcu_smart_ptr.hpp"
#include <algorithm>
#include <cassert>

namespace ultra {
namespace memory {

// Thread-local reader state
thread_local RcuManager::ReaderState* RcuManager::tl_reader_state_ = nullptr;

RcuManager::RcuManager() : global_epoch_(1) {
    // Start cleanup thread
    cleanup_thread_ = std::thread(&RcuManager::cleanup_loop, this);
}

RcuManager::~RcuManager() {
    // Signal shutdown
    shutdown_.store(true, std::memory_order_release);
    
    // Wait for cleanup thread
    if (cleanup_thread_.joinable()) {
        cleanup_thread_.join();
    }
    
    // Process any remaining deletions
    process_deletions();
    
    // Clean up reader states
    reader_states_.clear();
}

RcuManager::EpochType RcuManager::read_lock() noexcept {
    ReaderState* state = get_reader_state();
    if (!state) return INVALID_EPOCH;
    
    // Mark as active and get current epoch
    state->active.store(true, std::memory_order_release);
    EpochType epoch = global_epoch_.load(std::memory_order_acquire);
    state->epoch.store(epoch, std::memory_order_release);
    
    stats_.read_sections.fetch_add(1, std::memory_order_relaxed);
    stats_.active_readers.fetch_add(1, std::memory_order_relaxed);
    
    return epoch;
}

void RcuManager::read_unlock(EpochType epoch) noexcept {
    ReaderState* state = get_reader_state();
    if (!state) return;
    
    // Verify epoch matches
    assert(state->epoch.load(std::memory_order_acquire) == epoch);
    
    // Mark as inactive
    state->active.store(false, std::memory_order_release);
    state->epoch.store(INVALID_EPOCH, std::memory_order_release);
    
    stats_.active_readers.fetch_sub(1, std::memory_order_relaxed);
}

void RcuManager::defer_delete(DeleterFunc deleter) {
    if (!deleter) return;
    
    // Allocate new deletion entry
    DeferredDeletion* deletion = new DeferredDeletion{
        std::move(deleter),
        global_epoch_.load(std::memory_order_acquire)
    };
    
    // Add to queue atomically
    DeferredDeletion* old_tail = deletion_queue_tail_.exchange(deletion, std::memory_order_acq_rel);
    if (old_tail) {
        // Link to previous tail
        old_tail->next = deletion;
    } else {
        // First item in queue
        deletion_queue_head_.store(deletion, std::memory_order_release);
    }
    
    stats_.deferred_deletions.fetch_add(1, std::memory_order_relaxed);
}

void RcuManager::synchronize() {
    // Advance global epoch
    EpochType old_epoch = global_epoch_.fetch_add(1, std::memory_order_acq_rel);
    EpochType new_epoch = old_epoch + 1;
    
    // Wait for all readers in old epoch to finish
    bool all_finished = false;
    while (!all_finished) {
        all_finished = true;
        
        for (const auto& state_ptr : reader_states_) {
            if (!state_ptr) continue;
            
            ReaderState* state = state_ptr.get();
            if (state->active.load(std::memory_order_acquire)) {
                EpochType reader_epoch = state->epoch.load(std::memory_order_acquire);
                if (reader_epoch <= old_epoch) {
                    all_finished = false;
                    break;
                }
            }
        }
        
        if (!all_finished) {
            // Brief pause before checking again
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        }
    }
    
    stats_.synchronizations.fetch_add(1, std::memory_order_relaxed);
}

RcuManager& RcuManager::instance() {
    static RcuManager instance;
    return instance;
}

RcuManager::ReaderState* RcuManager::get_reader_state() {
    if (tl_reader_state_) {
        return tl_reader_state_;
    }
    
    // Allocate new reader state
    auto state = std::make_unique<ReaderState>();
    tl_reader_state_ = state.get();
    
    // Add to global list (this is not lock-free, but only happens once per thread)
    static std::mutex states_mutex;
    std::lock_guard<std::mutex> lock(states_mutex);
    reader_states_.push_back(std::move(state));
    reader_count_.fetch_add(1, std::memory_order_relaxed);
    
    return tl_reader_state_;
}

void RcuManager::cleanup_loop() {
    while (!shutdown_.load(std::memory_order_acquire)) {
        process_deletions();
        
        // Sleep for a short period
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void RcuManager::process_deletions() {
    EpochType min_epoch = find_minimum_epoch();
    
    // Process deletions that are safe to execute
    DeferredDeletion* current = deletion_queue_head_.load(std::memory_order_acquire);
    DeferredDeletion* prev = nullptr;
    
    while (current) {
        if (current->epoch < min_epoch) {
            // Safe to delete
            try {
                current->deleter();
                stats_.completed_deletions.fetch_add(1, std::memory_order_relaxed);
            } catch (const std::exception& e) {
                // Log error but continue processing
                // In a real implementation, you'd use a proper logging system
            }
            
            // Remove from queue
            DeferredDeletion* to_delete = current;
            current = current->next;
            
            if (prev) {
                prev->next = current;
            } else {
                deletion_queue_head_.store(current, std::memory_order_release);
            }
            
            delete to_delete;
        } else {
            // Not safe yet, move to next
            prev = current;
            current = current->next;
        }
    }
    
    // Update tail if we removed the last item
    if (!current && prev) {
        deletion_queue_tail_.store(prev, std::memory_order_release);
    } else if (!current) {
        deletion_queue_tail_.store(nullptr, std::memory_order_release);
    }
}

RcuManager::EpochType RcuManager::find_minimum_epoch() const noexcept {
    EpochType min_epoch = global_epoch_.load(std::memory_order_acquire);
    
    for (const auto& state_ptr : reader_states_) {
        if (!state_ptr) continue;
        
        ReaderState* state = state_ptr.get();
        if (state->active.load(std::memory_order_acquire)) {
            EpochType reader_epoch = state->epoch.load(std::memory_order_acquire);
            if (reader_epoch != INVALID_EPOCH) {
                min_epoch = std::min(min_epoch, reader_epoch);
            }
        }
    }
    
    return min_epoch;
}

} // namespace memory
} // namespace ultra