#pragma once

#include "lockfree_hash_table.hpp"
#include <thread>
#include <chrono>

namespace ultra_cpp {
namespace lockfree {

template<typename Key, typename Value, size_t Capacity>
HashTable<Key, Value, Capacity>::HashTable() {
    // Initialize all entries to empty state
    for (auto& entry : table_) {
        entry.key_hash.store(Entry::EMPTY_HASH, std::memory_order_relaxed);
        entry.value.store(nullptr, std::memory_order_relaxed);
        entry.version.store(0, std::memory_order_relaxed);
    }
}

template<typename Key, typename Value, size_t Capacity>
HashTable<Key, Value, Capacity>::~HashTable() {
    // Clean up all allocated values
    for (auto& entry : table_) {
        Value* val = entry.value.load(std::memory_order_relaxed);
        if (val != nullptr) {
            delete val;
        }
    }
}

template<typename Key, typename Value, size_t Capacity>
std::optional<Value> HashTable<Key, Value, Capacity>::get(const Key& key) const noexcept {
    const uint64_t key_hash = hash(key);
    size_t slot = key_hash & (Capacity - 1);
    size_t probe_distance = 0;
    
    while (probe_distance < Capacity) {
        const Entry& entry = table_[slot];
        
        // Load with acquire semantics to ensure we see consistent state
        uint64_t stored_hash = entry.key_hash.load(std::memory_order_acquire);
        
        if (stored_hash == Entry::EMPTY_HASH) {
            // Empty slot means key not found
            return std::nullopt;
        }
        
        if (stored_hash == key_hash) {
            // Potential match, load value with acquire semantics
            Value* val_ptr = entry.value.load(std::memory_order_acquire);
            if (val_ptr != nullptr) {
                return *val_ptr;
            }
        }
        
        slot = next_slot(slot);
        ++probe_distance;
    }
    
    return std::nullopt;
}

template<typename Key, typename Value, size_t Capacity>
bool HashTable<Key, Value, Capacity>::put(const Key& key, const Value& value) noexcept {
    const uint64_t key_hash = hash(key);
    size_t slot = key_hash & (Capacity - 1);
    size_t probe_distance = 0;
    
    // Allocate new value
    Value* new_value = new(std::nothrow) Value(value);
    if (!new_value) {
        return false; // Allocation failed
    }
    
    while (probe_distance < Capacity) {
        Entry& entry = table_[slot];
        
        uint64_t expected_hash = Entry::EMPTY_HASH;
        
        // Try to claim empty slot
        if (entry.key_hash.compare_exchange_weak(
                expected_hash, key_hash, 
                std::memory_order_acq_rel, std::memory_order_acquire)) {
            
            // Successfully claimed slot, store value
            entry.value.store(new_value, std::memory_order_release);
            stats_.size.fetch_add(1, std::memory_order_relaxed);
            
            // Update max probe distance
            uint64_t current_max = stats_.max_probe_distance.load(std::memory_order_relaxed);
            while (probe_distance > current_max && 
                   !stats_.max_probe_distance.compare_exchange_weak(
                       current_max, probe_distance, std::memory_order_relaxed)) {
                // Retry if another thread updated it
            }
            
            return true;
        }
        
        // Check if this is an update to existing key
        if (expected_hash == key_hash) {
            Value* old_value = entry.value.load(std::memory_order_acquire);
            if (old_value != nullptr) {
                // Update existing entry
                if (entry.value.compare_exchange_strong(
                        old_value, new_value, 
                        std::memory_order_acq_rel, std::memory_order_acquire)) {
                    
                    // Increment version for RCU
                    entry.version.fetch_add(1, std::memory_order_release);
                    
                    // Defer deletion of old value
                    defer_free(old_value);
                    return true;
                }
            }
        }
        
        // Slot occupied, continue probing
        slot = next_slot(slot);
        ++probe_distance;
        
        if (probe_distance > 0) {
            stats_.collisions.fetch_add(1, std::memory_order_relaxed);
        }
    }
    
    // Table full
    delete new_value;
    return false;
}

template<typename Key, typename Value, size_t Capacity>
bool HashTable<Key, Value, Capacity>::remove(const Key& key) noexcept {
    const uint64_t key_hash = hash(key);
    size_t slot = key_hash & (Capacity - 1);
    size_t probe_distance = 0;
    
    while (probe_distance < Capacity) {
        Entry& entry = table_[slot];
        
        uint64_t stored_hash = entry.key_hash.load(std::memory_order_acquire);
        
        if (stored_hash == Entry::EMPTY_HASH) {
            return false; // Key not found
        }
        
        if (stored_hash == key_hash) {
            // Found the key, mark as deleted
            Value* old_value = entry.value.exchange(nullptr, std::memory_order_acq_rel);
            if (old_value != nullptr) {
                entry.key_hash.store(Entry::DELETED_HASH, std::memory_order_release);
                entry.version.fetch_add(1, std::memory_order_release);
                
                stats_.size.fetch_sub(1, std::memory_order_relaxed);
                defer_free(old_value);
                return true;
            }
        }
        
        slot = next_slot(slot);
        ++probe_distance;
    }
    
    return false;
}

template<typename Key, typename Value, size_t Capacity>
uint64_t HashTable<Key, Value, Capacity>::hash(const Key& key) const noexcept {
    // Default hash using std::hash
    std::hash<Key> hasher;
    uint64_t h = hasher(key);
    return h == 0 ? 2 : h; // Avoid reserved values
}

template<typename Key, typename Value, size_t Capacity>
size_t HashTable<Key, Value, Capacity>::next_slot(size_t slot) const noexcept {
    return (slot + 1) & (Capacity - 1);
}

template<typename Key, typename Value, size_t Capacity>
void HashTable<Key, Value, Capacity>::rcu_synchronize() const noexcept {
    // Simple RCU synchronization - wait for all threads to pass through
    // a quiescent state. In a real implementation, this would be more sophisticated.
    std::this_thread::sleep_for(std::chrono::microseconds(1));
}

template<typename Key, typename Value, size_t Capacity>
void HashTable<Key, Value, Capacity>::defer_free(Value* ptr) const noexcept {
    // In a real RCU implementation, this would add to a deferred free list
    // For now, we'll use a simple synchronization approach
    std::thread([ptr]() {
        std::this_thread::sleep_for(std::chrono::microseconds(1));
        delete ptr;
    }).detach();
}

} // namespace lockfree
} // namespace ultra_cpp