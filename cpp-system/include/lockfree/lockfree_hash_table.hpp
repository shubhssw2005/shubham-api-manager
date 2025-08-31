#pragma once

#include <atomic>
#include <memory>
#include <functional>
#include <optional>
#include <cstdint>
#include <array>

namespace ultra_cpp {
namespace lockfree {

/**
 * Lock-free hash table with linear probing and RCU semantics
 * Provides O(1) average case operations with high concurrency
 */
template<typename Key, typename Value, size_t Capacity = 1024 * 1024>
class HashTable {
public:
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be power of 2");
    
    struct Entry {
        std::atomic<uint64_t> key_hash{0};
        std::atomic<Value*> value{nullptr};
        std::atomic<uint32_t> version{0};
        
        static constexpr uint64_t EMPTY_HASH = 0;
        static constexpr uint64_t DELETED_HASH = 1;
    };
    
    HashTable();
    ~HashTable();
    
    // Non-copyable, non-movable for safety
    HashTable(const HashTable&) = delete;
    HashTable& operator=(const HashTable&) = delete;
    HashTable(HashTable&&) = delete;
    HashTable& operator=(HashTable&&) = delete;
    
    // Core operations
    std::optional<Value> get(const Key& key) const noexcept;
    bool put(const Key& key, const Value& value) noexcept;
    bool remove(const Key& key) noexcept;
    
    // Statistics
    struct Stats {
        std::atomic<uint64_t> size{0};
        std::atomic<uint64_t> collisions{0};
        std::atomic<uint64_t> max_probe_distance{0};
    };
    
    const Stats& get_stats() const noexcept { return stats_; }
    
    // Capacity management
    size_t capacity() const noexcept { return Capacity; }
    size_t size() const noexcept { return stats_.size.load(std::memory_order_relaxed); }
    double load_factor() const noexcept { 
        return static_cast<double>(size()) / capacity(); 
    }
    
private:
    alignas(64) std::array<Entry, Capacity> table_;
    mutable Stats stats_;
    
    // Hash function
    uint64_t hash(const Key& key) const noexcept;
    
    // Linear probing helpers
    size_t find_slot(uint64_t key_hash) const noexcept;
    size_t next_slot(size_t slot) const noexcept;
    
    // RCU helpers for safe memory reclamation
    void rcu_synchronize() const noexcept;
    void defer_free(Value* ptr) const noexcept;
};



} // namespace lockfree
} // namespace ultra_cpp

#include "lockfree_hash_table_impl.hpp"