#pragma once

#include <atomic>
#include <memory>
#include <optional>
#include <cstdint>
#include <array>

namespace ultra_cpp {
namespace lockfree {

/**
 * Lock-free LRU cache with O(1) operations
 * Uses atomic operations and careful memory ordering for thread safety
 */
template<typename Key, typename Value, size_t Capacity = 1024>
class LRUCache {
public:
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be power of 2");
    static_assert(Capacity >= 4, "Capacity must be at least 4");
    
    LRUCache();
    ~LRUCache();
    
    // Non-copyable, non-movable for safety
    LRUCache(const LRUCache&) = delete;
    LRUCache& operator=(const LRUCache&) = delete;
    LRUCache(LRUCache&&) = delete;
    LRUCache& operator=(LRUCache&&) = delete;
    
    // Core operations - all O(1)
    std::optional<Value> get(const Key& key) noexcept;
    bool put(const Key& key, const Value& value) noexcept;
    bool remove(const Key& key) noexcept;
    
    // Status queries
    size_t size() const noexcept;
    size_t capacity() const noexcept { return Capacity; }
    bool empty() const noexcept { return size() == 0; }
    bool full() const noexcept { return size() >= Capacity; }
    
    // Statistics
    struct Stats {
        std::atomic<uint64_t> hits{0};
        std::atomic<uint64_t> misses{0};
        std::atomic<uint64_t> evictions{0};
        std::atomic<uint64_t> insertions{0};
        std::atomic<uint64_t> updates{0};
    };
    
    const Stats& get_stats() const noexcept { return stats_; }
    void reset_stats() noexcept;
    
    // Cache efficiency metrics
    double hit_rate() const noexcept;
    
private:
    struct Node {
        std::atomic<uint64_t> key_hash{0};
        std::atomic<Value*> value{nullptr};
        std::atomic<Node*> prev{nullptr};
        std::atomic<Node*> next{nullptr};
        std::atomic<uint64_t> access_time{0};
        std::atomic<uint32_t> ref_count{0};
        
        static constexpr uint64_t EMPTY_HASH = 0;
        static constexpr uint64_t DELETED_HASH = 1;
    };
    
    // Hash table for O(1) lookup
    alignas(64) std::array<Node, Capacity> hash_table_;
    
    // LRU list sentinels
    alignas(64) Node head_sentinel_;
    alignas(64) Node tail_sentinel_;
    
    // Global state
    alignas(64) std::atomic<uint64_t> current_time_{0};
    alignas(64) std::atomic<size_t> size_{0};
    alignas(64) mutable Stats stats_;
    
    // Hash function
    uint64_t hash(const Key& key) const noexcept;
    
    // Hash table operations
    Node* find_node(const Key& key, uint64_t key_hash) const noexcept;
    Node* find_empty_slot(uint64_t key_hash) const noexcept;
    
    // LRU list operations (lock-free)
    void move_to_head(Node* node) noexcept;
    void remove_from_list(Node* node) noexcept;
    void add_to_head(Node* node) noexcept;
    Node* remove_tail() noexcept;
    
    // Reference counting for safe memory reclamation
    void acquire_ref(Node* node) noexcept;
    void release_ref(Node* node) noexcept;
    
    // Time management
    uint64_t get_current_time() noexcept;
    
    static constexpr size_t MASK = Capacity - 1;
};

template<typename Key, typename Value, size_t Capacity>
LRUCache<Key, Value, Capacity>::LRUCache() {
    // Initialize hash table
    for (auto& node : hash_table_) {
        node.key_hash.store(Node::EMPTY_HASH, std::memory_order_relaxed);
        node.value.store(nullptr, std::memory_order_relaxed);
        node.prev.store(nullptr, std::memory_order_relaxed);
        node.next.store(nullptr, std::memory_order_relaxed);
        node.access_time.store(0, std::memory_order_relaxed);
        node.ref_count.store(0, std::memory_order_relaxed);
    }
    
    // Initialize LRU list sentinels
    head_sentinel_.next.store(&tail_sentinel_, std::memory_order_relaxed);
    tail_sentinel_.prev.store(&head_sentinel_, std::memory_order_relaxed);
}

template<typename Key, typename Value, size_t Capacity>
LRUCache<Key, Value, Capacity>::~LRUCache() {
    // Clean up all values
    for (auto& node : hash_table_) {
        Value* val = node.value.load(std::memory_order_relaxed);
        if (val != nullptr) {
            delete val;
        }
    }
}

template<typename Key, typename Value, size_t Capacity>
std::optional<Value> LRUCache<Key, Value, Capacity>::get(const Key& key) noexcept {
    const uint64_t key_hash = hash(key);
    Node* node = find_node(key, key_hash);
    
    if (node == nullptr) {
        stats_.misses.fetch_add(1, std::memory_order_relaxed);
        return std::nullopt;
    }
    
    // Acquire reference to prevent deletion
    acquire_ref(node);
    
    // Load value with acquire semantics
    Value* val_ptr = node->value.load(std::memory_order_acquire);
    if (val_ptr == nullptr) {
        release_ref(node);
        stats_.misses.fetch_add(1, std::memory_order_relaxed);
        return std::nullopt;
    }
    
    // Update access time and move to head
    node->access_time.store(get_current_time(), std::memory_order_relaxed);
    move_to_head(node);
    
    Value result = *val_ptr;
    release_ref(node);
    
    stats_.hits.fetch_add(1, std::memory_order_relaxed);
    return result;
}

template<typename Key, typename Value, size_t Capacity>
bool LRUCache<Key, Value, Capacity>::put(const Key& key, const Value& value) noexcept {
    const uint64_t key_hash = hash(key);
    
    // Check if key already exists
    Node* existing_node = find_node(key, key_hash);
    if (existing_node != nullptr) {
        // Update existing entry
        Value* new_value = new(std::nothrow) Value(value);
        if (!new_value) {
            return false;
        }
        
        acquire_ref(existing_node);
        Value* old_value = existing_node->value.exchange(new_value, std::memory_order_acq_rel);
        existing_node->access_time.store(get_current_time(), std::memory_order_relaxed);
        move_to_head(existing_node);
        release_ref(existing_node);
        
        if (old_value) {
            delete old_value;
        }
        
        stats_.updates.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    
    // Find empty slot or evict if necessary
    Node* slot = find_empty_slot(key_hash);
    if (slot == nullptr) {
        // Cache is full, evict LRU item
        slot = remove_tail();
        if (slot == nullptr) {
            return false; // Failed to evict
        }
        
        // Clean up evicted slot
        Value* old_value = slot->value.exchange(nullptr, std::memory_order_acq_rel);
        if (old_value) {
            delete old_value;
        }
        slot->key_hash.store(Node::EMPTY_HASH, std::memory_order_release);
        
        stats_.evictions.fetch_add(1, std::memory_order_relaxed);
        size_.fetch_sub(1, std::memory_order_relaxed);
    }
    
    // Insert new entry
    Value* new_value = new(std::nothrow) Value(value);
    if (!new_value) {
        return false;
    }
    
    slot->value.store(new_value, std::memory_order_relaxed);
    slot->access_time.store(get_current_time(), std::memory_order_relaxed);
    slot->key_hash.store(key_hash, std::memory_order_release);
    
    add_to_head(slot);
    size_.fetch_add(1, std::memory_order_relaxed);
    stats_.insertions.fetch_add(1, std::memory_order_relaxed);
    
    return true;
}

template<typename Key, typename Value, size_t Capacity>
bool LRUCache<Key, Value, Capacity>::remove(const Key& key) noexcept {
    const uint64_t key_hash = hash(key);
    Node* node = find_node(key, key_hash);
    
    if (node == nullptr) {
        return false;
    }
    
    acquire_ref(node);
    
    // Mark as deleted and remove from list
    Value* old_value = node->value.exchange(nullptr, std::memory_order_acq_rel);
    node->key_hash.store(Node::DELETED_HASH, std::memory_order_release);
    remove_from_list(node);
    
    release_ref(node);
    
    if (old_value) {
        delete old_value;
        size_.fetch_sub(1, std::memory_order_relaxed);
        return true;
    }
    
    return false;
}

template<typename Key, typename Value, size_t Capacity>
size_t LRUCache<Key, Value, Capacity>::size() const noexcept {
    return size_.load(std::memory_order_acquire);
}

template<typename Key, typename Value, size_t Capacity>
void LRUCache<Key, Value, Capacity>::reset_stats() noexcept {
    stats_.hits.store(0, std::memory_order_relaxed);
    stats_.misses.store(0, std::memory_order_relaxed);
    stats_.evictions.store(0, std::memory_order_relaxed);
    stats_.insertions.store(0, std::memory_order_relaxed);
    stats_.updates.store(0, std::memory_order_relaxed);
}

template<typename Key, typename Value, size_t Capacity>
double LRUCache<Key, Value, Capacity>::hit_rate() const noexcept {
    uint64_t hits = stats_.hits.load(std::memory_order_relaxed);
    uint64_t misses = stats_.misses.load(std::memory_order_relaxed);
    uint64_t total = hits + misses;
    
    return total > 0 ? static_cast<double>(hits) / total : 0.0;
}

// Implementation details in separate file
} // namespace lockfree
} // namespace ultra_cpp

#include "lockfree_lru_cache_impl.hpp"