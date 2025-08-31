#include "common/types.hpp"
#include <atomic>
#include <memory>
#include <chrono>

namespace ultra::cache {

template<typename Key, typename Value>
class LRUEvictionPolicy {
public:
    struct LRUNode {
        alignas(CACHE_LINE_SIZE) Key key;
        alignas(CACHE_LINE_SIZE) Value value;
        alignas(CACHE_LINE_SIZE) std::atomic<LRUNode*> prev{nullptr};
        alignas(CACHE_LINE_SIZE) std::atomic<LRUNode*> next{nullptr};
        alignas(CACHE_LINE_SIZE) std::atomic<u64> access_time{0};
        alignas(CACHE_LINE_SIZE) std::atomic<u32> access_count{0};
        alignas(CACHE_LINE_SIZE) std::atomic<bool> deleted{false};
        
        LRUNode() = default;
        LRUNode(const Key& k, const Value& v) : key(k), value(v) {
            auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            access_time.store(now, std::memory_order_relaxed);
            access_count.store(1, std::memory_order_relaxed);
        }
    };
    
    explicit LRUEvictionPolicy(size_t max_size) 
        : max_size_(max_size)
        , current_size_(0) {
        
        // Initialize sentinel nodes for doubly-linked list
        head_ = std::make_unique<LRUNode>();
        tail_ = std::make_unique<LRUNode>();
        
        head_->next.store(tail_.get(), std::memory_order_relaxed);
        tail_->prev.store(head_.get(), std::memory_order_relaxed);
    }
    
    ~LRUEvictionPolicy() {
        clear();
    }
    
    std::optional<Value> get(const Key& key) {
        auto node = find_node(key);
        if (node && !node->deleted.load(std::memory_order_acquire)) {
            // Update access information
            auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            node->access_time.store(now, std::memory_order_relaxed);
            node->access_count.fetch_add(1, std::memory_order_relaxed);
            
            // Move to front (most recently used)
            move_to_front(node);
            
            return node->value;
        }
        return std::nullopt;
    }
    
    bool put(const Key& key, const Value& value) {
        // Check if key already exists
        auto existing_node = find_node(key);
        if (existing_node && !existing_node->deleted.load(std::memory_order_acquire)) {
            // Update existing node
            existing_node->value = value;
            auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            existing_node->access_time.store(now, std::memory_order_relaxed);
            existing_node->access_count.fetch_add(1, std::memory_order_relaxed);
            
            move_to_front(existing_node);
            return false; // No eviction occurred
        }
        
        // Create new node
        auto new_node = std::make_shared<LRUNode>(key, value);
        
        // Check if we need to evict
        bool evicted = false;
        if (current_size_.load(std::memory_order_relaxed) >= max_size_) {
            evicted = evict_lru();
        }
        
        // Add new node to front
        add_to_front(new_node);
        current_size_.fetch_add(1, std::memory_order_relaxed);
        
        return evicted;
    }
    
    bool remove(const Key& key) {
        auto node = find_node(key);
        if (node && !node->deleted.load(std::memory_order_acquire)) {
            remove_node(node);
            return true;
        }
        return false;
    }
    
    void clear() {
        // Mark all nodes as deleted and reset list
        auto current = head_->next.load(std::memory_order_acquire);
        while (current != tail_.get()) {
            auto next = current->next.load(std::memory_order_acquire);
            current->deleted.store(true, std::memory_order_release);
            current = next;
        }
        
        head_->next.store(tail_.get(), std::memory_order_release);
        tail_->prev.store(head_.get(), std::memory_order_release);
        current_size_.store(0, std::memory_order_release);
    }
    
    size_t size() const {
        return current_size_.load(std::memory_order_relaxed);
    }
    
    size_t max_size() const {
        return max_size_;
    }
    
    double load_factor() const {
        return static_cast<double>(size()) / max_size();
    }

private:
    size_t max_size_;
    std::atomic<size_t> current_size_;
    std::unique_ptr<LRUNode> head_;
    std::unique_ptr<LRUNode> tail_;
    
    // Simple hash table for O(1) key lookup
    static constexpr size_t HASH_TABLE_SIZE = 65536;
    std::array<std::atomic<std::shared_ptr<LRUNode>>, HASH_TABLE_SIZE> hash_table_;
    
    std::shared_ptr<LRUNode> find_node(const Key& key) {
        size_t hash = std::hash<Key>{}(key) % HASH_TABLE_SIZE;
        
        // Linear probing for collision resolution
        for (size_t i = 0; i < HASH_TABLE_SIZE; ++i) {
            size_t index = (hash + i) % HASH_TABLE_SIZE;
            auto node = hash_table_[index].load(std::memory_order_acquire);
            
            if (!node) {
                return nullptr; // Empty slot, key not found
            }
            
            if (node->key == key && !node->deleted.load(std::memory_order_acquire)) {
                return node;
            }
        }
        
        return nullptr;
    }
    
    void add_to_hash_table(std::shared_ptr<LRUNode> node) {
        size_t hash = std::hash<Key>{}(node->key) % HASH_TABLE_SIZE;
        
        // Linear probing to find empty slot
        for (size_t i = 0; i < HASH_TABLE_SIZE; ++i) {
            size_t index = (hash + i) % HASH_TABLE_SIZE;
            std::shared_ptr<LRUNode> expected = nullptr;
            
            if (hash_table_[index].compare_exchange_strong(expected, node, std::memory_order_acq_rel)) {
                return; // Successfully inserted
            }
        }
        
        // Hash table is full - this shouldn't happen in normal operation
        // In production, we would resize the hash table
    }
    
    void remove_from_hash_table(const Key& key) {
        size_t hash = std::hash<Key>{}(key) % HASH_TABLE_SIZE;
        
        // Linear probing to find the key
        for (size_t i = 0; i < HASH_TABLE_SIZE; ++i) {
            size_t index = (hash + i) % HASH_TABLE_SIZE;
            auto node = hash_table_[index].load(std::memory_order_acquire);
            
            if (!node) {
                return; // Key not found
            }
            
            if (node->key == key) {
                // Mark slot as empty
                hash_table_[index].store(nullptr, std::memory_order_release);
                return;
            }
        }
    }
    
    void add_to_front(std::shared_ptr<LRUNode> node) {
        // Add to hash table first
        add_to_hash_table(node);
        
        // Add to front of doubly-linked list
        auto old_first = head_->next.load(std::memory_order_acquire);
        
        node->next.store(old_first, std::memory_order_relaxed);
        node->prev.store(head_.get(), std::memory_order_relaxed);
        
        old_first->prev.store(node.get(), std::memory_order_release);
        head_->next.store(node.get(), std::memory_order_release);
    }
    
    void move_to_front(std::shared_ptr<LRUNode> node) {
        // Remove from current position
        auto prev_node = node->prev.load(std::memory_order_acquire);
        auto next_node = node->next.load(std::memory_order_acquire);
        
        if (prev_node) {
            prev_node->next.store(next_node, std::memory_order_release);
        }
        if (next_node) {
            next_node->prev.store(prev_node, std::memory_order_release);
        }
        
        // Add to front
        auto old_first = head_->next.load(std::memory_order_acquire);
        
        node->next.store(old_first, std::memory_order_relaxed);
        node->prev.store(head_.get(), std::memory_order_relaxed);
        
        old_first->prev.store(node.get(), std::memory_order_release);
        head_->next.store(node.get(), std::memory_order_release);
    }
    
    void remove_node(std::shared_ptr<LRUNode> node) {
        // Remove from hash table
        remove_from_hash_table(node->key);
        
        // Remove from doubly-linked list
        auto prev_node = node->prev.load(std::memory_order_acquire);
        auto next_node = node->next.load(std::memory_order_acquire);
        
        if (prev_node) {
            prev_node->next.store(next_node, std::memory_order_release);
        }
        if (next_node) {
            next_node->prev.store(prev_node, std::memory_order_release);
        }
        
        node->deleted.store(true, std::memory_order_release);
        current_size_.fetch_sub(1, std::memory_order_relaxed);
    }
    
    bool evict_lru() {
        // Find least recently used node (at the tail)
        auto lru_node = tail_->prev.load(std::memory_order_acquire);
        
        if (lru_node == head_.get()) {
            return false; // List is empty
        }
        
        // Cast to shared_ptr for removal
        // In a real implementation, we'd maintain shared_ptrs properly
        auto shared_lru = find_node(lru_node->key);
        if (shared_lru) {
            remove_node(shared_lru);
            return true;
        }
        
        return false;
    }
};

} // namespace ultra::cache