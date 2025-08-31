#include "lockfree/lockfree_hash_table.hpp"
#include "common/types.hpp"
#include <atomic>
#include <memory>
#include <cstring>

namespace ultra::cache {

template<typename Key, typename Value>
class LockFreeHashTable {
public:
    struct Entry {
        alignas(CACHE_LINE_SIZE) std::atomic<Key*> key{nullptr};
        alignas(CACHE_LINE_SIZE) std::atomic<Value*> value{nullptr};
        alignas(CACHE_LINE_SIZE) std::atomic<u64> hash{0};
        alignas(CACHE_LINE_SIZE) std::atomic<bool> deleted{false};
        alignas(CACHE_LINE_SIZE) std::atomic<u64> version{0};
    };
    
    explicit LockFreeHashTable(size_t initial_capacity = 1024)
        : capacity_(next_power_of_two(initial_capacity))
        , mask_(capacity_ - 1)
        , size_(0) {
        
        entries_ = std::make_unique<Entry[]>(capacity_);
        
        // Initialize all entries
        for (size_t i = 0; i < capacity_; ++i) {
            entries_[i].key.store(nullptr, std::memory_order_relaxed);
            entries_[i].value.store(nullptr, std::memory_order_relaxed);
            entries_[i].hash.store(0, std::memory_order_relaxed);
            entries_[i].deleted.store(false, std::memory_order_relaxed);
            entries_[i].version.store(0, std::memory_order_relaxed);
        }
    }
    
    ~LockFreeHashTable() {
        // Clean up all allocated keys and values
        for (size_t i = 0; i < capacity_; ++i) {
            auto* key = entries_[i].key.load(std::memory_order_relaxed);
            auto* value = entries_[i].value.load(std::memory_order_relaxed);
            
            if (key != nullptr) {
                delete key;
            }
            if (value != nullptr) {
                delete value;
            }
        }
    }
    
    std::optional<Value> get(const Key& key) {
        u64 hash = hash_key(key);
        size_t index = hash & mask_;
        
        // Linear probing with quadratic increment
        for (size_t probe = 0; probe < capacity_; ++probe) {
            size_t current_index = (index + probe * probe) & mask_;
            Entry& entry = entries_[current_index];
            
            // Load entry data atomically
            u64 entry_hash = entry.hash.load(std::memory_order_acquire);
            Key* entry_key = entry.key.load(std::memory_order_acquire);
            Value* entry_value = entry.value.load(std::memory_order_acquire);
            bool is_deleted = entry.deleted.load(std::memory_order_acquire);
            
            // Empty slot - key not found
            if (entry_key == nullptr) {
                return std::nullopt;
            }
            
            // Skip deleted entries
            if (is_deleted) {
                continue;
            }
            
            // Check if this is our key
            if (entry_hash == hash && *entry_key == key) {
                if (entry_value != nullptr) {
                    return *entry_value;
                }
                return std::nullopt;
            }
        }
        
        return std::nullopt;
    }
    
    bool put(const Key& key, const Value& value) {
        u64 hash = hash_key(key);
        size_t index = hash & mask_;
        
        // Check if we need to resize
        if (size_.load(std::memory_order_relaxed) > capacity_ * 0.75) {
            resize();
            index = hash & mask_;
        }
        
        // Linear probing with quadratic increment
        for (size_t probe = 0; probe < capacity_; ++probe) {
            size_t current_index = (index + probe * probe) & mask_;
            Entry& entry = entries_[current_index];
            
            Key* entry_key = entry.key.load(std::memory_order_acquire);
            
            // Empty slot or deleted slot - try to claim it
            if (entry_key == nullptr || entry.deleted.load(std::memory_order_acquire)) {
                Key* new_key = new Key(key);
                Value* new_value = new Value(value);
                
                // Try to atomically set the key
                Key* expected = nullptr;
                if (entry.key.compare_exchange_strong(expected, new_key, std::memory_order_acq_rel)) {
                    // Successfully claimed the slot
                    entry.value.store(new_value, std::memory_order_release);
                    entry.hash.store(hash, std::memory_order_release);
                    entry.deleted.store(false, std::memory_order_release);
                    entry.version.fetch_add(1, std::memory_order_acq_rel);
                    
                    size_.fetch_add(1, std::memory_order_relaxed);
                    return true;
                } else {
                    // Someone else claimed it, clean up and continue
                    delete new_key;
                    delete new_value;
                }
            }
            
            // Check if this is an update to existing key
            u64 entry_hash = entry.hash.load(std::memory_order_acquire);
            if (entry_hash == hash && *entry_key == key && !entry.deleted.load(std::memory_order_acquire)) {
                // Update existing entry
                Value* old_value = entry.value.load(std::memory_order_acquire);
                Value* new_value = new Value(value);
                
                if (entry.value.compare_exchange_strong(old_value, new_value, std::memory_order_acq_rel)) {
                    entry.version.fetch_add(1, std::memory_order_acq_rel);
                    delete old_value;
                    return true;
                } else {
                    delete new_value;
                }
            }
        }
        
        return false; // Table is full
    }
    
    bool remove(const Key& key) {
        u64 hash = hash_key(key);
        size_t index = hash & mask_;
        
        // Linear probing with quadratic increment
        for (size_t probe = 0; probe < capacity_; ++probe) {
            size_t current_index = (index + probe * probe) & mask_;
            Entry& entry = entries_[current_index];
            
            Key* entry_key = entry.key.load(std::memory_order_acquire);
            
            // Empty slot - key not found
            if (entry_key == nullptr) {
                return false;
            }
            
            // Check if this is our key
            u64 entry_hash = entry.hash.load(std::memory_order_acquire);
            if (entry_hash == hash && *entry_key == key && !entry.deleted.load(std::memory_order_acquire)) {
                // Mark as deleted
                bool expected = false;
                if (entry.deleted.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
                    entry.version.fetch_add(1, std::memory_order_acq_rel);
                    size_.fetch_sub(1, std::memory_order_relaxed);
                    return true;
                }
            }
        }
        
        return false;
    }
    
    size_t size() const {
        return size_.load(std::memory_order_relaxed);
    }
    
    size_t capacity() const {
        return capacity_;
    }
    
    double load_factor() const {
        return static_cast<double>(size()) / capacity();
    }

private:
    std::unique_ptr<Entry[]> entries_;
    size_t capacity_;
    size_t mask_;
    std::atomic<size_t> size_;
    
    ULTRA_FORCE_INLINE u64 hash_key(const Key& key) const {
        // Use a simple but effective hash function
        return std::hash<Key>{}(key);
    }
    
    static size_t next_power_of_two(size_t n) {
        if (n <= 1) return 2;
        
        n--;
        n |= n >> 1;
        n |= n >> 2;
        n |= n >> 4;
        n |= n >> 8;
        n |= n >> 16;
        n |= n >> 32;
        n++;
        
        return n;
    }
    
    void resize() {
        // For now, we'll skip resize implementation to keep it simple
        // In a production system, this would create a new larger table
        // and migrate all entries atomically
    }
};

} // namespace ultra::cache