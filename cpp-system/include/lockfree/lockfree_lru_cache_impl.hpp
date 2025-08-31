#pragma once

#include "lockfree_lru_cache.hpp"
#include <functional>
#include <chrono>
#include <thread>

namespace ultra_cpp {
namespace lockfree {

template<typename Key, typename Value, size_t Capacity>
uint64_t LRUCache<Key, Value, Capacity>::hash(const Key& key) const noexcept {
    std::hash<Key> hasher;
    uint64_t h = hasher(key);
    return h == 0 ? 2 : h; // Avoid reserved values
}

template<typename Key, typename Value, size_t Capacity>
typename LRUCache<Key, Value, Capacity>::Node* 
LRUCache<Key, Value, Capacity>::find_node(const Key& key, uint64_t key_hash) const noexcept {
    size_t slot = key_hash & MASK;
    size_t probe_count = 0;
    
    while (probe_count < Capacity) {
        const Node& node = hash_table_[slot];
        uint64_t stored_hash = node.key_hash.load(std::memory_order_acquire);
        
        if (stored_hash == Node::EMPTY_HASH) {
            return nullptr; // Not found
        }
        
        if (stored_hash == key_hash) {
            Value* val = node.value.load(std::memory_order_acquire);
            if (val != nullptr) {
                return const_cast<Node*>(&node); // Found
            }
        }
        
        slot = (slot + 1) & MASK;
        ++probe_count;
    }
    
    return nullptr;
}

template<typename Key, typename Value, size_t Capacity>
typename LRUCache<Key, Value, Capacity>::Node* 
LRUCache<Key, Value, Capacity>::find_empty_slot(uint64_t key_hash) const noexcept {
    size_t slot = key_hash & MASK;
    size_t probe_count = 0;
    
    while (probe_count < Capacity) {
        Node& node = const_cast<Node&>(hash_table_[slot]);
        uint64_t stored_hash = node.key_hash.load(std::memory_order_acquire);
        
        if (stored_hash == Node::EMPTY_HASH || stored_hash == Node::DELETED_HASH) {
            // Try to claim this slot
            uint64_t expected = stored_hash;
            if (node.key_hash.compare_exchange_weak(
                    expected, key_hash, 
                    std::memory_order_acq_rel, std::memory_order_acquire)) {
                return &node;
            }
        }
        
        slot = (slot + 1) & MASK;
        ++probe_count;
    }
    
    return nullptr; // No empty slot found
}

template<typename Key, typename Value, size_t Capacity>
void LRUCache<Key, Value, Capacity>::move_to_head(Node* node) noexcept {
    if (node == nullptr) return;
    
    // Remove from current position
    remove_from_list(node);
    
    // Add to head
    add_to_head(node);
}

template<typename Key, typename Value, size_t Capacity>
void LRUCache<Key, Value, Capacity>::remove_from_list(Node* node) noexcept {
    if (node == nullptr) return;
    
    while (true) {
        Node* prev_node = node->prev.load(std::memory_order_acquire);
        Node* next_node = node->next.load(std::memory_order_acquire);
        
        if (prev_node == nullptr || next_node == nullptr) {
            // Node not in list or being modified by another thread
            return;
        }
        
        // Try to update prev->next
        Node* expected_next = node;
        if (prev_node->next.compare_exchange_weak(
                expected_next, next_node, 
                std::memory_order_acq_rel, std::memory_order_acquire)) {
            
            // Try to update next->prev
            Node* expected_prev = node;
            if (next_node->prev.compare_exchange_weak(
                    expected_prev, prev_node,
                    std::memory_order_acq_rel, std::memory_order_acquire)) {
                
                // Successfully removed, clear node's pointers
                node->prev.store(nullptr, std::memory_order_release);
                node->next.store(nullptr, std::memory_order_release);
                return;
            } else {
                // Restore prev->next if next->prev failed
                prev_node->next.store(node, std::memory_order_release);
            }
        }
        
        // Retry if CAS failed
        std::this_thread::yield();
    }
}

template<typename Key, typename Value, size_t Capacity>
void LRUCache<Key, Value, Capacity>::add_to_head(Node* node) noexcept {
    if (node == nullptr) return;
    
    while (true) {
        Node* head_next = head_sentinel_.next.load(std::memory_order_acquire);
        
        // Set up node's pointers
        node->prev.store(&head_sentinel_, std::memory_order_relaxed);
        node->next.store(head_next, std::memory_order_relaxed);
        
        // Try to update head->next
        if (head_sentinel_.next.compare_exchange_weak(
                head_next, node,
                std::memory_order_acq_rel, std::memory_order_acquire)) {
            
            // Try to update old_head->prev
            Node* expected_prev = &head_sentinel_;
            if (head_next->prev.compare_exchange_weak(
                    expected_prev, node,
                    std::memory_order_acq_rel, std::memory_order_acquire)) {
                return; // Successfully added
            } else {
                // Restore head->next if old_head->prev failed
                head_sentinel_.next.store(head_next, std::memory_order_release);
            }
        }
        
        // Retry if CAS failed
        std::this_thread::yield();
    }
}

template<typename Key, typename Value, size_t Capacity>
typename LRUCache<Key, Value, Capacity>::Node* 
LRUCache<Key, Value, Capacity>::remove_tail() noexcept {
    while (true) {
        Node* tail_prev = tail_sentinel_.prev.load(std::memory_order_acquire);
        
        if (tail_prev == &head_sentinel_) {
            return nullptr; // List is empty
        }
        
        // Try to remove the tail node
        Node* prev_prev = tail_prev->prev.load(std::memory_order_acquire);
        if (prev_prev == nullptr) {
            continue; // Node being modified, retry
        }
        
        // Update tail->prev
        if (tail_sentinel_.prev.compare_exchange_weak(
                tail_prev, prev_prev,
                std::memory_order_acq_rel, std::memory_order_acquire)) {
            
            // Update prev_prev->next
            Node* expected_next = tail_prev;
            if (prev_prev->next.compare_exchange_weak(
                    expected_next, &tail_sentinel_,
                    std::memory_order_acq_rel, std::memory_order_acquire)) {
                
                // Successfully removed, clear node's pointers
                tail_prev->prev.store(nullptr, std::memory_order_release);
                tail_prev->next.store(nullptr, std::memory_order_release);
                return tail_prev;
            } else {
                // Restore tail->prev if prev_prev->next failed
                tail_sentinel_.prev.store(tail_prev, std::memory_order_release);
            }
        }
        
        // Retry if CAS failed
        std::this_thread::yield();
    }
}

template<typename Key, typename Value, size_t Capacity>
void LRUCache<Key, Value, Capacity>::acquire_ref(Node* node) noexcept {
    if (node != nullptr) {
        node->ref_count.fetch_add(1, std::memory_order_acq_rel);
    }
}

template<typename Key, typename Value, size_t Capacity>
void LRUCache<Key, Value, Capacity>::release_ref(Node* node) noexcept {
    if (node != nullptr) {
        node->ref_count.fetch_sub(1, std::memory_order_acq_rel);
    }
}

template<typename Key, typename Value, size_t Capacity>
uint64_t LRUCache<Key, Value, Capacity>::get_current_time() noexcept {
    return current_time_.fetch_add(1, std::memory_order_relaxed);
}

} // namespace lockfree
} // namespace ultra_cpp