#include "lockfree/lockfree.hpp"
#include <iostream>

int main() {
    using namespace ultra_cpp::lockfree;
    
    // Test hash table compilation
    HashTable<int, std::string, 64> hash_table;
    hash_table.put(1, "test");
    auto result = hash_table.get(1);
    
    // Test ring buffer compilation
    MPMCRingBuffer<int, 32> ring_buffer;
    ring_buffer.try_enqueue(42);
    int value;
    ring_buffer.try_dequeue(value);
    
    // Test LRU cache compilation
    LRUCache<int, std::string, 16> cache;
    cache.put(1, "cached");
    auto cached = cache.get(1);
    
    // Test atomic ref count compilation
    AtomicRefCount<int> ref(new int(123));
    int val = *ref;
    
    std::cout << "All lock-free data structures compiled successfully!" << std::endl;
    return 0;
}