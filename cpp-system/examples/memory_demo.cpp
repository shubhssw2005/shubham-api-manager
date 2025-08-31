#include "memory/memory.hpp"
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>

using namespace ultra::memory;

void demonstrate_basic_allocation() {
    std::cout << "\n=== Basic Allocation Demo ===\n";
    
    MemoryManager& manager = MemoryManager::instance();
    
    // Allocate different sizes to trigger different strategies
    void* small_ptr = manager.allocate(128);      // Lock-free allocator
    void* medium_ptr = manager.allocate(8192);    // NUMA allocator
    void* large_ptr = manager.allocate(2097152);  // Memory mapping
    
    std::cout << "Small allocation (128 bytes): " << small_ptr << "\n";
    std::cout << "Medium allocation (8KB): " << medium_ptr << "\n";
    std::cout << "Large allocation (2MB): " << large_ptr << "\n";
    
    // Write to memory to verify it works
    if (small_ptr) {
        memset(small_ptr, 0xAA, 128);
        std::cout << "Small allocation verified\n";
    }
    
    if (medium_ptr) {
        memset(medium_ptr, 0xBB, 8192);
        std::cout << "Medium allocation verified\n";
    }
    
    if (large_ptr) {
        memset(large_ptr, 0xCC, 2097152);
        std::cout << "Large allocation verified\n";
    }
    
    // Cleanup
    manager.deallocate(small_ptr, 128);
    manager.deallocate(medium_ptr, 8192);
    manager.deallocate(large_ptr, 2097152);
    
    std::cout << "All allocations cleaned up\n";
}

void demonstrate_rcu_pointers() {
    std::cout << "\n=== RCU Smart Pointers Demo ===\n";
    
    struct SharedData {
        int value;
        std::string name;
        
        SharedData(int v, const std::string& n) : value(v), name(n) {}
    };
    
    MemoryManager& manager = MemoryManager::instance();
    
    // Create RCU shared pointer
    auto rcu_data = manager.make_rcu_shared<SharedData>(42, "test_data");
    
    std::cout << "Created RCU shared pointer\n";
    
    // Read from multiple threads
    std::vector<std::thread> readers;
    std::atomic<bool> stop_reading{false};
    
    for (int i = 0; i < 3; ++i) {
        readers.emplace_back([&, i]() {
            int read_count = 0;
            while (!stop_reading.load() && read_count < 1000) {
                {
                    ULTRA_RCU_READ_LOCK();
                    if (rcu_data) {
                        volatile int val = rcu_data->value;
                        volatile auto& name = rcu_data->name;
                        (void)val; (void)name; // Suppress unused warnings
                        ++read_count;
                    }
                }
                std::this_thread::sleep_for(std::chrono::microseconds(1));
            }
            std::cout << "Reader " << i << " completed " << read_count << " reads\n";
        });
    }
    
    // Let readers run for a bit
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Update the shared data (this will use RCU for safe updates)
    rcu_data.reset(new SharedData(123, "updated_data"));
    std::cout << "Updated shared data\n";
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    stop_reading.store(true);
    for (auto& reader : readers) {
        reader.join();
    }
    
    std::cout << "RCU demo completed\n";
}

void demonstrate_memory_scope() {
    std::cout << "\n=== Memory Scope Demo ===\n";
    
    MemoryManager& manager = MemoryManager::instance();
    
    {
        MemoryScope scope(manager);
        
        // Allocate various objects within the scope
        int* int_array = scope.allocate<int>(100);
        std::cout << "Allocated int array of 100 elements\n";
        
        // Initialize the array
        for (int i = 0; i < 100; ++i) {
            int_array[i] = i * i;
        }
        
        // Create custom objects
        struct TestObject {
            double value;
            std::vector<int> data;
            
            TestObject(double v) : value(v), data(10, static_cast<int>(v)) {}
        };
        
        TestObject* obj = scope.create<TestObject>(3.14159);
        std::cout << "Created TestObject with value: " << obj->value << "\n";
        
        // All memory will be automatically cleaned up when scope ends
        std::cout << "Scope will clean up automatically...\n";
    }
    
    std::cout << "Memory scope demo completed\n";
}

void demonstrate_ultra_containers() {
    std::cout << "\n=== Ultra Containers Demo ===\n";
    
    // Use ultra-optimized vector
    ultra_vector<int> vec = make_ultra_vector<int>();
    
    for (int i = 0; i < 1000; ++i) {
        vec.push_back(i);
    }
    
    std::cout << "Ultra vector size: " << vec.size() << "\n";
    
    // Use ultra-optimized map
    auto map = make_ultra_map<std::string, int>();
    
    map["first"] = 1;
    map["second"] = 2;
    map["third"] = 3;
    
    std::cout << "Ultra map size: " << map.size() << "\n";
    for (const auto& [key, value] : map) {
        std::cout << "  " << key << " -> " << value << "\n";
    }
    
    std::cout << "Ultra containers demo completed\n";
}

void demonstrate_memory_mapping() {
    std::cout << "\n=== Memory Mapping Demo ===\n";
    
    MemoryManager& manager = MemoryManager::instance();
    
    try {
        // Create a temporary file for demonstration
        std::string temp_file = "/tmp/ultra_memory_demo.dat";
        
        // Create and map a file
        auto mapped_file = manager.create_file(temp_file, 1024 * 1024); // 1MB
        
        if (mapped_file.is_valid()) {
            std::cout << "Created and mapped file: " << temp_file << "\n";
            std::cout << "Mapped size: " << mapped_file.size() << " bytes\n";
            
            // Write some data
            char* data = static_cast<char*>(mapped_file.data());
            strcpy(data, "Hello, Memory Mapped World!");
            
            // Sync to disk
            mapped_file.sync(false);
            std::cout << "Data written and synced to disk\n";
            
            // Advise kernel about access pattern
            mapped_file.advise_sequential();
            std::cout << "Advised kernel about sequential access\n";
            
            // The file will be automatically unmapped when mapped_file goes out of scope
        }
        
        // Clean up temp file
        unlink(temp_file.c_str());
        
    } catch (const std::exception& e) {
        std::cout << "Memory mapping demo failed: " << e.what() << "\n";
    }
    
    std::cout << "Memory mapping demo completed\n";
}

void performance_benchmark() {
    std::cout << "\n=== Performance Benchmark ===\n";
    
    MemoryManager& manager = MemoryManager::instance();
    
    const int num_iterations = 100000;
    const size_t alloc_size = 256;
    
    // Benchmark allocation/deallocation performance
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_iterations; ++i) {
        void* ptr = manager.allocate(alloc_size);
        if (ptr) {
            // Touch the memory
            memset(ptr, i & 0xFF, alloc_size);
            manager.deallocate(ptr, alloc_size);
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    
    double ns_per_op = static_cast<double>(duration.count()) / (num_iterations * 2);
    double ops_per_sec = 1e9 / ns_per_op;
    
    std::cout << "Performance results:\n";
    std::cout << "  " << ns_per_op << " ns per operation\n";
    std::cout << "  " << ops_per_sec << " operations per second\n";
    std::cout << "  " << (num_iterations * alloc_size * 2) / (1024 * 1024) << " MB processed\n";
}

int main() {
    std::cout << "Ultra Low-Latency Memory Management Demo\n";
    std::cout << "========================================\n";
    
    try {
        demonstrate_basic_allocation();
        demonstrate_rcu_pointers();
        demonstrate_memory_scope();
        demonstrate_ultra_containers();
        demonstrate_memory_mapping();
        performance_benchmark();
        
        // Print final statistics
        MemoryManager::instance().print_memory_info();
        
    } catch (const std::exception& e) {
        std::cerr << "Demo failed with exception: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "\nDemo completed successfully!\n";
    return 0;
}