#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <random>
#include <atomic>
#include "lockfree/lockfree.hpp"

using namespace ultra_cpp::lockfree;

void demonstrate_hash_table() {
    std::cout << "\n=== Lock-Free Hash Table Demo ===" << std::endl;
    
    // Create a hash table with 1024 slots
    HashTable<std::string, int, 1024> user_scores;
    
    // Basic operations
    user_scores.put("alice", 100);
    user_scores.put("bob", 85);
    user_scores.put("charlie", 92);
    
    std::cout << "Initial scores:" << std::endl;
    for (const auto& name : {"alice", "bob", "charlie"}) {
        auto score = user_scores.get(name);
        if (score.has_value()) {
            std::cout << "  " << name << ": " << score.value() << std::endl;
        }
    }
    
    // Update a score
    user_scores.put("alice", 105);
    std::cout << "Alice's updated score: " << user_scores.get("alice").value() << std::endl;
    
    // Concurrent operations
    const int num_threads = 4;
    const int ops_per_thread = 1000;
    std::vector<std::thread> threads;
    std::atomic<int> successful_ops{0};
    
    std::cout << "Running " << num_threads << " concurrent threads..." << std::endl;
    
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> score_dist(50, 100);
            
            for (int i = 0; i < ops_per_thread; ++i) {
                std::string user = "user_" + std::to_string(t) + "_" + std::to_string(i);
                int score = score_dist(gen);
                
                if (user_scores.put(user, score)) {
                    successful_ops.fetch_add(1);
                }
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    const auto& stats = user_scores.get_stats();
    std::cout << "Completed " << successful_ops.load() << " operations" << std::endl;
    std::cout << "Hash table size: " << stats.size.load() << std::endl;
    std::cout << "Load factor: " << user_scores.load_factor() << std::endl;
}

void demonstrate_ring_buffer() {
    std::cout << "\n=== MPMC Ring Buffer Demo ===" << std::endl;
    
    // Create a ring buffer for task processing
    MPMCRingBuffer<std::string, 64> task_queue;
    
    // Producer-consumer pattern
    const int num_producers = 2;
    const int num_consumers = 2;
    const int tasks_per_producer = 100;
    
    std::vector<std::thread> producers;
    std::vector<std::thread> consumers;
    std::atomic<bool> production_done{false};
    std::atomic<int> tasks_processed{0};
    
    std::cout << "Starting " << num_producers << " producers and " 
              << num_consumers << " consumers..." << std::endl;
    
    // Start producers
    for (int p = 0; p < num_producers; ++p) {
        producers.emplace_back([&, p]() {
            for (int i = 0; i < tasks_per_producer; ++i) {
                std::string task = "task_" + std::to_string(p) + "_" + std::to_string(i);
                
                while (!task_queue.try_enqueue(task)) {
                    std::this_thread::yield();
                }
                
                // Simulate task creation time
                std::this_thread::sleep_for(std::chrono::microseconds(10));
            }
        });
    }
    
    // Start consumers
    for (int c = 0; c < num_consumers; ++c) {
        consumers.emplace_back([&, c]() {
            std::string task;
            while (!production_done.load() || !task_queue.empty()) {
                if (task_queue.try_dequeue(task)) {
                    // Simulate task processing
                    std::this_thread::sleep_for(std::chrono::microseconds(5));
                    tasks_processed.fetch_add(1);
                } else {
                    std::this_thread::yield();
                }
            }
        });
    }
    
    // Wait for producers to finish
    for (auto& producer : producers) {
        producer.join();
    }
    production_done.store(true);
    
    // Wait for consumers to finish
    for (auto& consumer : consumers) {
        consumer.join();
    }
    
    const auto& stats = task_queue.get_stats();
    std::cout << "Processed " << tasks_processed.load() << " tasks" << std::endl;
    std::cout << "Enqueue success rate: " 
              << (100.0 * stats.enqueue_successes.load() / stats.enqueue_attempts.load()) 
              << "%" << std::endl;
    std::cout << "Dequeue success rate: " 
              << (100.0 * stats.dequeue_successes.load() / stats.dequeue_attempts.load()) 
              << "%" << std::endl;
}

void demonstrate_lru_cache() {
    std::cout << "\n=== Lock-Free LRU Cache Demo ===" << std::endl;
    
    // Create an LRU cache for web page caching
    LRUCache<std::string, std::string, 16> page_cache;
    
    // Simulate web page requests
    std::vector<std::string> pages = {
        "/home", "/about", "/products", "/contact", "/blog",
        "/login", "/register", "/profile", "/settings", "/help"
    };
    
    std::cout << "Simulating web page cache..." << std::endl;
    
    // Initial page loads (cache misses)
    for (const auto& page : pages) {
        std::string content = "Content for " + page;
        page_cache.put(page, content);
        std::cout << "Cached: " << page << std::endl;
    }
    
    // Simulate access pattern with some popular pages
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> page_dist(0, pages.size() - 1);
    std::uniform_real_distribution<> popularity_dist(0.0, 1.0);
    
    int cache_hits = 0;
    int cache_misses = 0;
    const int num_requests = 1000;
    
    for (int i = 0; i < num_requests; ++i) {
        std::string page;
        
        // 70% chance of accessing popular pages (first 3)
        if (popularity_dist(gen) < 0.7) {
            page = pages[page_dist(gen) % 3];
        } else {
            page = pages[page_dist(gen)];
        }
        
        auto content = page_cache.get(page);
        if (content.has_value()) {
            cache_hits++;
        } else {
            cache_misses++;
            // Cache miss - load and cache the page
            std::string new_content = "Content for " + page;
            page_cache.put(page, new_content);
        }
    }
    
    const auto& stats = page_cache.get_stats();
    std::cout << "Cache performance:" << std::endl;
    std::cout << "  Hits: " << cache_hits << " (" 
              << (100.0 * cache_hits / num_requests) << "%)" << std::endl;
    std::cout << "  Misses: " << cache_misses << " (" 
              << (100.0 * cache_misses / num_requests) << "%)" << std::endl;
    std::cout << "  Cache size: " << page_cache.size() << std::endl;
    std::cout << "  Evictions: " << stats.evictions.load() << std::endl;
}

void demonstrate_atomic_ref_count() {
    std::cout << "\n=== Atomic Reference Count Demo ===" << std::endl;
    
    // Shared resource management
    struct SharedResource {
        std::vector<int> data;
        std::string name;
        
        SharedResource(const std::string& n, size_t size) 
            : data(size, 42), name(n) {
            std::cout << "Created resource: " << name << std::endl;
        }
        
        ~SharedResource() {
            std::cout << "Destroyed resource: " << name << std::endl;
        }
    };
    
    // Create shared resource
    AtomicRefCount<SharedResource> shared_resource(
        new SharedResource("DatabaseConnection", 1000)
    );
    
    std::cout << "Initial reference count: " << shared_resource.use_count() << std::endl;
    
    // Simulate multiple threads sharing the resource
    const int num_threads = 4;
    std::vector<std::thread> threads;
    std::atomic<int> operations_completed{0};
    
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            // Each thread gets its own reference
            AtomicRefCount<SharedResource> local_ref(shared_resource);
            
            std::cout << "Thread " << t << " acquired reference (count: " 
                      << local_ref.use_count() << ")" << std::endl;
            
            // Use the shared resource
            for (int i = 0; i < 100; ++i) {
                int sum = 0;
                for (int val : local_ref->data) {
                    sum += val;
                }
                
                if (sum != 42000) {
                    std::cerr << "Data corruption detected!" << std::endl;
                }
                
                operations_completed.fetch_add(1);
            }
            
            std::cout << "Thread " << t << " completed work" << std::endl;
            // local_ref destructor will decrement reference count
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    std::cout << "All threads completed " << operations_completed.load() 
              << " operations" << std::endl;
    std::cout << "Final reference count: " << shared_resource.use_count() << std::endl;
    
    // Demonstrate hazard pointer system
    std::cout << "\nHazard Pointer System Demo:" << std::endl;
    auto& hp_system = HazardPointerSystem::instance();
    
    {
        HazardPointerGuard guard;
        int* protected_ptr = new int(123);
        
        guard.protect(protected_ptr);
        std::cout << "Protected pointer value: " << *guard.get<int>() << std::endl;
        
        // Retire the object (it won't be deleted while protected)
        hp_system.retire_object(protected_ptr, [](void* ptr) {
            std::cout << "Deleting protected object" << std::endl;
            delete static_cast<int*>(ptr);
        });
        
        hp_system.scan_and_reclaim();
        std::cout << "Object still protected, not deleted yet" << std::endl;
        
        // Guard destructor will release protection
    }
    
    // Now the object can be safely reclaimed
    hp_system.scan_and_reclaim();
    std::cout << "Object protection released, cleanup completed" << std::endl;
    
    const auto& hp_stats = hp_system.get_stats();
    std::cout << "Hazard pointer stats - Acquired: " << hp_stats.hazard_pointers_acquired.load()
              << ", Released: " << hp_stats.hazard_pointers_released.load() << std::endl;
}

int main() {
    std::cout << "Ultra Low-Latency Lock-Free Data Structures Demo" << std::endl;
    std::cout << "===============================================" << std::endl;
    
    demonstrate_hash_table();
    demonstrate_ring_buffer();
    demonstrate_lru_cache();
    demonstrate_atomic_ref_count();
    
    std::cout << "\nDemo completed successfully!" << std::endl;
    return 0;
}