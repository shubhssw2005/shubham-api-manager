#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <atomic>
#include <random>
#include <iomanip>
#include "lockfree/lockfree.hpp"

using namespace ultra_cpp::lockfree;

class BenchmarkTimer {
public:
    BenchmarkTimer() : start_(std::chrono::high_resolution_clock::now()) {}
    
    double elapsed_seconds() const {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_);
        return duration.count() / 1e9;
    }
    
    double elapsed_microseconds() const {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
        return duration.count();
    }
    
private:
    std::chrono::high_resolution_clock::time_point start_;
};

void benchmark_hash_table() {
    std::cout << "\n=== Hash Table Benchmarks ===" << std::endl;
    
    constexpr size_t CAPACITY = 1024 * 1024;
    HashTable<uint64_t, uint64_t, CAPACITY> hash_table;
    
    // Single-threaded insertion benchmark
    {
        const int num_ops = 1000000;
        BenchmarkTimer timer;
        
        for (int i = 0; i < num_ops; ++i) {
            hash_table.put(i, i * 2);
        }
        
        double elapsed = timer.elapsed_seconds();
        double ops_per_sec = num_ops / elapsed;
        
        std::cout << "Single-threaded insertions: " 
                  << std::fixed << std::setprecision(0) << ops_per_sec 
                  << " ops/sec" << std::endl;
    }
    
    // Single-threaded lookup benchmark
    {
        const int num_ops = 1000000;
        BenchmarkTimer timer;
        
        for (int i = 0; i < num_ops; ++i) {
            auto result = hash_table.get(i);
            if (!result.has_value() || result.value() != i * 2) {
                std::cerr << "Lookup verification failed!" << std::endl;
            }
        }
        
        double elapsed = timer.elapsed_seconds();
        double ops_per_sec = num_ops / elapsed;
        
        std::cout << "Single-threaded lookups: " 
                  << std::fixed << std::setprecision(0) << ops_per_sec 
                  << " ops/sec" << std::endl;
    }
    
    // Multi-threaded mixed operations
    {
        const int num_threads = std::thread::hardware_concurrency();
        const int ops_per_thread = 100000;
        std::vector<std::thread> threads;
        std::atomic<int> total_ops{0};
        
        BenchmarkTimer timer;
        
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back([&, t]() {
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_int_distribution<> op_dist(0, 2);
                std::uniform_int_distribution<uint64_t> key_dist(0, 999999);
                
                for (int i = 0; i < ops_per_thread; ++i) {
                    uint64_t key = key_dist(gen);
                    int op = op_dist(gen);
                    
                    switch (op) {
                        case 0: // put
                            hash_table.put(key, key * 3);
                            break;
                        case 1: // get
                            hash_table.get(key);
                            break;
                        case 2: // remove
                            hash_table.remove(key);
                            break;
                    }
                    
                    total_ops.fetch_add(1);
                }
            });
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
        
        double elapsed = timer.elapsed_seconds();
        double ops_per_sec = total_ops.load() / elapsed;
        
        std::cout << "Multi-threaded mixed (" << num_threads << " threads): " 
                  << std::fixed << std::setprecision(0) << ops_per_sec 
                  << " ops/sec" << std::endl;
    }
    
    // Print statistics
    const auto& stats = hash_table.get_stats();
    std::cout << "Final stats - Size: " << stats.size.load() 
              << ", Collisions: " << stats.collisions.load()
              << ", Max probe distance: " << stats.max_probe_distance.load() << std::endl;
}

void benchmark_ring_buffer() {
    std::cout << "\n=== Ring Buffer Benchmarks ===" << std::endl;
    
    MPMCRingBuffer<uint64_t, 1024> ring_buffer;
    
    // Single producer, single consumer
    {
        const int num_items = 1000000;
        std::atomic<bool> producer_done{false};
        std::atomic<int> items_consumed{0};
        
        BenchmarkTimer timer;
        
        std::thread producer([&]() {
            for (int i = 0; i < num_items; ++i) {
                while (!ring_buffer.try_enqueue(i)) {
                    utils::cpu_pause();
                }
            }
            producer_done.store(true);
        });
        
        std::thread consumer([&]() {
            uint64_t value;
            while (!producer_done.load() || !ring_buffer.empty()) {
                if (ring_buffer.try_dequeue(value)) {
                    items_consumed.fetch_add(1);
                } else {
                    utils::cpu_pause();
                }
            }
        });
        
        producer.join();
        consumer.join();
        
        double elapsed = timer.elapsed_seconds();
        double ops_per_sec = (num_items * 2) / elapsed; // enqueue + dequeue
        
        std::cout << "Single producer/consumer: " 
                  << std::fixed << std::setprecision(0) << ops_per_sec 
                  << " ops/sec" << std::endl;
        
        if (items_consumed.load() != num_items) {
            std::cerr << "Items consumed mismatch: " << items_consumed.load() 
                      << " vs " << num_items << std::endl;
        }
    }
    
    // Multiple producers, multiple consumers
    {
        const int num_producers = 4;
        const int num_consumers = 4;
        const int items_per_producer = 100000;
        const int total_items = num_producers * items_per_producer;
        
        std::vector<std::thread> producers;
        std::vector<std::thread> consumers;
        std::atomic<bool> all_produced{false};
        std::atomic<int> items_produced{0};
        std::atomic<int> items_consumed{0};
        
        BenchmarkTimer timer;
        
        // Start producers
        for (int p = 0; p < num_producers; ++p) {
            producers.emplace_back([&, p]() {
                for (int i = 0; i < items_per_producer; ++i) {
                    uint64_t value = p * items_per_producer + i;
                    while (!ring_buffer.try_enqueue(value)) {
                        utils::cpu_pause();
                    }
                    items_produced.fetch_add(1);
                }
            });
        }
        
        // Start consumers
        for (int c = 0; c < num_consumers; ++c) {
            consumers.emplace_back([&]() {
                uint64_t value;
                while (!all_produced.load() || !ring_buffer.empty()) {
                    if (ring_buffer.try_dequeue(value)) {
                        items_consumed.fetch_add(1);
                    } else {
                        utils::cpu_pause();
                    }
                }
            });
        }
        
        // Wait for producers
        for (auto& producer : producers) {
            producer.join();
        }
        all_produced.store(true);
        
        // Wait for consumers
        for (auto& consumer : consumers) {
            consumer.join();
        }
        
        double elapsed = timer.elapsed_seconds();
        double ops_per_sec = (total_items * 2) / elapsed; // enqueue + dequeue
        
        std::cout << "Multi producer/consumer (" << num_producers << "P/" << num_consumers << "C): " 
                  << std::fixed << std::setprecision(0) << ops_per_sec 
                  << " ops/sec" << std::endl;
        
        if (items_consumed.load() != total_items) {
            std::cerr << "Items consumed mismatch: " << items_consumed.load() 
                      << " vs " << total_items << std::endl;
        }
    }
    
    // Print statistics
    const auto& stats = ring_buffer.get_stats();
    std::cout << "Final stats - Enqueue success rate: " 
              << (100.0 * stats.enqueue_successes.load() / stats.enqueue_attempts.load()) << "%"
              << ", Dequeue success rate: "
              << (100.0 * stats.dequeue_successes.load() / stats.dequeue_attempts.load()) << "%"
              << ", Contention: " << stats.contention_count.load() << std::endl;
}

void benchmark_lru_cache() {
    std::cout << "\n=== LRU Cache Benchmarks ===" << std::endl;
    
    LRUCache<uint64_t, uint64_t, 1024> cache;
    
    // Pre-populate cache
    for (int i = 0; i < 1024; ++i) {
        cache.put(i, i * 2);
    }
    
    // Cache hit benchmark
    {
        const int num_ops = 1000000;
        BenchmarkTimer timer;
        
        for (int i = 0; i < num_ops; ++i) {
            uint64_t key = i % 1024; // Ensure cache hits
            auto result = cache.get(key);
            if (!result.has_value()) {
                std::cerr << "Cache hit failed!" << std::endl;
            }
        }
        
        double elapsed = timer.elapsed_seconds();
        double ops_per_sec = num_ops / elapsed;
        
        std::cout << "Cache hits: " 
                  << std::fixed << std::setprecision(0) << ops_per_sec 
                  << " ops/sec" << std::endl;
    }
    
    // Mixed operations benchmark
    {
        const int num_ops = 500000;
        BenchmarkTimer timer;
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> op_dist(0, 2);
        std::uniform_int_distribution<uint64_t> key_dist(0, 2047); // Some misses
        
        for (int i = 0; i < num_ops; ++i) {
            uint64_t key = key_dist(gen);
            int op = op_dist(gen);
            
            switch (op) {
                case 0: // put
                    cache.put(key, key * 3);
                    break;
                case 1: // get
                    cache.get(key);
                    break;
                case 2: // remove
                    cache.remove(key);
                    break;
            }
        }
        
        double elapsed = timer.elapsed_seconds();
        double ops_per_sec = num_ops / elapsed;
        
        std::cout << "Mixed operations: " 
                  << std::fixed << std::setprecision(0) << ops_per_sec 
                  << " ops/sec" << std::endl;
    }
    
    // Multi-threaded benchmark
    {
        const int num_threads = std::thread::hardware_concurrency();
        const int ops_per_thread = 50000;
        std::vector<std::thread> threads;
        std::atomic<int> total_ops{0};
        
        BenchmarkTimer timer;
        
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back([&, t]() {
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_int_distribution<> op_dist(0, 1); // Mostly gets and puts
                std::uniform_int_distribution<uint64_t> key_dist(0, 1535); // Mix of hits and misses
                
                for (int i = 0; i < ops_per_thread; ++i) {
                    uint64_t key = key_dist(gen);
                    int op = op_dist(gen);
                    
                    if (op == 0) {
                        cache.put(key, key * 4);
                    } else {
                        cache.get(key);
                    }
                    
                    total_ops.fetch_add(1);
                }
            });
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
        
        double elapsed = timer.elapsed_seconds();
        double ops_per_sec = total_ops.load() / elapsed;
        
        std::cout << "Multi-threaded (" << num_threads << " threads): " 
                  << std::fixed << std::setprecision(0) << ops_per_sec 
                  << " ops/sec" << std::endl;
    }
    
    // Print statistics
    const auto& stats = cache.get_stats();
    std::cout << "Final stats - Hit rate: " 
              << std::fixed << std::setprecision(1) << (cache.hit_rate() * 100) << "%"
              << ", Size: " << cache.size()
              << ", Evictions: " << stats.evictions.load() << std::endl;
}

void benchmark_atomic_ref_count() {
    std::cout << "\n=== Atomic Reference Count Benchmarks ===" << std::endl;
    
    // Single-threaded operations
    {
        const int num_ops = 1000000;
        BenchmarkTimer timer;
        
        for (int i = 0; i < num_ops; ++i) {
            AtomicRefCount<int> ref1(new int(i));
            AtomicRefCount<int> ref2(ref1);
            AtomicRefCount<int> ref3 = ref2;
            
            int value = *ref1;
            (void)value; // Suppress unused variable warning
            
            // Destructors will handle cleanup
        }
        
        double elapsed = timer.elapsed_seconds();
        double ops_per_sec = num_ops / elapsed;
        
        std::cout << "Single-threaded ref operations: " 
                  << std::fixed << std::setprecision(0) << ops_per_sec 
                  << " ops/sec" << std::endl;
    }
    
    // Multi-threaded shared reference
    {
        const int num_threads = std::thread::hardware_concurrency();
        const int ops_per_thread = 100000;
        std::vector<std::thread> threads;
        std::atomic<int> total_ops{0};
        
        AtomicRefCount<std::vector<int>> shared_ref(new std::vector<int>(1000, 42));
        
        BenchmarkTimer timer;
        
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back([&]() {
                for (int i = 0; i < ops_per_thread; ++i) {
                    AtomicRefCount<std::vector<int>> local_ref(shared_ref);
                    
                    // Access the shared data
                    int sum = 0;
                    for (int val : *local_ref) {
                        sum += val;
                    }
                    
                    if (sum != 42000) {
                        std::cerr << "Shared data corruption detected!" << std::endl;
                    }
                    
                    total_ops.fetch_add(1);
                }
            });
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
        
        double elapsed = timer.elapsed_seconds();
        double ops_per_sec = total_ops.load() / elapsed;
        
        std::cout << "Multi-threaded shared ref (" << num_threads << " threads): " 
                  << std::fixed << std::setprecision(0) << ops_per_sec 
                  << " ops/sec" << std::endl;
    }
    
    // Hazard pointer system benchmark
    {
        auto& hp_system = HazardPointerSystem::instance();
        const int num_ops = 100000;
        
        BenchmarkTimer timer;
        
        for (int i = 0; i < num_ops; ++i) {
            auto* hp = hp_system.acquire_hazard_pointer();
            if (hp != nullptr) {
                int* test_ptr = new int(i);
                hp->ptr.store(test_ptr, std::memory_order_release);
                
                // Simulate some work
                volatile int dummy = *test_ptr;
                (void)dummy;
                
                hp_system.release_hazard_pointer(hp);
                hp_system.retire_object(test_ptr, [](void* ptr) {
                    delete static_cast<int*>(ptr);
                });
            }
        }
        
        // Final cleanup
        hp_system.scan_and_reclaim();
        
        double elapsed = timer.elapsed_seconds();
        double ops_per_sec = num_ops / elapsed;
        
        std::cout << "Hazard pointer operations: " 
                  << std::fixed << std::setprecision(0) << ops_per_sec 
                  << " ops/sec" << std::endl;
        
        const auto& stats = hp_system.get_stats();
        std::cout << "HP stats - Acquired: " << stats.hazard_pointers_acquired.load()
                  << ", Released: " << stats.hazard_pointers_released.load()
                  << ", Retired: " << stats.objects_retired.load()
                  << ", Reclaimed: " << stats.objects_reclaimed.load() << std::endl;
    }
}

int main() {
    std::cout << "Ultra Low-Latency Lock-Free Data Structures Benchmarks" << std::endl;
    std::cout << "=======================================================" << std::endl;
    std::cout << "Hardware concurrency: " << std::thread::hardware_concurrency() << " threads" << std::endl;
    
    benchmark_hash_table();
    benchmark_ring_buffer();
    benchmark_lru_cache();
    benchmark_atomic_ref_count();
    
    std::cout << "\nBenchmarks completed!" << std::endl;
    return 0;
}