#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <thread>
#include <atomic>
#include <memory>

// Simple performance test program
class SimplePerformanceTest {
private:
    std::atomic<long long> counter{0};
    std::vector<std::thread> workers;
    
public:
    void runTest(int numThreads = 4, int iterations = 1000000) {
        std::cout << "Starting performance test with " << numThreads << " threads..." << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Launch worker threads
        for (int i = 0; i < numThreads; ++i) {
            workers.emplace_back([this, iterations]() {
                for (int j = 0; j < iterations; ++j) {
                    counter.fetch_add(1, std::memory_order_relaxed);
                }
            });
        }
        
        // Wait for all threads to complete
        for (auto& worker : workers) {
            worker.join();
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "Test completed!" << std::endl;
        std::cout << "Total operations: " << counter.load() << std::endl;
        std::cout << "Duration: " << duration.count() << "ms" << std::endl;
        std::cout << "Operations per second: " << (counter.load() * 1000) / duration.count() << std::endl;
    }
    
    void memoryTest() {
        std::cout << "Running memory allocation test..." << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        std::vector<std::unique_ptr<std::vector<int>>> allocations;
        
        for (int i = 0; i < 10000; ++i) {
            auto vec = std::make_unique<std::vector<int>>(1000, i);
            allocations.push_back(std::move(vec));
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "Memory test completed!" << std::endl;
        std::cout << "Allocated " << allocations.size() << " vectors" << std::endl;
        std::cout << "Duration: " << duration.count() << "ms" << std::endl;
    }
};

int main() {
    std::cout << "🚀 C++ System Performance Test" << std::endl;
    std::cout << "==============================" << std::endl;
    
    try {
        SimplePerformanceTest test;
        
        // Run atomic operations test
        test.runTest(4, 100000);
        
        std::cout << std::endl;
        
        // Run memory test
        test.memoryTest();
        
        std::cout << std::endl;
        std::cout << "✅ All tests completed successfully!" << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed: " << e.what() << std::endl;
        return 1;
    }
}