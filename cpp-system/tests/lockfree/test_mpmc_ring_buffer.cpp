#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include "lockfree/mpmc_ring_buffer.hpp"

using namespace ultra_cpp::lockfree;

class MPMCRingBufferTest : public ::testing::Test {
protected:
    void SetUp() override {
        ring_buffer = std::make_unique<MPMCRingBuffer<int, 64>>();
    }
    
    void TearDown() override {
        ring_buffer.reset();
    }
    
    std::unique_ptr<MPMCRingBuffer<int, 64>> ring_buffer;
};

TEST_F(MPMCRingBufferTest, BasicOperations) {
    // Test empty buffer
    EXPECT_TRUE(ring_buffer->empty());
    EXPECT_FALSE(ring_buffer->full());
    EXPECT_EQ(ring_buffer->size(), 0);
    
    // Test enqueue
    EXPECT_TRUE(ring_buffer->try_enqueue(1));
    EXPECT_TRUE(ring_buffer->try_enqueue(2));
    EXPECT_TRUE(ring_buffer->try_enqueue(3));
    
    EXPECT_FALSE(ring_buffer->empty());
    EXPECT_EQ(ring_buffer->size(), 3);
    
    // Test dequeue
    int value;
    EXPECT_TRUE(ring_buffer->try_dequeue(value));
    EXPECT_EQ(value, 1);
    
    EXPECT_TRUE(ring_buffer->try_dequeue(value));
    EXPECT_EQ(value, 2);
    
    EXPECT_TRUE(ring_buffer->try_dequeue(value));
    EXPECT_EQ(value, 3);
    
    EXPECT_TRUE(ring_buffer->empty());
    EXPECT_EQ(ring_buffer->size(), 0);
}

TEST_F(MPMCRingBufferTest, MoveSemantics) {
    // Test move enqueue
    std::string movable_string = "test_string";
    auto string_buffer = std::make_unique<MPMCRingBuffer<std::string, 64>>();
    
    EXPECT_TRUE(string_buffer->try_enqueue(std::move(movable_string)));
    
    std::string result;
    EXPECT_TRUE(string_buffer->try_dequeue(result));
    EXPECT_EQ(result, "test_string");
}

TEST_F(MPMCRingBufferTest, EmplaceOperation) {
    struct TestStruct {
        int a;
        double b;
        std::string c;
        
        TestStruct(int a, double b, const std::string& c) : a(a), b(b), c(c) {}
    };
    
    auto struct_buffer = std::make_unique<MPMCRingBuffer<TestStruct, 64>>();
    
    EXPECT_TRUE(struct_buffer->try_emplace(42, 3.14, "test"));
    
    TestStruct result(0, 0.0, "");
    EXPECT_TRUE(struct_buffer->try_dequeue(result));
    EXPECT_EQ(result.a, 42);
    EXPECT_DOUBLE_EQ(result.b, 3.14);
    EXPECT_EQ(result.c, "test");
}

TEST_F(MPMCRingBufferTest, FullBufferBehavior) {
    // Fill the buffer to capacity
    for (size_t i = 0; i < ring_buffer->capacity(); ++i) {
        EXPECT_TRUE(ring_buffer->try_enqueue(static_cast<int>(i)));
    }
    
    EXPECT_TRUE(ring_buffer->full());
    EXPECT_EQ(ring_buffer->size(), ring_buffer->capacity());
    
    // Try to add one more (should fail)
    EXPECT_FALSE(ring_buffer->try_enqueue(999));
    
    // Dequeue one item
    int value;
    EXPECT_TRUE(ring_buffer->try_dequeue(value));
    EXPECT_EQ(value, 0);
    
    EXPECT_FALSE(ring_buffer->full());
    
    // Now we should be able to enqueue again
    EXPECT_TRUE(ring_buffer->try_enqueue(999));
}

TEST_F(MPMCRingBufferTest, EmptyBufferBehavior) {
    // Try to dequeue from empty buffer
    int value;
    EXPECT_FALSE(ring_buffer->try_dequeue(value));
    
    // Add and remove one item
    EXPECT_TRUE(ring_buffer->try_enqueue(42));
    EXPECT_TRUE(ring_buffer->try_dequeue(value));
    EXPECT_EQ(value, 42);
    
    // Should be empty again
    EXPECT_TRUE(ring_buffer->empty());
    EXPECT_FALSE(ring_buffer->try_dequeue(value));
}

TEST_F(MPMCRingBufferTest, SingleProducerSingleConsumer) {
    const int num_items = 10000;
    std::atomic<bool> producer_done{false};
    std::atomic<int> items_consumed{0};
    
    // Producer thread
    std::thread producer([&]() {
        for (int i = 0; i < num_items; ++i) {
            while (!ring_buffer->try_enqueue(i)) {
                std::this_thread::yield();
            }
        }
        producer_done.store(true);
    });
    
    // Consumer thread
    std::thread consumer([&]() {
        int value;
        int expected = 0;
        
        while (!producer_done.load() || !ring_buffer->empty()) {
            if (ring_buffer->try_dequeue(value)) {
                EXPECT_EQ(value, expected);
                ++expected;
                items_consumed.fetch_add(1);
            } else {
                std::this_thread::yield();
            }
        }
    });
    
    producer.join();
    consumer.join();
    
    EXPECT_EQ(items_consumed.load(), num_items);
    EXPECT_TRUE(ring_buffer->empty());
}

TEST_F(MPMCRingBufferTest, MultipleProducersMultipleConsumers) {
    const int num_producers = 4;
    const int num_consumers = 4;
    const int items_per_producer = 1000;
    const int total_items = num_producers * items_per_producer;
    
    std::vector<std::thread> producers;
    std::vector<std::thread> consumers;
    std::atomic<int> items_produced{0};
    std::atomic<int> items_consumed{0};
    std::atomic<bool> all_produced{false};
    
    // Start producers
    for (int p = 0; p < num_producers; ++p) {
        producers.emplace_back([&, p]() {
            for (int i = 0; i < items_per_producer; ++i) {
                int value = p * items_per_producer + i;
                while (!ring_buffer->try_enqueue(value)) {
                    std::this_thread::yield();
                }
                items_produced.fetch_add(1);
            }
        });
    }
    
    // Start consumers
    for (int c = 0; c < num_consumers; ++c) {
        consumers.emplace_back([&]() {
            int value;
            while (!all_produced.load() || !ring_buffer->empty()) {
                if (ring_buffer->try_dequeue(value)) {
                    items_consumed.fetch_add(1);
                } else {
                    std::this_thread::yield();
                }
            }
        });
    }
    
    // Wait for all producers to finish
    for (auto& producer : producers) {
        producer.join();
    }
    all_produced.store(true);
    
    // Wait for all consumers to finish
    for (auto& consumer : consumers) {
        consumer.join();
    }
    
    EXPECT_EQ(items_produced.load(), total_items);
    EXPECT_EQ(items_consumed.load(), total_items);
    EXPECT_TRUE(ring_buffer->empty());
}

TEST_F(MPMCRingBufferTest, Statistics) {
    const auto& stats = ring_buffer->get_stats();
    
    // Initial state
    EXPECT_EQ(stats.enqueue_attempts.load(), 0);
    EXPECT_EQ(stats.enqueue_successes.load(), 0);
    EXPECT_EQ(stats.dequeue_attempts.load(), 0);
    EXPECT_EQ(stats.dequeue_successes.load(), 0);
    
    // Perform some operations
    ring_buffer->try_enqueue(1);
    ring_buffer->try_enqueue(2);
    
    int value;
    ring_buffer->try_dequeue(value);
    
    // Failed dequeue from empty buffer
    ring_buffer->try_dequeue(value);
    
    EXPECT_EQ(stats.enqueue_attempts.load(), 2);
    EXPECT_EQ(stats.enqueue_successes.load(), 2);
    EXPECT_EQ(stats.dequeue_attempts.load(), 2);
    EXPECT_EQ(stats.dequeue_successes.load(), 1);
    
    // Reset stats
    ring_buffer->reset_stats();
    EXPECT_EQ(stats.enqueue_attempts.load(), 0);
    EXPECT_EQ(stats.enqueue_successes.load(), 0);
    EXPECT_EQ(stats.dequeue_attempts.load(), 0);
    EXPECT_EQ(stats.dequeue_successes.load(), 0);
}

TEST_F(MPMCRingBufferTest, PerformanceBenchmark) {
    const int num_operations = 1000000;
    
    // Benchmark single-threaded enqueue/dequeue
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_operations; ++i) {
        while (!ring_buffer->try_enqueue(i)) {
            // Spin until successful
        }
        
        int value;
        while (!ring_buffer->try_dequeue(value)) {
            // Spin until successful
        }
        
        EXPECT_EQ(value, i);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double ops_per_second = (num_operations * 2 * 1000000.0) / duration.count(); // *2 for enqueue+dequeue
    
    // Should achieve at least 1M ops/sec (very conservative)
    EXPECT_GT(ops_per_second, 1000000.0);
    
    std::cout << "Ring buffer performance: " << ops_per_second << " ops/sec" << std::endl;
}