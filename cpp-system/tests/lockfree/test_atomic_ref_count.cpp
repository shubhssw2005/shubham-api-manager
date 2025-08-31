#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <atomic>
#include <memory>
#include "lockfree/atomic_ref_count.hpp"

using namespace ultra_cpp::lockfree;

class AtomicRefCountTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test objects
        test_int = new int(42);
        test_string = new std::string("test_string");
    }
    
    void TearDown() override {
        // Cleanup is handled by AtomicRefCount destructors
    }
    
    int* test_int;
    std::string* test_string;
};

TEST_F(AtomicRefCountTest, BasicOperations) {
    // Test construction
    AtomicRefCount<int> ref1(test_int);
    EXPECT_FALSE(ref1.empty());
    EXPECT_TRUE(ref1);
    EXPECT_EQ(ref1.get(), test_int);
    EXPECT_EQ(*ref1, 42);
    EXPECT_EQ(ref1->operator int(), 42);
    EXPECT_EQ(ref1.use_count(), 1);
    EXPECT_TRUE(ref1.unique());
    
    // Test copy construction
    AtomicRefCount<int> ref2(ref1);
    EXPECT_EQ(ref1.use_count(), 2);
    EXPECT_EQ(ref2.use_count(), 2);
    EXPECT_FALSE(ref1.unique());
    EXPECT_FALSE(ref2.unique());
    EXPECT_EQ(ref1.get(), ref2.get());
    
    // Test assignment
    AtomicRefCount<int> ref3;
    EXPECT_TRUE(ref3.empty());
    EXPECT_EQ(ref3.use_count(), 0);
    
    ref3 = ref1;
    EXPECT_EQ(ref1.use_count(), 3);
    EXPECT_EQ(ref2.use_count(), 3);
    EXPECT_EQ(ref3.use_count(), 3);
}

TEST_F(AtomicRefCountTest, MoveSemantics) {
    AtomicRefCount<std::string> ref1(test_string);
    EXPECT_EQ(ref1.use_count(), 1);
    
    // Test move construction
    AtomicRefCount<std::string> ref2(std::move(ref1));
    EXPECT_TRUE(ref1.empty());
    EXPECT_EQ(ref1.use_count(), 0);
    EXPECT_EQ(ref2.use_count(), 1);
    EXPECT_EQ(*ref2, "test_string");
    
    // Test move assignment
    AtomicRefCount<std::string> ref3;
    ref3 = std::move(ref2);
    EXPECT_TRUE(ref2.empty());
    EXPECT_EQ(ref2.use_count(), 0);
    EXPECT_EQ(ref3.use_count(), 1);
    EXPECT_EQ(*ref3, "test_string");
}

TEST_F(AtomicRefCountTest, ResetAndRelease) {
    AtomicRefCount<int> ref1(test_int);
    AtomicRefCount<int> ref2(ref1);
    
    EXPECT_EQ(ref1.use_count(), 2);
    
    // Test reset
    ref1.reset();
    EXPECT_TRUE(ref1.empty());
    EXPECT_EQ(ref1.use_count(), 0);
    EXPECT_EQ(ref2.use_count(), 1);
    EXPECT_TRUE(ref2.unique());
    
    // Test reset with new pointer
    int* new_int = new int(99);
    ref1.reset(new_int);
    EXPECT_FALSE(ref1.empty());
    EXPECT_EQ(ref1.use_count(), 1);
    EXPECT_EQ(*ref1, 99);
    
    // Test release
    int* released_ptr = ref1.release();
    EXPECT_TRUE(ref1.empty());
    EXPECT_EQ(ref1.use_count(), 0);
    EXPECT_EQ(released_ptr, new_int);
    EXPECT_EQ(*released_ptr, 99);
    
    // Manual cleanup since we released the pointer
    delete released_ptr;
}

TEST_F(AtomicRefCountTest, ComparisonOperators) {
    AtomicRefCount<int> ref1(test_int);
    AtomicRefCount<int> ref2(ref1);
    AtomicRefCount<int> ref3(new int(100));
    AtomicRefCount<int> ref4;
    
    // Test equality
    EXPECT_TRUE(ref1 == ref2);
    EXPECT_FALSE(ref1 == ref3);
    EXPECT_FALSE(ref1 == ref4);
    
    // Test inequality
    EXPECT_FALSE(ref1 != ref2);
    EXPECT_TRUE(ref1 != ref3);
    EXPECT_TRUE(ref1 != ref4);
    
    // Test less than (pointer comparison)
    bool less_result = ref1 < ref3;
    EXPECT_TRUE(less_result == (ref1.get() < ref3.get()));
}

TEST_F(AtomicRefCountTest, ConcurrentOperations) {
    const int num_threads = 8;
    const int operations_per_thread = 1000;
    
    AtomicRefCount<int> original_ref(test_int);
    std::vector<std::thread> threads;
    std::atomic<int> max_ref_count{0};
    std::atomic<int> total_operations{0};
    
    // Concurrent copy and access operations
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < operations_per_thread; ++i) {
                // Create local copy
                AtomicRefCount<int> local_ref(original_ref);
                
                // Access the value
                int value = *local_ref;
                EXPECT_EQ(value, 42);
                
                // Update max ref count
                int current_count = local_ref.use_count();
                int current_max = max_ref_count.load();
                while (current_count > current_max && 
                       !max_ref_count.compare_exchange_weak(current_max, current_count)) {
                    // Retry
                }
                
                total_operations.fetch_add(1);
                
                // local_ref destructor will decrement ref count
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    EXPECT_EQ(total_operations.load(), num_threads * operations_per_thread);
    EXPECT_GT(max_ref_count.load(), 1); // Should have seen concurrent references
    EXPECT_EQ(original_ref.use_count(), 1); // Back to original count
}

TEST_F(AtomicRefCountTest, ConcurrentCopyAssignment) {
    const int num_threads = 4;
    const int operations_per_thread = 500;
    
    std::vector<AtomicRefCount<int>> refs(num_threads);
    std::vector<std::thread> threads;
    std::atomic<int> successful_assignments{0};
    
    // Initialize first ref
    refs[0].reset(test_int);
    
    // Concurrent assignment operations
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < operations_per_thread; ++i) {
                // Assign from a random other ref
                int source_index = (t + i) % num_threads;
                refs[t] = refs[source_index];
                
                if (!refs[t].empty()) {
                    successful_assignments.fetch_add(1);
                    
                    // Verify the value
                    int value = *refs[t];
                    EXPECT_EQ(value, 42);
                }
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    EXPECT_GT(successful_assignments.load(), 0);
    
    // At least one ref should still point to the original object
    int total_refs = 0;
    for (const auto& ref : refs) {
        if (!ref.empty() && ref.get() == test_int) {
            total_refs += ref.use_count();
            break;
        }
    }
    EXPECT_GT(total_refs, 0);
}

TEST_F(AtomicRefCountTest, HazardPointerSystem) {
    auto& hp_system = HazardPointerSystem::instance();
    
    // Test hazard pointer acquisition and release
    auto* hp1 = hp_system.acquire_hazard_pointer();
    ASSERT_NE(hp1, nullptr);
    
    auto* hp2 = hp_system.acquire_hazard_pointer();
    ASSERT_NE(hp2, nullptr);
    EXPECT_NE(hp1, hp2);
    
    // Test pointer protection
    int* test_ptr = new int(123);
    hp1->ptr.store(test_ptr, std::memory_order_release);
    
    // Retire the object
    hp_system.retire_object(test_ptr, [](void* ptr) {
        delete static_cast<int*>(ptr);
    });
    
    // Object should not be reclaimed yet (it's protected)
    hp_system.scan_and_reclaim();
    
    // Release hazard pointer
    hp_system.release_hazard_pointer(hp1);
    hp_system.release_hazard_pointer(hp2);
    
    // Now object should be reclaimable
    hp_system.scan_and_reclaim();
    
    const auto& stats = hp_system.get_stats();
    EXPECT_GT(stats.hazard_pointers_acquired.load(), 0);
    EXPECT_GT(stats.hazard_pointers_released.load(), 0);
    EXPECT_GT(stats.objects_retired.load(), 0);
}

TEST_F(AtomicRefCountTest, HazardPointerGuard) {
    int* test_ptr = new int(456);
    
    {
        HazardPointerGuard guard(test_ptr);
        
        // Pointer should be protected
        int* protected_ptr = guard.get<int>();
        EXPECT_EQ(protected_ptr, test_ptr);
        EXPECT_EQ(*protected_ptr, 456);
        
        // Test move semantics
        HazardPointerGuard moved_guard(std::move(guard));
        EXPECT_EQ(moved_guard.get<int>(), test_ptr);
        EXPECT_EQ(guard.get<int>(), nullptr);
        
        // Test reset
        moved_guard.reset();
        EXPECT_EQ(moved_guard.get<int>(), nullptr);
        
        // Test protect
        moved_guard.protect(test_ptr);
        EXPECT_EQ(moved_guard.get<int>(), test_ptr);
        
        // Guard destructor will release the hazard pointer
    }
    
    // Clean up
    delete test_ptr;
}