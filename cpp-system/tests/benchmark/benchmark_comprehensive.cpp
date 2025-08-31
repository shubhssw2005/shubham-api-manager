#include "../framework/test_framework.hpp"
#include "cache/ultra_cache.hpp"
#include "lockfree/atomic_ref_count.hpp"
#include "memory/numa_allocator.hpp"
#include "common/logger.hpp"
#include <benchmark/benchmark.h>
#include <random>
#include <string>
#include <vector>

using namespace ultra::testing;

// Cache benchmarks
class CacheBenchmark : public UltraBenchmarkFixture {
protected:
    void SetUp(const ::benchmark::State& state) override {
        UltraBenchmarkFixture::SetUp(state);
        cache_ = std::make_unique<ultra::cache::UltraCache<int, std::string>>(state.range(0));
        
        // Pre-populate cache for read benchmarks
        for (int i = 0; i < state.range(0) / 2; ++i) {
            cache_->put(i, "value_" + std::to_string(i));
        }
    }
    
    std::unique_ptr<ultra::cache::UltraCache<int, std::string>> cache_;
};

BENCHMARK_DEFINE_F(CacheBenchmark, CacheGet)(benchmark::State& state) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, state.range(0) / 2 - 1);
    
    for (auto _ : state) {
        int key = dis(gen);
        auto result = cache_->get(key);
        benchmark::DoNotOptimize(result);
    }
    
    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * sizeof(int));
}

BENCHMARK_DEFINE_F(CacheBenchmark, CachePut)(benchmark::State& state) {
    int key = state.range(0);
    
    for (auto _ : state) {
        cache_->put(key++, "benchmark_value_" + std::to_string(key));
    }
    
    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * (sizeof(int) + 20)); // Approximate value size
}

BENCHMARK_DEFINE_F(CacheBenchmark, CacheMixed)(benchmark::State& state) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> key_dis(0, state.range(0) - 1);
    std::uniform_real_distribution<> op_dis(0.0, 1.0);
    
    for (auto _ : state) {
        int key = key_dis(gen);
        if (op_dis(gen) < 0.8) { // 80% reads, 20% writes
            auto result = cache_->get(key);
            benchmark::DoNotOptimize(result);
        } else {
            cache_->put(key, "mixed_value_" + std::to_string(key));
        }
    }
    
    state.SetItemsProcessed(state.iterations());
}

// Register cache benchmarks with different cache sizes
BENCHMARK_REGISTER_F(CacheBenchmark, CacheGet)
    ->Range(1000, 1000000)
    ->Unit(benchmark::kNanosecond);

BENCHMARK_REGISTER_F(CacheBenchmark, CachePut)
    ->Range(1000, 1000000)
    ->Unit(benchmark::kNanosecond);

BENCHMARK_REGISTER_F(CacheBenchmark, CacheMixed)
    ->Range(1000, 1000000)
    ->Unit(benchmark::kNanosecond);

// Memory allocation benchmarks
class MemoryBenchmark : public UltraBenchmarkFixture {
protected:
    void SetUp(const ::benchmark::State& state) override {
        UltraBenchmarkFixture::SetUp(state);
        allocator_ = std::make_unique<ultra::memory::NUMAAllocator>(0);
    }
    
    std::unique_ptr<ultra::memory::NUMAAllocator> allocator_;
};

BENCHMARK_DEFINE_F(MemoryBenchmark, SmallAllocations)(benchmark::State& state) {
    size_t size = state.range(0);
    
    for (auto _ : state) {
        void* ptr = allocator_->allocate(size);
        benchmark::DoNotOptimize(ptr);
        allocator_->deallocate(ptr, size);
    }
    
    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * size);
}

BENCHMARK_DEFINE_F(MemoryBenchmark, AlignedAllocations)(benchmark::State& state) {
    size_t size = 1024;
    size_t alignment = state.range(0);
    
    for (auto _ : state) {
        void* ptr = allocator_->allocate_aligned(size, alignment);
        benchmark::DoNotOptimize(ptr);
        allocator_->deallocate(ptr, size);
    }
    
    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * size);
}

BENCHMARK_REGISTER_F(MemoryBenchmark, SmallAllocations)
    ->Range(64, 4096)
    ->Unit(benchmark::kNanosecond);

BENCHMARK_REGISTER_F(MemoryBenchmark, AlignedAllocations)
    ->Range(8, 256)
    ->Unit(benchmark::kNanosecond);

// Lock-free data structure benchmarks
class LockFreeBenchmark : public UltraBenchmarkFixture {
protected:
    void SetUp(const ::benchmark::State& state) override {
        UltraBenchmarkFixture::SetUp(state);
        ref_count_ = std::make_unique<ultra::lockfree::AtomicRefCount>(1);
    }
    
    std::unique_ptr<ultra::lockfree::AtomicRefCount> ref_count_;
};

BENCHMARK_DEFINE_F(LockFreeBenchmark, RefCountIncrement)(benchmark::State& state) {
    for (auto _ : state) {
        ref_count_->increment();
    }
    
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK_DEFINE_F(LockFreeBenchmark, RefCountDecrement)(benchmark::State& state) {
    // Pre-increment to avoid hitting zero
    for (int i = 0; i < state.iterations(); ++i) {
        ref_count_->increment();
    }
    
    for (auto _ : state) {
        ref_count_->decrement();
    }
    
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK_DEFINE_F(LockFreeBenchmark, RefCountMixed)(benchmark::State& state) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    for (auto _ : state) {
        if (dis(gen) < 0.6) { // 60% increments, 40% decrements
            ref_count_->increment();
        } else {
            ref_count_->decrement();
        }
    }
    
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK_REGISTER_F(LockFreeBenchmark, RefCountIncrement)
    ->Unit(benchmark::kNanosecond);

BENCHMARK_REGISTER_F(LockFreeBenchmark, RefCountDecrement)
    ->Unit(benchmark::kNanosecond);

BENCHMARK_REGISTER_F(LockFreeBenchmark, RefCountMixed)
    ->Unit(benchmark::kNanosecond);

// Logging benchmarks
static void BM_LoggingInfo(benchmark::State& state) {
    auto logger = ultra::common::Logger::instance();
    int counter = 0;
    
    for (auto _ : state) {
        logger->info("Benchmark log message {}", counter++);
    }
    
    state.SetItemsProcessed(state.iterations());
}

static void BM_LoggingDebug(benchmark::State& state) {
    auto logger = ultra::common::Logger::instance();
    int counter = 0;
    
    for (auto _ : state) {
        logger->debug("Debug benchmark log message {}", counter++);
    }
    
    state.SetItemsProcessed(state.iterations());
}

static void BM_LoggingError(benchmark::State& state) {
    auto logger = ultra::common::Logger::instance();
    int counter = 0;
    
    for (auto _ : state) {
        logger->error("Error benchmark log message {}", counter++);
    }
    
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BM_LoggingInfo)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_LoggingDebug)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_LoggingError)->Unit(benchmark::kNanosecond);

// Concurrent benchmarks
static void BM_CacheConcurrentReads(benchmark::State& state) {
    static ultra::cache::UltraCache<int, std::string> cache(10000);
    static std::once_flag init_flag;
    
    std::call_once(init_flag, []() {
        for (int i = 0; i < 5000; ++i) {
            cache.put(i, "concurrent_value_" + std::to_string(i));
        }
    });
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 4999);
    
    for (auto _ : state) {
        int key = dis(gen);
        auto result = cache.get(key);
        benchmark::DoNotOptimize(result);
    }
    
    state.SetItemsProcessed(state.iterations());
}

static void BM_CacheConcurrentWrites(benchmark::State& state) {
    static ultra::cache::UltraCache<int, std::string> cache(10000);
    static std::atomic<int> key_counter{10000};
    
    for (auto _ : state) {
        int key = key_counter.fetch_add(1);
        cache.put(key, "concurrent_write_" + std::to_string(key));
    }
    
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BM_CacheConcurrentReads)
    ->Threads(1)->Threads(2)->Threads(4)->Threads(8)
    ->Unit(benchmark::kNanosecond);

BENCHMARK(BM_CacheConcurrentWrites)
    ->Threads(1)->Threads(2)->Threads(4)->Threads(8)
    ->Unit(benchmark::kNanosecond);

// Latency distribution benchmarks
static void BM_LatencyDistribution(benchmark::State& state) {
    ultra::cache::UltraCache<int, std::string> cache(1000);
    std::vector<uint64_t> latencies;
    latencies.reserve(state.iterations());
    
    // Warm up
    for (int i = 0; i < 100; ++i) {
        cache.put(i, "warmup_value");
    }
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        auto result = cache.get(42);
        auto end = std::chrono::high_resolution_clock::now();
        
        uint64_t latency_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            end - start).count();
        latencies.push_back(latency_ns);
        
        benchmark::DoNotOptimize(result);
    }
    
    // Calculate and report percentiles
    std::sort(latencies.begin(), latencies.end());
    size_t count = latencies.size();
    
    if (count > 0) {
        state.counters["P50"] = latencies[count * 50 / 100];
        state.counters["P95"] = latencies[count * 95 / 100];
        state.counters["P99"] = latencies[count * 99 / 100];
        state.counters["P999"] = latencies[count * 999 / 1000];
        state.counters["Max"] = latencies[count - 1];
    }
    
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BM_LatencyDistribution)
    ->Iterations(10000)
    ->Unit(benchmark::kNanosecond);

// Memory bandwidth benchmarks
static void BM_MemoryBandwidth(benchmark::State& state) {
    size_t size = state.range(0);
    std::vector<uint8_t> src(size, 0x42);
    std::vector<uint8_t> dst(size);
    
    for (auto _ : state) {
        std::memcpy(dst.data(), src.data(), size);
        benchmark::DoNotOptimize(dst.data());
        benchmark::ClobberMemory();
    }
    
    state.SetBytesProcessed(state.iterations() * size);
}

BENCHMARK(BM_MemoryBandwidth)
    ->Range(1024, 1024*1024*16) // 1KB to 16MB
    ->Unit(benchmark::kNanosecond);

// CPU cache effects benchmark
static void BM_CacheEffects(benchmark::State& state) {
    size_t array_size = state.range(0);
    std::vector<int> array(array_size);
    
    // Initialize array
    for (size_t i = 0; i < array_size; ++i) {
        array[i] = i;
    }
    
    size_t sum = 0;
    for (auto _ : state) {
        // Sequential access pattern
        for (size_t i = 0; i < array_size; ++i) {
            sum += array[i];
        }
        benchmark::DoNotOptimize(sum);
    }
    
    state.SetItemsProcessed(state.iterations() * array_size);
    state.SetBytesProcessed(state.iterations() * array_size * sizeof(int));
}

BENCHMARK(BM_CacheEffects)
    ->Range(1024, 1024*1024) // 1K to 1M elements
    ->Unit(benchmark::kNanosecond);

BENCHMARK_MAIN();