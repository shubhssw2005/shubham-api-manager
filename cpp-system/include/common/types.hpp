#pragma once

#include <cstdint>
#include <chrono>
#include <atomic>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <unordered_set>
#include <unordered_map>

namespace ultra {

// Basic integer types
using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

using i8 = std::int8_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;

// Timestamp type
using timestamp_t = std::chrono::time_point<std::chrono::steady_clock>;

// Cache line size for alignment
constexpr size_t CACHE_LINE_SIZE = 64;

// Aligned atomic for performance
template<typename T>
struct alignas(CACHE_LINE_SIZE) aligned_atomic : public std::atomic<T> {
    using std::atomic<T>::atomic;
    
    // Default constructor
    aligned_atomic() : std::atomic<T>{} {}
    
    // Constructor from value
    aligned_atomic(T value) : std::atomic<T>(value) {}
};

// Force inline macro
#define ULTRA_FORCE_INLINE __attribute__((always_inline)) inline

// Likely/unlikely macros for branch prediction
#define ULTRA_LIKELY(x) __builtin_expect(!!(x), 1)
#define ULTRA_UNLIKELY(x) __builtin_expect(!!(x), 0)

} // namespace ultra