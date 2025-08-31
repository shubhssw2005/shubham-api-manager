#pragma once

#include <memory>
#include <vector>
#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <atomic>

namespace ultra_cpp {
namespace performance {

/**
 * Cache line size constants
 */
constexpr size_t CACHE_LINE_SIZE = 64;
constexpr size_t L1_CACHE_SIZE = 32 * 1024;      // 32KB typical L1
constexpr size_t L2_CACHE_SIZE = 256 * 1024;     // 256KB typical L2
constexpr size_t L3_CACHE_SIZE = 8 * 1024 * 1024; // 8MB typical L3

/**
 * Cache-line aligned allocator
 */
template<typename T>
class CacheAlignedAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    
    template<typename U>
    struct rebind {
        using other = CacheAlignedAllocator<U>;
    };
    
    CacheAlignedAllocator() noexcept = default;
    
    template<typename U>
    CacheAlignedAllocator(const CacheAlignedAllocator<U>&) noexcept {}
    
    pointer allocate(size_type n) {
        if (n == 0) return nullptr;
        
        size_type bytes = n * sizeof(T);
        size_type aligned_bytes = (bytes + CACHE_LINE_SIZE - 1) & ~(CACHE_LINE_SIZE - 1);
        
        void* ptr = std::aligned_alloc(CACHE_LINE_SIZE, aligned_bytes);
        if (!ptr) {
            throw std::bad_alloc();
        }
        
        return static_cast<pointer>(ptr);
    }
    
    void deallocate(pointer p, size_type) noexcept {
        std::free(p);
    }
    
    template<typename U>
    bool operator==(const CacheAlignedAllocator<U>&) const noexcept {
        return true;
    }
    
    template<typename U>
    bool operator!=(const CacheAlignedAllocator<U>&) const noexcept {
        return false;
    }
};

/**
 * Cache-friendly vector with prefetching
 */
template<typename T>
class CacheFriendlyVector {
public:
    using value_type = T;
    using allocator_type = CacheAlignedAllocator<T>;
    using size_type = std::size_t;
    using iterator = T*;
    using const_iterator = const T*;
    
    CacheFriendlyVector() : data_(nullptr), size_(0), capacity_(0) {}
    
    explicit CacheFriendlyVector(size_type count) 
        : CacheFriendlyVector() {
        resize(count);
    }
    
    ~CacheFriendlyVector() {
        clear();
        if (data_) {
            allocator_.deallocate(data_, capacity_);
        }
    }
    
    // Move constructor and assignment
    CacheFriendlyVector(CacheFriendlyVector&& other) noexcept
        : data_(other.data_), size_(other.size_), capacity_(other.capacity_) {
        other.data_ = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
    }
    
    CacheFriendlyVector& operator=(CacheFriendlyVector&& other) noexcept {
        if (this != &other) {
            clear();
            if (data_) {
                allocator_.deallocate(data_, capacity_);
            }
            
            data_ = other.data_;
            size_ = other.size_;
            capacity_ = other.capacity_;
            
            other.data_ = nullptr;
            other.size_ = 0;
            other.capacity_ = 0;
        }
        return *this;
    }
    
    // Disable copy constructor and assignment
    CacheFriendlyVector(const CacheFriendlyVector&) = delete;
    CacheFriendlyVector& operator=(const CacheFriendlyVector&) = delete;
    
    void resize(size_type new_size) {
        if (new_size > capacity_) {
            reserve(new_size);
        }
        
        // Construct new elements
        for (size_type i = size_; i < new_size; ++i) {
            new (data_ + i) T();
        }
        
        // Destroy excess elements
        for (size_type i = new_size; i < size_; ++i) {
            data_[i].~T();
        }
        
        size_ = new_size;
    }
    
    void reserve(size_type new_capacity) {
        if (new_capacity <= capacity_) return;
        
        // Align capacity to cache line boundaries
        size_type aligned_capacity = ((new_capacity * sizeof(T) + CACHE_LINE_SIZE - 1) 
                                    / CACHE_LINE_SIZE) * CACHE_LINE_SIZE / sizeof(T);
        
        T* new_data = allocator_.allocate(aligned_capacity);
        
        // Move existing elements
        for (size_type i = 0; i < size_; ++i) {
            new (new_data + i) T(std::move(data_[i]));
            data_[i].~T();
        }
        
        if (data_) {
            allocator_.deallocate(data_, capacity_);
        }
        
        data_ = new_data;
        capacity_ = aligned_capacity;
    }
    
    void push_back(const T& value) {
        if (size_ >= capacity_) {
            reserve(capacity_ == 0 ? 1 : capacity_ * 2);
        }
        new (data_ + size_) T(value);
        ++size_;
    }
    
    void push_back(T&& value) {
        if (size_ >= capacity_) {
            reserve(capacity_ == 0 ? 1 : capacity_ * 2);
        }
        new (data_ + size_) T(std::move(value));
        ++size_;
    }
    
    void clear() {
        for (size_type i = 0; i < size_; ++i) {
            data_[i].~T();
        }
        size_ = 0;
    }
    
    // Access with prefetching
    T& operator[](size_type index) {
        prefetch_next(index);
        return data_[index];
    }
    
    const T& operator[](size_type index) const {
        prefetch_next(index);
        return data_[index];
    }
    
    T& at(size_type index) {
        if (index >= size_) {
            throw std::out_of_range("Index out of range");
        }
        return (*this)[index];
    }
    
    // Iterators
    iterator begin() { return data_; }
    iterator end() { return data_ + size_; }
    const_iterator begin() const { return data_; }
    const_iterator end() const { return data_ + size_; }
    const_iterator cbegin() const { return data_; }
    const_iterator cend() const { return data_ + size_; }
    
    // Capacity
    size_type size() const { return size_; }
    size_type capacity() const { return capacity_; }
    bool empty() const { return size_ == 0; }
    
    // Data access
    T* data() { return data_; }
    const T* data() const { return data_; }

private:
    T* data_;
    size_type size_;
    size_type capacity_;
    allocator_type allocator_;
    
    void prefetch_next(size_type index) const {
        // Prefetch next cache line if we're near the end of current one
        constexpr size_type elements_per_line = CACHE_LINE_SIZE / sizeof(T);
        if ((index + 1) % elements_per_line == 0 && index + elements_per_line < size_) {
#ifdef __builtin_prefetch
            __builtin_prefetch(data_ + index + elements_per_line, 0, 3);
#endif
        }
    }
};

/**
 * Structure of Arrays (SoA) container for better cache locality
 */
template<typename... Types>
class StructureOfArrays {
public:
    using size_type = std::size_t;
    
    StructureOfArrays() = default;
    
    explicit StructureOfArrays(size_type count) {
        resize(count);
    }
    
    void resize(size_type new_size) {
        resize_impl<0>(new_size);
    }
    
    void reserve(size_type new_capacity) {
        reserve_impl<0>(new_capacity);
    }
    
    template<size_t Index>
    auto& get_array() {
        return std::get<Index>(arrays_);
    }
    
    template<size_t Index>
    const auto& get_array() const {
        return std::get<Index>(arrays_);
    }
    
    template<size_t Index>
    auto& get(size_type i) {
        return get_array<Index>()[i];
    }
    
    template<size_t Index>
    const auto& get(size_type i) const {
        return get_array<Index>()[i];
    }
    
    void push_back(const Types&... values) {
        push_back_impl<0>(values...);
    }
    
    size_type size() const {
        return std::get<0>(arrays_).size();
    }
    
    bool empty() const {
        return std::get<0>(arrays_).empty();
    }
    
    void clear() {
        clear_impl<0>();
    }

private:
    std::tuple<CacheFriendlyVector<Types>...> arrays_;
    
    template<size_t Index>
    void resize_impl(size_type new_size) {
        std::get<Index>(arrays_).resize(new_size);
        if constexpr (Index + 1 < sizeof...(Types)) {
            resize_impl<Index + 1>(new_size);
        }
    }
    
    template<size_t Index>
    void reserve_impl(size_type new_capacity) {
        std::get<Index>(arrays_).reserve(new_capacity);
        if constexpr (Index + 1 < sizeof...(Types)) {
            reserve_impl<Index + 1>(new_capacity);
        }
    }
    
    template<size_t Index, typename First, typename... Rest>
    void push_back_impl(const First& first, const Rest&... rest) {
        std::get<Index>(arrays_).push_back(first);
        if constexpr (sizeof...(Rest) > 0) {
            push_back_impl<Index + 1>(rest...);
        }
    }
    
    template<size_t Index>
    void clear_impl() {
        std::get<Index>(arrays_).clear();
        if constexpr (Index + 1 < sizeof...(Types)) {
            clear_impl<Index + 1>();
        }
    }
};

/**
 * Cache-friendly hash table with linear probing
 */
template<typename Key, typename Value, typename Hash = std::hash<Key>>
class CacheFriendlyHashMap {
public:
    struct Entry {
        alignas(CACHE_LINE_SIZE) Key key;
        Value value;
        std::atomic<bool> occupied{false};
        std::atomic<bool> deleted{false};
    };
    
    using size_type = std::size_t;
    
    explicit CacheFriendlyHashMap(size_type initial_capacity = 1024)
        : capacity_(next_power_of_two(initial_capacity))
        , mask_(capacity_ - 1)
        , size_(0) {
        entries_ = allocator_.allocate(capacity_);
        
        // Initialize entries
        for (size_type i = 0; i < capacity_; ++i) {
            new (entries_ + i) Entry();
        }
    }
    
    ~CacheFriendlyHashMap() {
        if (entries_) {
            for (size_type i = 0; i < capacity_; ++i) {
                entries_[i].~Entry();
            }
            allocator_.deallocate(entries_, capacity_);
        }
    }
    
    // Disable copy constructor and assignment
    CacheFriendlyHashMap(const CacheFriendlyHashMap&) = delete;
    CacheFriendlyHashMap& operator=(const CacheFriendlyHashMap&) = delete;
    
    bool insert(const Key& key, const Value& value) {
        if (size_ * 4 >= capacity_ * 3) { // 75% load factor
            rehash();
        }
        
        size_type index = hash_(key) & mask_;
        
        for (size_type i = 0; i < capacity_; ++i) {
            Entry& entry = entries_[index];
            
            bool expected = false;
            if (entry.occupied.compare_exchange_weak(expected, true)) {
                // Successfully claimed this slot
                entry.key = key;
                entry.value = value;
                entry.deleted.store(false);
                ++size_;
                return true;
            } else if (!entry.deleted.load() && entry.key == key) {
                // Key already exists, update value
                entry.value = value;
                return false;
            }
            
            index = (index + 1) & mask_;
        }
        
        // Table is full (shouldn't happen with proper load factor)
        return false;
    }
    
    bool find(const Key& key, Value& value) const {
        size_type index = hash_(key) & mask_;
        
        for (size_type i = 0; i < capacity_; ++i) {
            const Entry& entry = entries_[index];
            
            if (!entry.occupied.load()) {
                return false; // Empty slot, key not found
            }
            
            if (!entry.deleted.load() && entry.key == key) {
                value = entry.value;
                return true;
            }
            
            index = (index + 1) & mask_;
        }
        
        return false;
    }
    
    bool erase(const Key& key) {
        size_type index = hash_(key) & mask_;
        
        for (size_type i = 0; i < capacity_; ++i) {
            Entry& entry = entries_[index];
            
            if (!entry.occupied.load()) {
                return false; // Empty slot, key not found
            }
            
            if (!entry.deleted.load() && entry.key == key) {
                entry.deleted.store(true);
                --size_;
                return true;
            }
            
            index = (index + 1) & mask_;
        }
        
        return false;
    }
    
    size_type size() const { return size_; }
    size_type capacity() const { return capacity_; }
    bool empty() const { return size_ == 0; }
    
    double load_factor() const {
        return static_cast<double>(size_) / capacity_;
    }

private:
    Entry* entries_;
    size_type capacity_;
    size_type mask_;
    std::atomic<size_type> size_;
    Hash hash_;
    CacheAlignedAllocator<Entry> allocator_;
    
    static size_type next_power_of_two(size_type n) {
        --n;
        n |= n >> 1;
        n |= n >> 2;
        n |= n >> 4;
        n |= n >> 8;
        n |= n >> 16;
        n |= n >> 32;
        return ++n;
    }
    
    void rehash() {
        size_type old_capacity = capacity_;
        Entry* old_entries = entries_;
        
        capacity_ *= 2;
        mask_ = capacity_ - 1;
        entries_ = allocator_.allocate(capacity_);
        
        // Initialize new entries
        for (size_type i = 0; i < capacity_; ++i) {
            new (entries_ + i) Entry();
        }
        
        size_type old_size = size_;
        size_ = 0;
        
        // Rehash existing entries
        for (size_type i = 0; i < old_capacity; ++i) {
            Entry& old_entry = old_entries[i];
            if (old_entry.occupied.load() && !old_entry.deleted.load()) {
                insert(old_entry.key, old_entry.value);
            }
        }
        
        // Clean up old entries
        for (size_type i = 0; i < old_capacity; ++i) {
            old_entries[i].~Entry();
        }
        allocator_.deallocate(old_entries, old_capacity);
    }
};

/**
 * Memory prefetching utilities
 */
class PrefetchManager {
public:
    enum class Locality {
        TEMPORAL_NO_REUSE = 0,    // No temporal locality (use once)
        TEMPORAL_LOW = 1,         // Low temporal locality
        TEMPORAL_MODERATE = 2,    // Moderate temporal locality
        TEMPORAL_HIGH = 3         // High temporal locality (default)
    };
    
    enum class AccessType {
        READ = 0,
        WRITE = 1
    };
    
    static void prefetch(const void* addr, AccessType access = AccessType::READ, 
                        Locality locality = Locality::TEMPORAL_HIGH) {
#ifdef __builtin_prefetch
        __builtin_prefetch(addr, static_cast<int>(access), static_cast<int>(locality));
#else
        (void)addr; (void)access; (void)locality;
#endif
    }
    
    template<typename T>
    static void prefetch_range(const T* start, size_t count, 
                              AccessType access = AccessType::READ,
                              Locality locality = Locality::TEMPORAL_HIGH) {
        const char* ptr = reinterpret_cast<const char*>(start);
        const char* end = ptr + count * sizeof(T);
        
        for (; ptr < end; ptr += CACHE_LINE_SIZE) {
            prefetch(ptr, access, locality);
        }
    }
    
    template<typename Container>
    static void prefetch_container(const Container& container,
                                  AccessType access = AccessType::READ,
                                  Locality locality = Locality::TEMPORAL_HIGH) {
        if (!container.empty()) {
            prefetch_range(container.data(), container.size(), access, locality);
        }
    }
};

/**
 * Cache performance analyzer
 */
class CacheAnalyzer {
public:
    struct CacheStats {
        uint64_t l1_hits = 0;
        uint64_t l1_misses = 0;
        uint64_t l2_hits = 0;
        uint64_t l2_misses = 0;
        uint64_t l3_hits = 0;
        uint64_t l3_misses = 0;
        uint64_t tlb_misses = 0;
        uint64_t branch_mispredictions = 0;
        
        double l1_hit_rate() const {
            uint64_t total = l1_hits + l1_misses;
            return total > 0 ? static_cast<double>(l1_hits) / total : 0.0;
        }
        
        double l2_hit_rate() const {
            uint64_t total = l2_hits + l2_misses;
            return total > 0 ? static_cast<double>(l2_hits) / total : 0.0;
        }
        
        double l3_hit_rate() const {
            uint64_t total = l3_hits + l3_misses;
            return total > 0 ? static_cast<double>(l3_hits) / total : 0.0;
        }
    };
    
    CacheAnalyzer();
    ~CacheAnalyzer();
    
    void start_monitoring();
    void stop_monitoring();
    CacheStats get_stats() const;
    void reset_stats();
    
    // Benchmark cache-friendly vs cache-unfriendly access patterns
    template<typename T>
    static double benchmark_sequential_access(const std::vector<T>& data, size_t iterations = 1000);
    
    template<typename T>
    static double benchmark_random_access(const std::vector<T>& data, size_t iterations = 1000);
    
    template<typename T>
    static double benchmark_strided_access(const std::vector<T>& data, size_t stride, size_t iterations = 1000);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace performance
} // namespace ultra_cpp