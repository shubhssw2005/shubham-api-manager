#pragma once

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string>
#include <atomic>
#include <memory>
#include <vector>
#include <unordered_map>
#include <mutex>

// Define missing constants for non-Linux systems
#ifndef MAP_HUGETLB
#define MAP_HUGETLB 0
#endif
#ifndef MAP_HUGE_SHIFT
#define MAP_HUGE_SHIFT 0
#endif
#ifndef MAP_POPULATE
#define MAP_POPULATE 0
#endif
#ifndef MAP_LOCKED
#define MAP_LOCKED 0
#endif

namespace ultra {
namespace memory {

/**
 * Memory-mapped file I/O with huge pages support
 * Provides high-performance file operations with zero-copy semantics
 */
class MmapAllocator {
public:
    enum class HugePageSize {
        NONE = 0,
        SIZE_2MB = 2 * 1024 * 1024,
        SIZE_1GB = 1024 * 1024 * 1024
    };
    
    enum class AccessMode {
        READ_ONLY = PROT_READ,
        WRITE_ONLY = PROT_WRITE,
        READ_WRITE = PROT_READ | PROT_WRITE
    };
    
    enum class MapFlags {
        PRIVATE = MAP_PRIVATE,
        SHARED = MAP_SHARED,
        ANONYMOUS = MAP_ANONYMOUS,
        POPULATE = MAP_POPULATE,  // Pre-fault pages
        LOCKED = MAP_LOCKED       // Lock pages in memory
    };
    
    struct Config {
        HugePageSize huge_page_size = HugePageSize::SIZE_2MB;
        bool use_huge_pages = true;
        bool prefault_pages = true;
        bool lock_pages = false;
        size_t alignment = 4096; // Page alignment
    };
    
    explicit MmapAllocator(const Config& config = {});
    ~MmapAllocator();
    
    // Non-copyable, non-movable
    MmapAllocator(const MmapAllocator&) = delete;
    MmapAllocator& operator=(const MmapAllocator&) = delete;
    
    /**
     * Memory-mapped file handle
     */
    class MappedFile {
    public:
        MappedFile() = default;
        ~MappedFile();
        
        // Non-copyable, movable
        MappedFile(const MappedFile&) = delete;
        MappedFile& operator=(const MappedFile&) = delete;
        MappedFile(MappedFile&& other) noexcept;
        MappedFile& operator=(MappedFile&& other) noexcept;
        
        /**
         * Get pointer to mapped memory
         */
        void* data() const noexcept { return data_; }
        
        /**
         * Get size of mapped region
         */
        size_t size() const noexcept { return size_; }
        
        /**
         * Check if mapping is valid
         */
        bool is_valid() const noexcept { return data_ != nullptr && data_ != MAP_FAILED; }
        
        /**
         * Sync changes to disk
         * @param async If true, perform asynchronous sync
         */
        bool sync(bool async = false) const noexcept;
        
        /**
         * Advise kernel about access patterns
         */
        bool advise_sequential() const noexcept;
        bool advise_random() const noexcept;
        bool advise_willneed() const noexcept;
        bool advise_dontneed() const noexcept;
        
        /**
         * Lock/unlock pages in memory
         */
        bool lock_pages() const noexcept;
        bool unlock_pages() const noexcept;
        
        /**
         * Prefault pages to avoid page faults during access
         */
        bool prefault() const noexcept;
        
    private:
        friend class MmapAllocator;
        
        MappedFile(void* data, size_t size, int fd, bool huge_pages);
        
        void* data_ = nullptr;
        size_t size_ = 0;
        int fd_ = -1;
        bool huge_pages_ = false;
        
        void cleanup();
    };
    
    /**
     * Map existing file into memory
     * @param filename Path to file
     * @param access Access mode (read/write/both)
     * @param flags Mapping flags
     * @param offset Offset in file (must be page-aligned)
     * @param size Size to map (0 = entire file)
     * @return Mapped file handle
     */
    MappedFile map_file(const std::string& filename,
                       AccessMode access = AccessMode::READ_WRITE,
                       int flags = static_cast<int>(MapFlags::SHARED),
                       off_t offset = 0,
                       size_t size = 0);
    
    /**
     * Create and map new file
     * @param filename Path to new file
     * @param size Size of file to create
     * @param access Access mode
     * @param flags Mapping flags
     * @return Mapped file handle
     */
    MappedFile create_file(const std::string& filename,
                          size_t size,
                          AccessMode access = AccessMode::READ_WRITE,
                          int flags = static_cast<int>(MapFlags::SHARED));
    
    /**
     * Map anonymous memory (not backed by file)
     * @param size Size to allocate
     * @param access Access mode
     * @param flags Additional mapping flags
     * @return Mapped memory handle
     */
    MappedFile map_anonymous(size_t size,
                            AccessMode access = AccessMode::READ_WRITE,
                            int flags = 0);
    
    /**
     * Check if huge pages are available
     */
    static bool huge_pages_available(HugePageSize size = HugePageSize::SIZE_2MB);
    
    /**
     * Get system page size
     */
    static size_t get_page_size() noexcept;
    
    /**
     * Get huge page size
     */
    static size_t get_huge_page_size(HugePageSize size = HugePageSize::SIZE_2MB) noexcept;
    
    /**
     * Allocation statistics
     */
    struct Stats {
        std::atomic<uint64_t> files_mapped{0};
        std::atomic<uint64_t> files_unmapped{0};
        std::atomic<uint64_t> bytes_mapped{0};
        std::atomic<uint64_t> bytes_unmapped{0};
        std::atomic<uint64_t> huge_page_allocations{0};
        std::atomic<uint64_t> page_faults{0};
        std::atomic<uint64_t> sync_operations{0};
    };
    
    const Stats& get_stats() const noexcept { return stats_; }
    
private:
    Config config_;
    Stats stats_;
    std::mutex mappings_mutex_;
    std::unordered_map<void*, size_t> active_mappings_;
    
    void* do_mmap(void* addr, size_t length, int prot, int flags, int fd, off_t offset);
    bool should_use_huge_pages(size_t size) const noexcept;
    int get_huge_page_flags() const noexcept;
    void track_mapping(void* addr, size_t size);
    void untrack_mapping(void* addr);
};

/**
 * RAII wrapper for memory-mapped regions
 */
class ScopedMmap {
public:
    ScopedMmap(MmapAllocator& allocator, const std::string& filename, 
               MmapAllocator::AccessMode access = MmapAllocator::AccessMode::READ_WRITE)
        : file_(allocator.map_file(filename, access)) {}
    
    ScopedMmap(MmapAllocator& allocator, size_t size,
               MmapAllocator::AccessMode access = MmapAllocator::AccessMode::READ_WRITE)
        : file_(allocator.map_anonymous(size, access)) {}
    
    // Non-copyable, movable
    ScopedMmap(const ScopedMmap&) = delete;
    ScopedMmap& operator=(const ScopedMmap&) = delete;
    ScopedMmap(ScopedMmap&&) = default;
    ScopedMmap& operator=(ScopedMmap&&) = default;
    
    void* data() const noexcept { return file_.data(); }
    size_t size() const noexcept { return file_.size(); }
    bool is_valid() const noexcept { return file_.is_valid(); }
    
    MmapAllocator::MappedFile& get_file() noexcept { return file_; }
    const MmapAllocator::MappedFile& get_file() const noexcept { return file_; }
    
private:
    MmapAllocator::MappedFile file_;
};

/**
 * Memory-mapped vector for large datasets
 */
template<typename T>
class MmapVector {
public:
    using value_type = T;
    using size_type = size_t;
    using difference_type = ptrdiff_t;
    using reference = T&;
    using const_reference = const T&;
    using pointer = T*;
    using const_pointer = const T*;
    using iterator = T*;
    using const_iterator = const T*;
    
    MmapVector(MmapAllocator& allocator, const std::string& filename, size_t capacity)
        : allocator_(allocator), capacity_(capacity), size_(0) {
        
        size_t file_size = capacity * sizeof(T);
        file_ = allocator_.create_file(filename, file_size);
        
        if (!file_.is_valid()) {
            throw std::runtime_error("Failed to create memory-mapped file: " + filename);
        }
    }
    
    MmapVector(MmapAllocator& allocator, size_t capacity)
        : allocator_(allocator), capacity_(capacity), size_(0) {
        
        size_t file_size = capacity * sizeof(T);
        file_ = allocator_.map_anonymous(file_size);
        
        if (!file_.is_valid()) {
            throw std::runtime_error("Failed to create anonymous memory mapping");
        }
    }
    
    // Accessors
    reference operator[](size_type pos) { return data()[pos]; }
    const_reference operator[](size_type pos) const { return data()[pos]; }
    
    reference at(size_type pos) {
        if (pos >= size_) throw std::out_of_range("MmapVector::at");
        return data()[pos];
    }
    
    const_reference at(size_type pos) const {
        if (pos >= size_) throw std::out_of_range("MmapVector::at");
        return data()[pos];
    }
    
    reference front() { return data()[0]; }
    const_reference front() const { return data()[0]; }
    
    reference back() { return data()[size_ - 1]; }
    const_reference back() const { return data()[size_ - 1]; }
    
    pointer data() noexcept { return static_cast<pointer>(file_.data()); }
    const_pointer data() const noexcept { return static_cast<const_pointer>(file_.data()); }
    
    // Iterators
    iterator begin() noexcept { return data(); }
    const_iterator begin() const noexcept { return data(); }
    const_iterator cbegin() const noexcept { return data(); }
    
    iterator end() noexcept { return data() + size_; }
    const_iterator end() const noexcept { return data() + size_; }
    const_iterator cend() const noexcept { return data() + size_; }
    
    // Capacity
    bool empty() const noexcept { return size_ == 0; }
    size_type size() const noexcept { return size_; }
    size_type capacity() const noexcept { return capacity_; }
    
    // Modifiers
    void push_back(const T& value) {
        if (size_ >= capacity_) {
            throw std::runtime_error("MmapVector capacity exceeded");
        }
        new(data() + size_) T(value);
        ++size_;
    }
    
    void push_back(T&& value) {
        if (size_ >= capacity_) {
            throw std::runtime_error("MmapVector capacity exceeded");
        }
        new(data() + size_) T(std::move(value));
        ++size_;
    }
    
    template<typename... Args>
    void emplace_back(Args&&... args) {
        if (size_ >= capacity_) {
            throw std::runtime_error("MmapVector capacity exceeded");
        }
        new(data() + size_) T(std::forward<Args>(args)...);
        ++size_;
    }
    
    void pop_back() {
        if (size_ > 0) {
            data()[--size_].~T();
        }
    }
    
    void clear() {
        for (size_t i = 0; i < size_; ++i) {
            data()[i].~T();
        }
        size_ = 0;
    }
    
    void resize(size_type new_size) {
        if (new_size > capacity_) {
            throw std::runtime_error("MmapVector resize exceeds capacity");
        }
        
        if (new_size > size_) {
            for (size_t i = size_; i < new_size; ++i) {
                new(data() + i) T();
            }
        } else if (new_size < size_) {
            for (size_t i = new_size; i < size_; ++i) {
                data()[i].~T();
            }
        }
        size_ = new_size;
    }
    
    // File operations
    bool sync(bool async = false) { return file_.sync(async); }
    bool prefault() { return file_.prefault(); }
    
private:
    MmapAllocator& allocator_;
    MmapAllocator::MappedFile file_;
    size_type capacity_;
    size_type size_;
};

} // namespace memory
} // namespace ultra