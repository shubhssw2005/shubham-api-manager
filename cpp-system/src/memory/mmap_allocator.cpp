#include "memory/mmap_allocator.hpp"
#include <sys/stat.h>
#include <errno.h>
#include <cstring>
#include <stdexcept>
#include <algorithm>

namespace ultra {
namespace memory {

MmapAllocator::MmapAllocator(const Config& config) : config_(config) {
    // Validate configuration
    if (config_.alignment == 0 || (config_.alignment & (config_.alignment - 1)) != 0) {
        throw std::invalid_argument("Alignment must be a power of 2");
    }
    
    // Check huge page availability
    if (config_.use_huge_pages && !huge_pages_available(config_.huge_page_size)) {
        std::cerr << "Warning: Requested huge pages not available, falling back to regular pages\n";
    }
}

MmapAllocator::~MmapAllocator() {
    // Cleanup tracked mappings
    std::lock_guard<std::mutex> lock(mappings_mutex_);
    for (const auto& [addr, size] : active_mappings_) {
        munmap(addr, size);
    }
}

// MappedFile implementation
MmapAllocator::MappedFile::MappedFile(void* data, size_t size, int fd, bool huge_pages)
    : data_(data), size_(size), fd_(fd), huge_pages_(huge_pages) {}

MmapAllocator::MappedFile::~MappedFile() {
    cleanup();
}

MmapAllocator::MappedFile::MappedFile(MappedFile&& other) noexcept
    : data_(other.data_), size_(other.size_), fd_(other.fd_), huge_pages_(other.huge_pages_) {
    other.data_ = nullptr;
    other.size_ = 0;
    other.fd_ = -1;
    other.huge_pages_ = false;
}

MmapAllocator::MappedFile& MmapAllocator::MappedFile::operator=(MappedFile&& other) noexcept {
    if (this != &other) {
        cleanup();
        
        data_ = other.data_;
        size_ = other.size_;
        fd_ = other.fd_;
        huge_pages_ = other.huge_pages_;
        
        other.data_ = nullptr;
        other.size_ = 0;
        other.fd_ = -1;
        other.huge_pages_ = false;
    }
    return *this;
}

bool MmapAllocator::MappedFile::sync(bool async) const noexcept {
    if (!is_valid()) return false;
    
    int flags = async ? MS_ASYNC : MS_SYNC;
    return msync(data_, size_, flags) == 0;
}

bool MmapAllocator::MappedFile::advise_sequential() const noexcept {
    if (!is_valid()) return false;
    return madvise(data_, size_, MADV_SEQUENTIAL) == 0;
}

bool MmapAllocator::MappedFile::advise_random() const noexcept {
    if (!is_valid()) return false;
    return madvise(data_, size_, MADV_RANDOM) == 0;
}

bool MmapAllocator::MappedFile::advise_willneed() const noexcept {
    if (!is_valid()) return false;
    return madvise(data_, size_, MADV_WILLNEED) == 0;
}

bool MmapAllocator::MappedFile::advise_dontneed() const noexcept {
    if (!is_valid()) return false;
    return madvise(data_, size_, MADV_DONTNEED) == 0;
}

bool MmapAllocator::MappedFile::lock_pages() const noexcept {
    if (!is_valid()) return false;
    return mlock(data_, size_) == 0;
}

bool MmapAllocator::MappedFile::unlock_pages() const noexcept {
    if (!is_valid()) return false;
    return munlock(data_, size_) == 0;
}

bool MmapAllocator::MappedFile::prefault() const noexcept {
    if (!is_valid()) return false;
    
    // Touch every page to prefault
    const size_t page_size = get_page_size();
    volatile char* ptr = static_cast<volatile char*>(data_);
    
    for (size_t offset = 0; offset < size_; offset += page_size) {
        // Read from each page to trigger page fault
        volatile char dummy = ptr[offset];
        (void)dummy; // Suppress unused variable warning
    }
    
    return true;
}

void MmapAllocator::MappedFile::cleanup() {
    if (is_valid()) {
        munmap(data_, size_);
        data_ = nullptr;
        size_ = 0;
    }
    
    if (fd_ >= 0) {
        close(fd_);
        fd_ = -1;
    }
}

// MmapAllocator implementation
MmapAllocator::MappedFile MmapAllocator::map_file(const std::string& filename,
                                                  AccessMode access,
                                                  int flags,
                                                  off_t offset,
                                                  size_t size) {
    // Open file
    int open_flags = O_RDONLY;
    if (access == AccessMode::WRITE_ONLY) {
        open_flags = O_WRONLY;
    } else if (access == AccessMode::READ_WRITE) {
        open_flags = O_RDWR;
    }
    
    int fd = open(filename.c_str(), open_flags);
    if (fd < 0) {
        throw std::runtime_error("Failed to open file: " + filename + " (" + strerror(errno) + ")");
    }
    
    // Get file size if not specified
    if (size == 0) {
        struct stat st;
        if (fstat(fd, &st) < 0) {
            close(fd);
            throw std::runtime_error("Failed to get file size: " + filename);
        }
        size = st.st_size - offset;
    }
    
    // Perform mapping
    void* addr = do_mmap(nullptr, size, static_cast<int>(access), flags, fd, offset);
    if (addr == MAP_FAILED) {
        close(fd);
        throw std::runtime_error("Failed to map file: " + filename + " (" + strerror(errno) + ")");
    }
    
    // Track mapping
    track_mapping(addr, size);
    
    // Update statistics
    stats_.files_mapped.fetch_add(1, std::memory_order_relaxed);
    stats_.bytes_mapped.fetch_add(size, std::memory_order_relaxed);
    
    bool huge_pages = should_use_huge_pages(size);
    MappedFile mapped_file(addr, size, fd, huge_pages);
    
    // Apply optimizations
    if (config_.prefault_pages) {
        mapped_file.prefault();
    }
    
    if (config_.lock_pages) {
        mapped_file.lock_pages();
    }
    
    return mapped_file;
}

MmapAllocator::MappedFile MmapAllocator::create_file(const std::string& filename,
                                                     size_t size,
                                                     AccessMode access,
                                                     int flags) {
    // Create file
    int fd = open(filename.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0644);
    if (fd < 0) {
        throw std::runtime_error("Failed to create file: " + filename + " (" + strerror(errno) + ")");
    }
    
    // Set file size
    if (ftruncate(fd, size) < 0) {
        close(fd);
        throw std::runtime_error("Failed to set file size: " + filename);
    }
    
    // Perform mapping
    void* addr = do_mmap(nullptr, size, static_cast<int>(access), flags, fd, 0);
    if (addr == MAP_FAILED) {
        close(fd);
        throw std::runtime_error("Failed to map created file: " + filename + " (" + strerror(errno) + ")");
    }
    
    // Track mapping
    track_mapping(addr, size);
    
    // Update statistics
    stats_.files_mapped.fetch_add(1, std::memory_order_relaxed);
    stats_.bytes_mapped.fetch_add(size, std::memory_order_relaxed);
    
    bool huge_pages = should_use_huge_pages(size);
    MappedFile mapped_file(addr, size, fd, huge_pages);
    
    // Apply optimizations
    if (config_.prefault_pages) {
        mapped_file.prefault();
    }
    
    if (config_.lock_pages) {
        mapped_file.lock_pages();
    }
    
    return mapped_file;
}

MmapAllocator::MappedFile MmapAllocator::map_anonymous(size_t size,
                                                       AccessMode access,
                                                       int flags) {
    // Add anonymous flag
    flags |= MAP_ANONYMOUS | MAP_PRIVATE;
    
    // Add huge page flags if appropriate (Linux only)
#ifdef __linux__
    if (should_use_huge_pages(size)) {
        flags |= get_huge_page_flags();
        stats_.huge_page_allocations.fetch_add(1, std::memory_order_relaxed);
    }
#endif
    
    // Add populate flag if prefaulting is enabled
    if (config_.prefault_pages) {
        flags |= MAP_POPULATE;
    }
    
    // Add locked flag if page locking is enabled
    if (config_.lock_pages) {
        flags |= MAP_LOCKED;
    }
    
    // Perform mapping
    void* addr = do_mmap(nullptr, size, static_cast<int>(access), flags, -1, 0);
    if (addr == MAP_FAILED) {
        throw std::runtime_error("Failed to map anonymous memory (" + std::string(strerror(errno)) + ")");
    }
    
    // Track mapping
    track_mapping(addr, size);
    
    // Update statistics
    stats_.files_mapped.fetch_add(1, std::memory_order_relaxed);
    stats_.bytes_mapped.fetch_add(size, std::memory_order_relaxed);
    
    bool huge_pages = should_use_huge_pages(size);
    return MappedFile(addr, size, -1, huge_pages);
}

bool MmapAllocator::huge_pages_available(HugePageSize size) {
    // Check if huge pages are available by trying to read from /proc/meminfo
    std::string proc_file;
    switch (size) {
        case HugePageSize::SIZE_2MB:
            proc_file = "/proc/meminfo";
            break;
        case HugePageSize::SIZE_1GB:
            proc_file = "/proc/meminfo";
            break;
        default:
            return false;
    }
    
    FILE* f = fopen(proc_file.c_str(), "r");
    if (!f) return false;
    
    char line[256];
    bool found = false;
    
    while (fgets(line, sizeof(line), f)) {
        if ((size == HugePageSize::SIZE_2MB && strstr(line, "HugePages_Total:")) ||
            (size == HugePageSize::SIZE_1GB && strstr(line, "HugePages_Total:"))) {
            int total = 0;
            if (sscanf(line, "%*s %d", &total) == 1 && total > 0) {
                found = true;
            }
            break;
        }
    }
    
    fclose(f);
    return found;
}

size_t MmapAllocator::get_page_size() noexcept {
    static size_t page_size = sysconf(_SC_PAGESIZE);
    return page_size;
}

size_t MmapAllocator::get_huge_page_size(HugePageSize size) noexcept {
    return static_cast<size_t>(size);
}

void* MmapAllocator::do_mmap(void* addr, size_t length, int prot, int flags, int fd, off_t offset) {
    // Align size to page boundary
    size_t page_size = get_page_size();
    length = (length + page_size - 1) & ~(page_size - 1);
    
    // Align offset to page boundary
    offset = offset & ~(page_size - 1);
    
    return mmap(addr, length, prot, flags, fd, offset);
}

bool MmapAllocator::should_use_huge_pages(size_t size) const noexcept {
    if (!config_.use_huge_pages) return false;
    
    size_t huge_page_size = get_huge_page_size(config_.huge_page_size);
    return size >= huge_page_size && huge_pages_available(config_.huge_page_size);
}

int MmapAllocator::get_huge_page_flags() const noexcept {
#ifdef __linux__
    switch (config_.huge_page_size) {
        case HugePageSize::SIZE_2MB:
            return MAP_HUGETLB | (21 << MAP_HUGE_SHIFT); // 2^21 = 2MB
        case HugePageSize::SIZE_1GB:
            return MAP_HUGETLB | (30 << MAP_HUGE_SHIFT); // 2^30 = 1GB
        default:
            return 0;
    }
#else
    return 0; // No huge page support on non-Linux systems
#endif
}

void MmapAllocator::track_mapping(void* addr, size_t size) {
    std::lock_guard<std::mutex> lock(mappings_mutex_);
    active_mappings_[addr] = size;
}

void MmapAllocator::untrack_mapping(void* addr) {
    std::lock_guard<std::mutex> lock(mappings_mutex_);
    auto it = active_mappings_.find(addr);
    if (it != active_mappings_.end()) {
        stats_.files_unmapped.fetch_add(1, std::memory_order_relaxed);
        stats_.bytes_unmapped.fetch_add(it->second, std::memory_order_relaxed);
        active_mappings_.erase(it);
    }
}

} // namespace memory
} // namespace ultra