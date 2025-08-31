#include "gpu-compute/gpu_compute_engine.hpp"
#include "common/logger.hpp"
#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include <mutex>
#include <unordered_map>
#include <algorithm>
#include <cassert>

namespace ultra {
namespace gpu {

// Memory block structure for the pool
struct MemoryBlock {
    void* ptr;
    size_t size;
    bool is_free;
    size_t alignment;
    
    MemoryBlock(void* p, size_t s, size_t align = 256) 
        : ptr(p), size(s), is_free(true), alignment(align) {}
};

class CUDAMemoryPool::Impl {
public:
    size_t pool_size_;
    void* base_ptr_;
    std::vector<MemoryBlock> blocks_;
    std::mutex pool_mutex_;
    size_t used_size_;
    
    // Free list organized by size for efficient allocation
    std::unordered_map<size_t, std::vector<size_t>> free_blocks_by_size_;
    
    explicit Impl(size_t pool_size) 
        : pool_size_(pool_size), base_ptr_(nullptr), used_size_(0) {
        
        // Allocate the entire pool at once
        cudaError_t error = cudaMalloc(&base_ptr_, pool_size_);
        if (error != cudaSuccess) {
            LOG_ERROR("Failed to allocate GPU memory pool: {}", cudaGetErrorString(error));
            throw std::runtime_error("Failed to allocate GPU memory pool");
        }
        
        // Initialize with one large free block
        blocks_.emplace_back(base_ptr_, pool_size_);
        free_blocks_by_size_[pool_size_].push_back(0);
        
        LOG_INFO("CUDA Memory Pool initialized with {} MB", pool_size_ / (1024 * 1024));
    }
    
    ~Impl() {
        if (base_ptr_) {
            cudaFree(base_ptr_);
            LOG_INFO("CUDA Memory Pool destroyed");
        }
    }
    
    void* allocate(size_t size, size_t alignment) {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        
        // Align size to the requested alignment
        size_t aligned_size = align_size(size, alignment);
        
        // Find a suitable free block
        size_t block_idx = find_free_block(aligned_size);
        if (block_idx == SIZE_MAX) {
            LOG_WARNING("GPU memory pool allocation failed: no suitable block for {} bytes", aligned_size);
            return nullptr;
        }
        
        MemoryBlock& block = blocks_[block_idx];
        
        // Remove from free list
        remove_from_free_list(block_idx);
        
        // Split block if necessary
        if (block.size > aligned_size + alignment) {
            split_block(block_idx, aligned_size);
        }
        
        block.is_free = false;
        used_size_ += block.size;
        
        LOG_DEBUG("GPU memory allocated: {} bytes at {}", block.size, block.ptr);
        return block.ptr;
    }
    
    void deallocate(void* ptr) {
        if (!ptr) return;
        
        std::lock_guard<std::mutex> lock(pool_mutex_);
        
        // Find the block containing this pointer
        auto it = std::find_if(blocks_.begin(), blocks_.end(),
            [ptr](const MemoryBlock& block) {
                return block.ptr == ptr;
            });
        
        if (it == blocks_.end()) {
            LOG_WARNING("Attempted to deallocate invalid GPU pointer: {}", ptr);
            return;
        }
        
        size_t block_idx = std::distance(blocks_.begin(), it);
        MemoryBlock& block = blocks_[block_idx];
        
        if (block.is_free) {
            LOG_WARNING("Attempted to deallocate already free GPU pointer: {}", ptr);
            return;
        }
        
        block.is_free = true;
        used_size_ -= block.size;
        
        // Try to merge with adjacent free blocks
        merge_adjacent_blocks(block_idx);
        
        // Add to free list
        add_to_free_list(block_idx);
        
        LOG_DEBUG("GPU memory deallocated: {} bytes at {}", block.size, ptr);
    }
    
private:
    size_t align_size(size_t size, size_t alignment) {
        return (size + alignment - 1) & ~(alignment - 1);
    }
    
    size_t find_free_block(size_t size) {
        // First try exact size match
        auto it = free_blocks_by_size_.find(size);
        if (it != free_blocks_by_size_.end() && !it->second.empty()) {
            size_t block_idx = it->second.back();
            return block_idx;
        }
        
        // Find smallest block that fits
        size_t best_idx = SIZE_MAX;
        size_t best_size = SIZE_MAX;
        
        for (const auto& [block_size, indices] : free_blocks_by_size_) {
            if (block_size >= size && block_size < best_size && !indices.empty()) {
                best_size = block_size;
                best_idx = indices.back();
            }
        }
        
        return best_idx;
    }
    
    void split_block(size_t block_idx, size_t size) {
        MemoryBlock& block = blocks_[block_idx];
        
        if (block.size <= size) return;
        
        // Create new block for the remaining space
        void* new_ptr = static_cast<char*>(block.ptr) + size;
        size_t new_size = block.size - size;
        
        blocks_.emplace_back(new_ptr, new_size, block.alignment);
        size_t new_block_idx = blocks_.size() - 1;
        
        // Update original block
        block.size = size;
        
        // Add new block to free list
        add_to_free_list(new_block_idx);
    }
    
    void merge_adjacent_blocks(size_t block_idx) {
        MemoryBlock& block = blocks_[block_idx];
        
        // Try to merge with next block
        for (size_t i = 0; i < blocks_.size(); ++i) {
            if (i == block_idx) continue;
            
            MemoryBlock& other = blocks_[i];
            if (!other.is_free) continue;
            
            // Check if blocks are adjacent
            char* block_end = static_cast<char*>(block.ptr) + block.size;
            char* other_start = static_cast<char*>(other.ptr);
            
            if (block_end == other_start) {
                // Merge other into block
                remove_from_free_list(i);
                block.size += other.size;
                
                // Remove the merged block
                blocks_.erase(blocks_.begin() + i);
                if (i < block_idx) block_idx--;
                break;
            }
            
            // Check reverse direction
            char* other_end = static_cast<char*>(other.ptr) + other.size;
            char* block_start = static_cast<char*>(block.ptr);
            
            if (other_end == block_start) {
                // Merge block into other
                remove_from_free_list(i);
                other.size += block.size;
                other.ptr = block.ptr;
                
                // Remove the current block
                blocks_.erase(blocks_.begin() + block_idx);
                break;
            }
        }
    }
    
    void add_to_free_list(size_t block_idx) {
        const MemoryBlock& block = blocks_[block_idx];
        free_blocks_by_size_[block.size].push_back(block_idx);
    }
    
    void remove_from_free_list(size_t block_idx) {
        const MemoryBlock& block = blocks_[block_idx];
        auto& indices = free_blocks_by_size_[block.size];
        
        auto it = std::find(indices.begin(), indices.end(), block_idx);
        if (it != indices.end()) {
            indices.erase(it);
            
            // Clean up empty size entries
            if (indices.empty()) {
                free_blocks_by_size_.erase(block.size);
            }
        }
    }
};

// CUDAMemoryPool implementation
CUDAMemoryPool::CUDAMemoryPool(size_t pool_size) 
    : pimpl_(std::make_unique<Impl>(pool_size)) {
}

CUDAMemoryPool::~CUDAMemoryPool() = default;

void* CUDAMemoryPool::allocate(size_t size, size_t alignment) {
    return pimpl_->allocate(size, alignment);
}

void CUDAMemoryPool::deallocate(void* ptr) {
    pimpl_->deallocate(ptr);
}

size_t CUDAMemoryPool::get_total_size() const {
    return pimpl_->pool_size_;
}

size_t CUDAMemoryPool::get_used_size() const {
    std::lock_guard<std::mutex> lock(pimpl_->pool_mutex_);
    return pimpl_->used_size_;
}

size_t CUDAMemoryPool::get_free_size() const {
    std::lock_guard<std::mutex> lock(pimpl_->pool_mutex_);
    return pimpl_->pool_size_ - pimpl_->used_size_;
}

} // namespace gpu
} // namespace ultra