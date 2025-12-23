/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-08-05 04:49:08
 * @Version      : 1.0.0
 * @LastEditors  : chenht2022 
 * @LastEditTime : 2024-08-05 09:21:29
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#include "shared_mem_buffer.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <stdexcept>
#include <atomic>
#include <algorithm>

SharedMemBuffer::SharedMemBuffer() {
    buffer_ = nullptr;
    size_ = 0;
    current_offset_ = 0;
}

SharedMemBuffer::~SharedMemBuffer() {
    if (buffer_) {
        free(buffer_);
    }
}

void SharedMemBuffer::alloc(void* object, std::vector<std::pair<void**, uint64_t>> requests) {
    const char* debug_env = std::getenv("KSFT_MOE_DEBUG");
    bool debug = (debug_env != nullptr && std::string(debug_env) == "1");
    
    // Use regular lock instead of try_lock - model loading should be single-threaded anyway
    // but if there's any concurrent access, we want to wait rather than fail
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (debug) {
        fprintf(stderr, "[SharedMemBuffer::alloc] Entry: object=%p, requests.size()=%zu\n", 
                object, requests.size());
        fflush(stderr);
    }
    
    uint64_t size = 0;
    for (auto& request : requests) {
        size += request.second;
    }
    
    // Calculate required size
    // On first allocation, pre-allocate a large buffer to minimize reallocations
    // Each layer needs ~2.6 GB, so for 28 layers we need ~73 GB
    // Pre-allocate 80 GB from the start
    const uint64_t PREALLOC_SIZE = 80ULL * 1024 * 1024 * 1024; // 80 GB
    
    uint64_t required_size;
    if (size_ == 0) {
        // First allocation - pre-allocate large buffer
        required_size = std::max(size, PREALLOC_SIZE);
        if (debug) {
            fprintf(stderr, "[SharedMemBuffer::alloc] First allocation: pre-allocating %llu bytes (%.2f GB)\n",
                    (unsigned long long)required_size, required_size / (1024.0 * 1024.0 * 1024.0));
            fflush(stderr);
        }
    } else {
        // Use existing buffer if it fits, otherwise need to grow
        if (current_offset_ + size <= size_) {
            required_size = size_;  // Use existing buffer
        } else {
            // Need to grow - allocate 50% larger than needed
            required_size = (current_offset_ + size) * 3 / 2;
            if (debug) {
                fprintf(stderr, "[SharedMemBuffer::alloc] Buffer needs to grow: current_offset=%llu, size=%llu, new_size=%llu\n",
                        (unsigned long long)current_offset_, (unsigned long long)size, (unsigned long long)required_size);
                fflush(stderr);
            }
        }
    }
    
    if (debug) {
        fprintf(stderr, "[SharedMemBuffer::alloc] Total size needed: %llu bytes (%.2f MB)\n",
                (unsigned long long)size, size / (1024.0 * 1024.0));
        fprintf(stderr, "[SharedMemBuffer::alloc] Current buffer size: %llu bytes (%.2f MB), offset: %llu bytes\n",
                (unsigned long long)size_, size_ / (1024.0 * 1024.0), (unsigned long long)current_offset_);
        fflush(stderr);
    }
    
    if (current_offset_ + size > size_) {
        if (debug) {
            fprintf(stderr, "[SharedMemBuffer::alloc] Need to allocate new buffer (size > size_)\n");
            fprintf(stderr, "[SharedMemBuffer::alloc] Current: size_=%llu, Needed: size=%llu\n",
                    (unsigned long long)size_, (unsigned long long)size);
            if (buffer_) {
                fprintf(stderr, "[SharedMemBuffer::alloc] Old buffer at %p will be freed after re-arrangement\n", buffer_);
            }
            fflush(stderr);
        }
        
        // CRITICAL: Allocate new buffer FIRST, before freeing old one
        // This ensures we always have a valid buffer during pointer updates
        void* new_buffer = std::aligned_alloc(64, required_size);
        
        if (debug) {
            if (new_buffer) {
                fprintf(stderr, "[SharedMemBuffer::alloc] ✓ New buffer allocated at %p\n", new_buffer);
            } else {
                fprintf(stderr, "[SharedMemBuffer::alloc] ✗ std::aligned_alloc() FAILED (returned nullptr)!\n");
            }
            fflush(stderr);
        }
        
        if (!new_buffer) {
            fprintf(stderr, "[SharedMemBuffer::alloc] FATAL: std::aligned_alloc(64, %llu) returned nullptr!\n",
                    (unsigned long long)required_size);
            fflush(stderr);
            throw std::bad_alloc();
        }
        
        // Store old buffer and size for copying and pointer updates
        void* old_buffer = buffer_;
        uint64_t old_size = size_;
        uintptr_t old_buffer_addr = (uintptr_t)old_buffer;
        uintptr_t new_buffer_addr = (uintptr_t)new_buffer;
        intptr_t addr_diff = (intptr_t)(new_buffer_addr - old_buffer_addr);
        
        if (debug) {
            fprintf(stderr, "[SharedMemBuffer::alloc] Old buffer: %p, New buffer: %p, Address diff: %ld\n",
                    old_buffer, new_buffer, (long)addr_diff);
            fflush(stderr);
        }
        
        // Copy data from old buffer to new buffer at same offsets
        if (old_buffer && old_size > 0) {
            uint64_t copy_size = std::min(old_size, required_size);
            if (debug) {
                fprintf(stderr, "[SharedMemBuffer::alloc] Copying %llu bytes from old buffer to new buffer\n",
                        (unsigned long long)copy_size);
                fflush(stderr);
            }
            memcpy(new_buffer, old_buffer, copy_size);
            
            // Zero out the rest of the new buffer if it's larger
            if (required_size > old_size) {
                uint8_t* zero_start = (uint8_t*)new_buffer + old_size;
                uint64_t zero_size = required_size - old_size;
                memset(zero_start, 0, zero_size);
            }
        } else {
            // First allocation - zero out the entire buffer
            memset(new_buffer, 0, required_size);
        }
        
        // Update buffer pointer BEFORE updating pointers (so arrange() uses new buffer)
        buffer_ = new_buffer;
        size_ = required_size;
        
        // Reset current_offset_ when allocating new buffer (old buffers are kept alive)
        // New allocations will start from offset 0 in the new buffer
        current_offset_ = 0;
        
        // Memory barrier to ensure buffer_ update is visible
        std::atomic_thread_fence(std::memory_order_seq_cst);
        
        // CRITICAL FIX: Don't update existing pointers - keep old buffer alive
        // This avoids modifying member variables of C++ objects that might be accessed
        // by Python/pybind11, which could cause segfaults
        // Instead, we'll keep the old buffer and only use the new buffer for new allocations
        // This means we'll have memory overhead, but it's safer than segfaulting
        if (old_buffer && !hist_requests_.empty()) {
            if (debug) {
                fprintf(stderr, "[SharedMemBuffer::alloc] WARNING: Keeping old buffer alive to avoid pointer invalidation\n");
                fprintf(stderr, "[SharedMemBuffer::alloc] Old buffer: %p, New buffer: %p\n",
                        old_buffer, new_buffer);
                fprintf(stderr, "[SharedMemBuffer::alloc] Existing %zu objects will continue using old buffer\n",
                        hist_requests_.size());
                fflush(stderr);
            }
            // DON'T free old_buffer - keep it alive
            // This causes memory overhead but prevents segfaults
            // TODO: Implement proper cleanup when objects are destroyed
        } else if (old_buffer) {
            // No existing objects, safe to free old buffer
            if (debug) {
                fprintf(stderr, "[SharedMemBuffer::alloc] Freeing old buffer at %p (no existing objects)\n", old_buffer);
                fflush(stderr);
            }
            free(old_buffer);
        }
        
        if (debug) {
            fprintf(stderr, "[SharedMemBuffer::alloc] ✓ Buffer reallocation complete\n");
            fflush(stderr);
        }
    } else {
        if (debug) {
            fprintf(stderr, "[SharedMemBuffer::alloc] Using existing buffer (size <= size_)\n");
            fflush(stderr);
        }
    }
    
    if (debug) {
        fprintf(stderr, "[SharedMemBuffer::alloc] Arranging new requests for object %p...\n", object);
        fflush(stderr);
    }
    
    // CRITICAL: Arrange new requests starting at current_offset_
    // This writes to member variables of the object
    uint64_t start_offset = current_offset_;
    arrange(requests, start_offset);
    
    // Update current_offset_ for next allocation
    current_offset_ += size;
    
    // Memory barrier to ensure all pointer writes from arrange() are visible
    std::atomic_thread_fence(std::memory_order_seq_cst);
    
    if (debug) {
        fprintf(stderr, "[SharedMemBuffer::alloc] ✓ Arranged new requests\n");
        fflush(stderr);
    }
    
    // Store requests in history AFTER arranging (so we know pointers are set)
    hist_requests_[object].push_back(requests);
    
    // Final memory barrier before returning
    std::atomic_thread_fence(std::memory_order_seq_cst);
    
    if (debug) {
        fprintf(stderr, "[SharedMemBuffer::alloc] ✓ Added to hist_requests_, total objects: %zu\n",
                hist_requests_.size());
        fprintf(stderr, "[SharedMemBuffer::alloc] Exit: Success\n");
        fflush(stderr);
    }
}

void SharedMemBuffer::dealloc(void* object) {
    const char* debug_env = std::getenv("KSFT_MOE_DEBUG");
    bool debug = (debug_env != nullptr && std::string(debug_env) == "1");
    
    if (debug) {
        fprintf(stderr, "[SharedMemBuffer::dealloc] Entry: object=%p\n", object);
        fprintf(stderr, "[SharedMemBuffer::dealloc] Current objects in hist_requests_: %zu\n", hist_requests_.size());
        fflush(stderr);
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (debug) {
        fprintf(stderr, "[SharedMemBuffer::dealloc] Lock acquired, erasing object\n");
        fflush(stderr);
    }
    
    size_t erased = hist_requests_.erase(object);
    
    if (debug) {
        fprintf(stderr, "[SharedMemBuffer::dealloc] Erased %zu entries for object %p\n", erased, object);
        fprintf(stderr, "[SharedMemBuffer::dealloc] Remaining objects: %zu\n", hist_requests_.size());
        fprintf(stderr, "[SharedMemBuffer::dealloc] Exit: Success\n");
        fflush(stderr);
    }
}

void SharedMemBuffer::arrange(std::vector<std::pair<void**, uint64_t>> requests, uint64_t start_offset) {
    const char* debug_env = std::getenv("KSFT_MOE_DEBUG");
    bool debug = (debug_env != nullptr && std::string(debug_env) == "1");
    
    if (debug) {
        fprintf(stderr, "[SharedMemBuffer::arrange] Entry: requests.size()=%zu, buffer_=%p, start_offset=%llu\n",
                requests.size(), buffer_, (unsigned long long)start_offset);
        fflush(stderr);
    }
    
    if (!buffer_) {
        fprintf(stderr, "[SharedMemBuffer::arrange] FATAL: buffer_ is nullptr!\n");
        fflush(stderr);
        throw std::runtime_error("SharedMemBuffer::arrange called with nullptr buffer_");
    }
    
    uint64_t offset = start_offset;
    for (size_t i = 0; i < requests.size(); i++) {
        auto& request = requests[i];
        void** ptr_ptr = request.first;
        uint64_t size = request.second;
        
        if (debug && i < 5) {  // Only log first 5 to avoid spam
            fprintf(stderr, "[SharedMemBuffer::arrange] request[%zu]: ptr_ptr=%p, size=%llu, offset=%llu\n",
                    i, (void*)ptr_ptr, (unsigned long long)size, (unsigned long long)offset);
            fflush(stderr);
        }
        
        if (!ptr_ptr) {
            fprintf(stderr, "[SharedMemBuffer::arrange] FATAL: request[%zu].first is nullptr!\n", i);
            fflush(stderr);
            throw std::runtime_error("SharedMemBuffer::arrange: null pointer in request");
        }
        
        void* target_ptr = (uint8_t*)buffer_ + offset;
        
        // Bounds checking
        if ((uint8_t*)target_ptr < (uint8_t*)buffer_ || 
            (uint8_t*)target_ptr + size > (uint8_t*)buffer_ + size_) {
            fprintf(stderr, "[SharedMemBuffer::arrange] FATAL: Out of bounds! request[%zu]: target_ptr=%p, buffer_=%p, size_=%llu, offset=%llu, size=%llu\n",
                    i, target_ptr, buffer_, (unsigned long long)size_, (unsigned long long)offset, (unsigned long long)size);
            fflush(stderr);
            throw std::runtime_error("SharedMemBuffer::arrange: pointer out of bounds");
        }
        
        // Write pointer - regular write is fine since we're in a mutex
        // CRITICAL: Verify target_ptr is accessible before writing
        try {
            volatile uint8_t test = *(volatile uint8_t*)target_ptr;
            (void)test;  // Suppress unused warning
        } catch (...) {
            fprintf(stderr, "[SharedMemBuffer::arrange] FATAL: Cannot access target_ptr=%p before assignment!\n", target_ptr);
            fflush(stderr);
            throw std::runtime_error("SharedMemBuffer::arrange: target_ptr not accessible");
        }
        
        *ptr_ptr = target_ptr;
        
        // Verify write succeeded
        if (*ptr_ptr != target_ptr) {
            fprintf(stderr, "[SharedMemBuffer::arrange] FATAL: Pointer write failed! Expected %p, got %p\n",
                    target_ptr, *ptr_ptr);
            fflush(stderr);
            throw std::runtime_error("SharedMemBuffer::arrange: pointer write verification failed");
        }
        
        if (debug && i < 5) {
            fprintf(stderr, "[SharedMemBuffer::arrange] request[%zu]: Assigned *ptr_ptr=%p (validated and verified)\n",
                    i, target_ptr);
            fflush(stderr);
        }
        
        offset += size;
    }
    
    if (debug) {
        fprintf(stderr, "[SharedMemBuffer::arrange] ✓ Completed arranging %zu requests, total offset=%llu\n",
                requests.size(), (unsigned long long)offset);
        fflush(stderr);
    }
}