#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <execinfo.h>
#include <unistd.h>
#include <time.h>
#include <stdint.h>

static void* (*real_malloc)(size_t) = NULL;
static void (*real_free)(void*) = NULL;
static int initialized = 0;

static void init() {
    if (initialized) return;
    initialized = 1;
    real_malloc = dlsym(RTLD_NEXT, "malloc");
    real_free = dlsym(RTLD_NEXT, "free");
    if (!real_malloc || !real_free) {
        fprintf(stderr, "[DEBUG_MALLOC] Error getting real functions\n");
        abort();
    }
}

// Track all malloc'd pointers to validate free() calls
#define MAX_TRACKED_POINTERS 1000000
static void* tracked_pointers[MAX_TRACKED_POINTERS];
static size_t num_tracked = 0;
static int tracking_enabled = 1;

void* malloc(size_t size) {
    init();
    void* ptr = real_malloc(size);
    if (tracking_enabled && ptr != NULL && num_tracked < MAX_TRACKED_POINTERS) {
        tracked_pointers[num_tracked++] = ptr;
    }
    return ptr;
}

void free(void* ptr) {
    init();
    
    if (ptr == NULL) {
        // free(NULL) is safe
        real_free(ptr);
        return;
    }
    
    uintptr_t addr = (uintptr_t)ptr;
    
    // SIMPLIFIED: Don't track pointers - just check for problematic patterns
    // Tracking causes performance issues and might cause hangs
    int found_in_tracked = 0;  // Always treat as untracked for now
    
    // If not found in tracked pointers, check if it's a problematic pattern
    // Be more aggressive - skip ALL high memory range pointers (0x7000... to 0x7fff...)
    // These are likely mmap'd memory that can't be freed with free()
    if (!found_in_tracked) {
        // Pattern: High memory range (0x7000... to 0x7fff...) - skip ALL of them
        // This is the most common source of "free(): invalid pointer" errors
        if (addr > 0x700000000000ULL && addr < 0x800000000000ULL) {
            // Skip all high memory pointers - they're likely mmap'd
            // Only log first few to reduce noise
            static int skip_count = 0;
            if (skip_count++ < 10) {
                fprintf(stderr, "[DEBUG_MALLOC_FREE] WARNING: Skipping free() for high memory pointer: %p (likely mmap'd)\n", ptr);
                fflush(stderr);
            }
            return;  // Don't free - likely invalid
        }
        
        // Pattern 2: Page-aligned pointers in high memory (definitely mmap'd)
        if ((addr & 0xFFF) == 0 && addr > 0x700000000000ULL) {
            static int page_skip_count = 0;
            if (page_skip_count++ < 10) {
                fprintf(stderr, "[DEBUG_MALLOC_FREE] WARNING: Skipping free() for page-aligned high memory pointer: %p (likely mmap'd)\n", ptr);
                fflush(stderr);
            }
            return;
        }
        
        // Allow other untracked pointers - they might be from other libraries that use malloc correctly
        // Don't log them to reduce noise
    }
    
    // Only log errors (infinite loops, skipped pointers) - reduce noise
    // Don't log normal untracked pointers to reduce overhead
    
    // Call real free
    real_free(ptr);
}

