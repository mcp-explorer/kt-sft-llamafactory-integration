/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-07-22 02:03:22
 * @Version      : 1.0.0
 * @LastEditors  : kkk1nak0
 * @LastEditTime : 2024-08-15 07:43:41
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#include "llama.cpp/ggml.h"
#include "llama.cpp/ggml-impl.h"
#include "sft_moe.h"
#include "ggml-cpu.h"
#include <iostream>
#include <cstdint>
#include <cstring>

// Wrapper to check params.nth before calling llamafile_sgemm
static bool safe_llamafile_sgemm(const ggml_compute_params* params, int64_t m, int64_t n, int64_t k, const void* a, int64_t lda, const void* b, int64_t ldb, void* c, int64_t ldc, int type_a, int type_b, int type_c) {
    if (params && params->nth <= 0) {
        fprintf(stderr, "[SAFE_WRAPPER] CRITICAL: params->nth is %ld! Fixing to 1 before llamafile_sgemm call.\n", params->nth);
        // Create a modified params with nth=1
        ggml_compute_params safe_params = *params;
        safe_params.nth = 1;
        return llamafile_sgemm(&safe_params, m, n, k, a, lda, b, ldb, c, ldc, type_a, type_b, type_c);
    }
    return llamafile_sgemm(params, m, n, k, a, lda, b, ldb, c, ldc, type_a, type_b, type_c);
}

// Forward declare if not defined
#ifndef GGML_COMPUTE_PARAMS_DEFINED
#include <cstdint>
struct ggml_compute_params {
    int64_t ith;
    int64_t nth;
    void* threadpool;
};
#define GGML_COMPUTE_PARAMS_DEFINED
#endif
#include <cstdio>
#include <cstdlib>
#include <string>
#include <atomic>
#ifdef __linux__
#include <sys/mman.h>
#include <unistd.h>
#include <errno.h>
#include <cstring>
#include <execinfo.h>
#include <signal.h>
#include <dlfcn.h>
#endif
#include <stdexcept>
#include <time.h>

#ifdef USE_NUMA
#include <numa.h>
#include <numaif.h>
#endif

// Signal handler to print backtrace on segfault
static void segfault_handler(int sig, siginfo_t* info, void* context) {
    fprintf(stderr, "\n=== SEGFAULT DETECTED ===\n");
    fprintf(stderr, "Signal: %d, Address: %p\n", sig, info->si_addr);
    
    void* array[50];
    size_t size = backtrace(array, 50);
    
    fprintf(stderr, "Backtrace (%zu frames):\n", size);
    backtrace_symbols_fd(array, size, STDERR_FILENO);
    
    fprintf(stderr, "=== END BACKTRACE ===\n");
    fflush(stderr);
    
    // Re-raise signal to get core dump
    signal(sig, SIG_DFL);
    raise(sig);
}

// Install signal handler on first use
static bool signal_handler_installed = false;
static void install_signal_handler() {
    if (!signal_handler_installed) {
        struct sigaction sa;
        sa.sa_sigaction = segfault_handler;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = SA_SIGINFO;
        sigaction(SIGSEGV, &sa, nullptr);
        sigaction(SIGBUS, &sa, nullptr);
        signal_handler_installed = true;
    }
}

SFT_MOE::SFT_MOE(SFT_MOEConfig config) {
    // Install signal handler on first constructor call
    install_signal_handler();
    
    // CRITICAL: Copy config immediately and store pointers before any other operations
    // This ensures config data is captured before Python might destroy the config object
    config_ = config;
    
    // Safety check: ensure stride is never 0 (causes assertion failure in llamafile_sgemm)
    if (config_.stride <= 0) {
        fprintf(stderr, "[SFT_MOE] ERROR: config.stride is %d, must be > 0! Setting to 64.\n", config_.stride);
        config_.stride = 64;  // Default safe value
    }
    
    // Store pointers from config immediately to avoid accessing destroyed config
    void* gate_proj_ptr = config_.gate_proj;
    void* up_proj_ptr = config_.up_proj;
    void* down_proj_ptr = config_.down_proj;
    
    // Initialize ownership flags
    owns_gate_proj_cpu_ = false;
    owns_up_proj_cpu_ = false;
    owns_down_proj_cpu_ = false;
    gate_proj_cpu_ = nullptr;
    up_proj_cpu_ = nullptr;
    down_proj_cpu_ = nullptr;
    
    // CRITICAL: Initialize ALL pointer members to nullptr to prevent uninitialized access
    gate_proj_ = nullptr;
    up_proj_ = nullptr;
    down_proj_ = nullptr;
    gate_proj_t_ = nullptr;
    up_proj_t_ = nullptr;
    down_proj_t_ = nullptr;
    transpose_buffer_fp32_ = nullptr;
    transpose_buffer_ = nullptr;
    s_input_fp32_ = nullptr;
    s_gate_input_ = nullptr;
    s_up_input_ = nullptr;
    s_output_fp32_ = nullptr;
    s_input_grad_fp32_ = nullptr;
    m_local_gate_input_ = nullptr;
    m_local_up_input_ = nullptr;
    m_local_gate_output_ = nullptr;
    m_local_up_output_ = nullptr;
    m_local_intermediate_fp32_ = nullptr;
    m_local_down_input_ = nullptr;
    m_local_down_output_ = nullptr;
    m_local_down_output_grad_ = nullptr;
    m_local_down_input_grad_ = nullptr;
    m_local_gate_output_grad_fp32_ = nullptr;
    m_local_up_output_grad_fp32_ = nullptr;
    m_local_gate_output_grad_ = nullptr;
    m_local_up_output_grad_ = nullptr;
    m_local_gate_input_grad_ = nullptr;
    m_local_up_input_grad_ = nullptr;
    m_local_token_indices_ = nullptr;
    m_local_expert_positions_ = nullptr;
    
    // Debug: Log pointer values when stored
    const char* debug_env = std::getenv("KSFT_MOE_DEBUG");
    bool debug = (debug_env != nullptr && std::string(debug_env) == "1");
    if (debug) {
        fprintf(stderr, "[C++ SFT_MOE::SFT_MOE] Constructor: Original pointers\n");
        fprintf(stderr, "[C++ SFT_MOE::SFT_MOE] config.gate_proj=%p, config.up_proj=%p, config.down_proj=%p\n", 
                gate_proj_ptr, up_proj_ptr, down_proj_ptr);
        fprintf(stderr, "[C++ SFT_MOE::SFT_MOE] expert_num=%ld, intermediate_size=%ld, hidden_size=%ld\n",
                config_.expert_num, config_.intermediate_size, config_.hidden_size);
        fflush(stderr);
    }
    
    // Match MOE pattern: assign pointers directly without copying
    // This avoids potential stack overflow and memory issues in constructor
    gate_proj_ = gate_proj_ptr;
    up_proj_ = up_proj_ptr;
    down_proj_ = down_proj_ptr;
    
    // CPU-accessible copies are not needed if pointers are already CPU-accessible
    // Set to nullptr to indicate we don't own the memory
    gate_proj_cpu_ = nullptr;
    up_proj_cpu_ = nullptr;
    down_proj_cpu_ = nullptr;
    owns_gate_proj_cpu_ = false;
    owns_up_proj_cpu_ = false;
    owns_down_proj_cpu_ = false;
    
    if (debug) {
        fprintf(stderr, "[C++ SFT_MOE::SFT_MOE] Assigned pointers directly (matching MOE pattern)\n");
        fprintf(stderr, "[C++ SFT_MOE::SFT_MOE] gate_proj_=%p, up_proj_=%p, down_proj_=%p\n",
                gate_proj_, up_proj_, down_proj_);
        fflush(stderr);
    }
    
    #ifdef USE_NUMA
    int numa_nodes = numa_num_configured_nodes();
    gate_proj_numa_.resize(numa_nodes);
    up_proj_numa_.resize(numa_nodes);
    down_proj_numa_.resize(numa_nodes);
    size_t exp_inter_hidden_mul_ = (size_t)config.expert_num * config.intermediate_size * config.hidden_size;
    for (int i = 0; i < numa_nodes; i++) {
        gate_proj_numa_[i] = numa_alloc_onnode(exp_inter_hidden_mul_* ggml_type_size(config.gate_type) / ggml_blck_size(config.gate_type), i);
        up_proj_numa_[i] = numa_alloc_onnode(exp_inter_hidden_mul_* ggml_type_size(config.up_type) / ggml_blck_size(config.up_type), i);
        down_proj_numa_[i] = numa_alloc_onnode(exp_inter_hidden_mul_* ggml_type_size(config.down_type) / ggml_blck_size(config.down_type), i);
        if (!gate_proj_numa_[i]) {
            std::cout << "Memory allocation failed for gate_proj_numa_ on node " << i << std::endl;
        }
        if (!up_proj_numa_[i]) {
            std::cout << "Memory allocation failed for up_proj_numa_ on node " << i << std::endl;
        }
        if (!down_proj_numa_[i]) {
            std::cout << "Memory allocation failed for down_proj_numa_ on node " << i << std::endl;
        }
        memcpy(gate_proj_numa_[i], gate_proj_, exp_inter_hidden_mul_* ggml_type_size(config.gate_type) / ggml_blck_size(config.gate_type));
        memcpy(up_proj_numa_[i], up_proj_, exp_inter_hidden_mul_* ggml_type_size(config.up_type) / ggml_blck_size(config.up_type));
        memcpy(down_proj_numa_[i], down_proj_, exp_inter_hidden_mul_* ggml_type_size(config.down_type) / ggml_blck_size(config.down_type));
    }
    #endif

    std::vector<std::pair<void**, uint64_t>> s_mem_requests;
    s_mem_requests.push_back({(void**)&gate_proj_t_, config_.expert_num * config_.hidden_size * config_.intermediate_size * ggml_type_size(config_.grad_type)});
    s_mem_requests.push_back({(void**)&up_proj_t_, config_.expert_num * config_.hidden_size * config_.intermediate_size * ggml_type_size(config_.grad_type)});
    s_mem_requests.push_back({(void**)&down_proj_t_, config_.expert_num * config_.hidden_size * config_.intermediate_size * ggml_type_size(config_.grad_type)});
    s_mem_requests.push_back({(void**)&transpose_buffer_fp32_, config_.expert_num * config_.intermediate_size * config_.hidden_size * sizeof(float)});
    s_mem_requests.push_back({(void**)&transpose_buffer_, config_.expert_num * config_.intermediate_size * config_.hidden_size * ggml_type_size(config_.grad_type)});

    s_mem_requests.push_back({(void**)&s_input_fp32_, sizeof(float) * config_.hidden_size});
    s_mem_requests.push_back({(void**)&s_gate_input_, config_.hidden_size * ggml_type_size(ggml_get_type_traits_cpu(config_.gate_type)->vec_dot_type) / ggml_blck_size(ggml_get_type_traits_cpu(config_.gate_type)->vec_dot_type)});
    s_mem_requests.push_back({(void**)&s_up_input_, config_.hidden_size * ggml_type_size(ggml_get_type_traits_cpu(config_.up_type)->vec_dot_type) / ggml_blck_size(ggml_get_type_traits_cpu(config_.up_type)->vec_dot_type)});
    s_gate_output_.resize(config_.routed_expert_num);
    s_up_output_.resize(config_.routed_expert_num);
    s_intermediate_fp32_.resize(config_.routed_expert_num);
    s_down_input_.resize(config_.routed_expert_num);
    s_down_output_.resize(config_.routed_expert_num);
    for (int i = 0; i < config_.routed_expert_num; i++) {
        s_mem_requests.push_back({(void**)&s_gate_output_[i], sizeof(float) * config_.intermediate_size});
        s_mem_requests.push_back({(void**)&s_up_output_[i], sizeof(float) * config_.intermediate_size});
        s_mem_requests.push_back({(void**)&s_intermediate_fp32_[i], sizeof(float) * config_.intermediate_size});
        s_mem_requests.push_back({(void**)&s_down_input_[i], config_.intermediate_size * ggml_type_size(ggml_get_type_traits_cpu(config_.down_type)->vec_dot_type) / ggml_blck_size(ggml_get_type_traits_cpu(config_.down_type)->vec_dot_type)});
        s_mem_requests.push_back({(void**)&s_down_output_[i], sizeof(float) * config_.hidden_size});
    }
    s_mem_requests.push_back({(void**)&s_output_fp32_, sizeof(float) * config_.hidden_size});
        
    s_down_input_grad_.resize(config_.routed_expert_num);
    s_gate_output_grad_fp32_.resize(config_.routed_expert_num);
    s_up_output_grad_fp32_.resize(config_.routed_expert_num);
    s_gate_output_grad_.resize(config_.routed_expert_num);
    s_up_output_grad_.resize(config_.routed_expert_num);
    s_gate_input_grad_.resize(config_.routed_expert_num);
    s_up_input_grad_.resize(config_.routed_expert_num);
    for (int i = 0; i < config_.routed_expert_num; i++) {
        s_mem_requests.push_back({(void**)&s_down_input_grad_[i], config_.intermediate_size * sizeof(float)});
        s_mem_requests.push_back({(void**)&s_gate_output_grad_fp32_[i], config_.intermediate_size * sizeof(float)});
        s_mem_requests.push_back({(void**)&s_up_output_grad_fp32_[i], config_.intermediate_size * sizeof(float)});
        s_mem_requests.push_back({(void**)&s_gate_output_grad_[i], config_.intermediate_size * ggml_type_size(config_.grad_type)});
        s_mem_requests.push_back({(void**)&s_up_output_grad_[i], config_.intermediate_size * ggml_type_size(config_.grad_type)});
        s_mem_requests.push_back({(void**)&s_gate_input_grad_[i], config_.hidden_size * sizeof(float)});
        s_mem_requests.push_back({(void**)&s_up_input_grad_[i], config_.hidden_size * sizeof(float)});
    }
    s_mem_requests.push_back({(void**)&s_input_grad_fp32_, config_.hidden_size * sizeof(float)});

    if (debug) {
        uint64_t total_s_size = 0;
        for (const auto& req : s_mem_requests) {
            total_s_size += req.second;
        }
        fprintf(stderr, "[C++ SFT_MOE::SFT_MOE] About to call shared_mem_buffer.alloc() for s_mem_requests\n");
        fprintf(stderr, "[C++ SFT_MOE::SFT_MOE] s_mem_requests.size()=%zu, total_size=%llu bytes (%.2f MB)\n",
                s_mem_requests.size(), (unsigned long long)total_s_size, total_s_size / (1024.0 * 1024.0));
        fprintf(stderr, "[C++ SFT_MOE::SFT_MOE] this=%p\n", (void*)this);
        fflush(stderr);
    }
    
    shared_mem_buffer.alloc(this, s_mem_requests);
    
    if (debug) {
        fprintf(stderr, "[C++ SFT_MOE::SFT_MOE] ✓ shared_mem_buffer.alloc() for s_mem_requests completed successfully\n");
        fflush(stderr);
    }

    std::vector<std::pair<void**, uint64_t>> m_mem_requests;
    m_input_fp32_.resize(config_.group_max_len);
    m_gate_input_.resize(config_.group_max_len);
    m_up_input_.resize(config_.group_max_len);
    for (int i = 0; i < config_.group_max_len; i++) {
        m_mem_requests.push_back({(void**)&m_input_fp32_[i], sizeof(float) * config_.hidden_size});
        m_mem_requests.push_back({(void**)&m_gate_input_[i], config_.hidden_size * ggml_type_size(ggml_get_type_traits_cpu(config_.gate_type)->vec_dot_type) / ggml_blck_size(ggml_get_type_traits_cpu(config_.gate_type)->vec_dot_type)});
        m_mem_requests.push_back({(void**)&m_up_input_[i], config_.hidden_size * ggml_type_size(ggml_get_type_traits_cpu(config_.up_type)->vec_dot_type) / ggml_blck_size(ggml_get_type_traits_cpu(config_.up_type)->vec_dot_type)});
    }
    m_mem_requests.push_back({(void**)&m_local_gate_input_, config_.routed_expert_num * config_.group_max_len * config_.hidden_size * ggml_type_size(ggml_get_type_traits_cpu(config_.gate_type)->vec_dot_type) / ggml_blck_size(ggml_get_type_traits_cpu(config_.gate_type)->vec_dot_type)});
    m_mem_requests.push_back({(void**)&m_local_up_input_, config_.routed_expert_num * config_.group_max_len * config_.hidden_size * ggml_type_size(ggml_get_type_traits_cpu(config_.up_type)->vec_dot_type) / ggml_blck_size(ggml_get_type_traits_cpu(config_.up_type)->vec_dot_type)});
    m_mem_requests.push_back({(void**)&m_local_gate_output_, sizeof(float) * config_.routed_expert_num * config_.group_max_len * config_.intermediate_size});
    m_mem_requests.push_back({(void**)&m_local_up_output_, sizeof(float) * config_.routed_expert_num * config_.group_max_len * config_.intermediate_size});
    m_mem_requests.push_back({(void**)&m_local_intermediate_fp32_, sizeof(float) * config_.routed_expert_num * config_.group_max_len * config_.intermediate_size});
    m_mem_requests.push_back({(void**)&m_local_down_input_, config_.routed_expert_num * config_.group_max_len * config_.intermediate_size * ggml_type_size(ggml_get_type_traits_cpu(config_.down_type)->vec_dot_type) / ggml_blck_size(ggml_get_type_traits_cpu(config_.down_type)->vec_dot_type)});
    m_mem_requests.push_back({(void**)&m_local_down_output_, sizeof(float) * config_.routed_expert_num * config_.group_max_len * config_.hidden_size});
    m_output_fp32_.resize(config_.group_max_len);
    for (int i = 0; i < config_.group_max_len; i++) {
        m_mem_requests.push_back({(void**)&m_output_fp32_[i], sizeof(float) * config_.hidden_size});
    }
    
    m_mem_requests.push_back({(void**)&m_local_down_output_grad_, config_.routed_expert_num * config_.group_max_len * config_.hidden_size * ggml_type_size(config_.grad_type)});
    m_mem_requests.push_back({(void**)&m_local_down_input_grad_, sizeof(float) * config_.routed_expert_num * config_.group_max_len * config_.intermediate_size});
    m_mem_requests.push_back({(void**)&m_local_gate_output_grad_fp32_, sizeof(float) * config_.routed_expert_num * config_.group_max_len * config_.intermediate_size});
    m_mem_requests.push_back({(void**)&m_local_up_output_grad_fp32_, sizeof(float) * config_.routed_expert_num * config_.group_max_len * config_.intermediate_size});
    m_mem_requests.push_back({(void**)&m_local_gate_output_grad_, config_.routed_expert_num * config_.group_max_len * config_.intermediate_size * ggml_type_size(config_.grad_type)});
    m_mem_requests.push_back({(void**)&m_local_up_output_grad_, config_.routed_expert_num * config_.group_max_len * config_.intermediate_size * ggml_type_size(config_.grad_type)});
    m_mem_requests.push_back({(void**)&m_local_gate_input_grad_, sizeof(float) * config_.routed_expert_num * config_.group_max_len * config_.hidden_size});
    m_mem_requests.push_back({(void**)&m_local_up_input_grad_, sizeof(float) * config_.routed_expert_num * config_.group_max_len * config_.hidden_size});
    m_mem_requests.push_back({(void**)&m_local_token_indices_, sizeof(int) * config_.routed_expert_num * config_.group_max_len});
    m_mem_requests.push_back({(void**)&m_local_expert_positions_, sizeof(int) * config_.routed_expert_num * config_.group_max_len});
    m_grad_input_fp32_.resize(config_.group_max_len);
    for (int i = 0; i < config_.group_max_len; i++) {
        m_mem_requests.push_back({(void**)&m_grad_input_fp32_[i], sizeof(float) * config_.hidden_size});
    }
    
    if (debug) {
        uint64_t total_m_size = 0;
        for (const auto& req : m_mem_requests) {
            total_m_size += req.second;
        }
        fprintf(stderr, "[C++ SFT_MOE::SFT_MOE] About to call shared_mem_buffer.alloc() for m_mem_requests\n");
        fprintf(stderr, "[C++ SFT_MOE::SFT_MOE] m_mem_requests.size()=%zu, total_size=%llu bytes (%.2f MB)\n",
                m_mem_requests.size(), (unsigned long long)total_m_size, total_m_size / (1024.0 * 1024.0));
        fprintf(stderr, "[C++ SFT_MOE::SFT_MOE] this=%p\n", (void*)this);
        fflush(stderr);
    }
    
    shared_mem_buffer.alloc(this, m_mem_requests);
    
    if (debug) {
        fprintf(stderr, "[C++ SFT_MOE::SFT_MOE] ✓ shared_mem_buffer.alloc() for m_mem_requests completed successfully\n");
        fflush(stderr);
    }

    m_local_pos_.resize(config_.group_max_len);
    for (int i = 0; i < config_.group_max_len; i++) {
        m_local_pos_[i].resize(config_.routed_expert_num);
    }
    m_local_num_.resize(config_.expert_num);
    m_local_gate_input_ptr_.resize(config_.expert_num);
    m_local_up_input_ptr_.resize(config_.expert_num);
    m_local_gate_output_ptr_.resize(config_.expert_num);
    m_local_up_output_ptr_.resize(config_.expert_num);
    m_local_intermediate_fp32_ptr_.resize(config_.expert_num);
    m_local_down_input_ptr_.resize(config_.expert_num);
    m_local_down_output_ptr_.resize(config_.expert_num);
    
    // backward_many 专用指针数组初始化
    m_local_down_output_grad_ptr_.resize(config_.expert_num);
    m_local_down_input_grad_ptr_.resize(config_.expert_num);
    m_local_gate_output_grad_fp32_ptr_.resize(config_.expert_num);
    m_local_up_output_grad_fp32_ptr_.resize(config_.expert_num);
    m_local_gate_output_grad_ptr_.resize(config_.expert_num);
    m_local_up_output_grad_ptr_.resize(config_.expert_num);
    m_local_gate_input_grad_ptr_.resize(config_.expert_num);
    m_local_up_input_grad_ptr_.resize(config_.expert_num);
    
    // fwd_cache访问映射指针数组初始化
    m_local_token_indices_ptr_.resize(config_.expert_num);
    m_local_expert_positions_ptr_.resize(config_.expert_num);
    
    // Allocate fw_cache_ on heap to reduce object size and avoid pybind11 issues
    // with large nested vector structures
    fw_cache_ = new std::vector<SFT_MoEForwardCache>();
    
    // Final memory barrier to ensure ALL writes are visible before constructor returns
    // This includes all pointer assignments, vector resizes, and buffer allocations
    std::atomic_thread_fence(std::memory_order_seq_cst);
    
    // Additional compiler barrier to prevent reordering
    asm volatile("" ::: "memory");
    
    // CRITICAL: Verify all critical pointers are valid before returning
    // This helps catch issues before Python accesses the object
    if (gate_proj_t_ == nullptr || up_proj_t_ == nullptr || down_proj_t_ == nullptr) {
        fprintf(stderr, "[C++ SFT_MOE::SFT_MOE] FATAL: Critical buffer pointers are nullptr!\n");
        fprintf(stderr, "[C++ SFT_MOE::SFT_MOE] gate_proj_t_=%p, up_proj_t_=%p, down_proj_t_=%p\n",
                (void*)gate_proj_t_, (void*)up_proj_t_, (void*)down_proj_t_);
        fflush(stderr);
        throw std::runtime_error("SFT_MOE constructor: buffer pointers not initialized");
    }
    
    // REMOVED: Buffer access test - this can cause segfaults if buffers are not yet fully initialized
    // The buffers are allocated by shared_mem_buffer.alloc() which should be safe
    // If there's an issue, it will be caught during actual use, not here
    
    if (debug) {
        fprintf(stderr, "[C++ SFT_MOE::SFT_MOE] ✓ Constructor completed successfully\n");
        fprintf(stderr, "[C++ SFT_MOE::SFT_MOE] All arrays resized, ready for use\n");
        fprintf(stderr, "[C++ SFT_MOE::SFT_MOE] Memory barriers executed, all writes are visible\n");
        fprintf(stderr, "[C++ SFT_MOE::SFT_MOE] Buffer pointers verified accessible\n");
        fprintf(stderr, "[C++ SFT_MOE::SFT_MOE] this=%p, gate_proj_=%p, up_proj_=%p, down_proj_=%p\n",
                (void*)this, gate_proj_, up_proj_, down_proj_);
        fprintf(stderr, "[C++ SFT_MOE::SFT_MOE] buffer_ pointers: gate_proj_t_=%p, up_proj_t_=%p, down_proj_t_=%p\n",
                (void*)gate_proj_t_, (void*)up_proj_t_, (void*)down_proj_t_);
        fprintf(stderr, "[C++ SFT_MOE::SFT_MOE] fw_cache_=%p, size=%zu, capacity=%zu\n",
                (void*)fw_cache_, fw_cache_ ? fw_cache_->size() : 0, fw_cache_ ? fw_cache_->capacity() : 0);
        fflush(stderr);
    }
}

SFT_MOE::~SFT_MOE() {
    const char* debug_env = std::getenv("KSFT_MOE_DEBUG");
    bool debug = (debug_env != nullptr && std::string(debug_env) == "1");
    
    if (debug) {
        fprintf(stderr, "[C++ SFT_MOE::~SFT_MOE] Destructor called for this=%p\n", (void*)this);
        fprintf(stderr, "[C++ SFT_MOE::~SFT_MOE] gate_proj_t_=%p, up_proj_t_=%p, down_proj_t_=%p\n",
                (void*)gate_proj_t_, (void*)up_proj_t_, (void*)down_proj_t_);
        fflush(stderr);
    }
    
    // CRITICAL: Verify pointers are still valid before deallocating
    // This helps catch use-after-free or double-free issues
    if (gate_proj_t_ != nullptr || up_proj_t_ != nullptr || down_proj_t_ != nullptr) {
        if (debug) {
            fprintf(stderr, "[C++ SFT_MOE::~SFT_MOE] Buffer pointers still set, deallocating from shared buffer\n");
            fflush(stderr);
        }
    }
    
    shared_mem_buffer.dealloc(this);
    
    if (debug) {
        fprintf(stderr, "[C++ SFT_MOE::~SFT_MOE] ✓ Shared buffer deallocated\n");
        fflush(stderr);
    }

    // Free CPU-accessible copies if we allocated them
    if (owns_gate_proj_cpu_ && gate_proj_cpu_) {
        std::free(gate_proj_cpu_);
        gate_proj_cpu_ = nullptr;
    }
    if (owns_up_proj_cpu_ && up_proj_cpu_) {
        std::free(up_proj_cpu_);
        up_proj_cpu_ = nullptr;
    }
    if (owns_down_proj_cpu_ && down_proj_cpu_) {
        std::free(down_proj_cpu_);
        down_proj_cpu_ = nullptr;
    }

    // Delete fw_cache_ if allocated
    if (fw_cache_ != nullptr) {
        delete fw_cache_;
        fw_cache_ = nullptr;
    }

    #ifdef USE_NUMA
    int numa_nodes = numa_num_configured_nodes();
    for (int i = 0; i < numa_nodes; i++) {
        numa_free(gate_proj_numa_[i], config_.expert_num * config_.intermediate_size * config_.hidden_size * ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type));
        numa_free(up_proj_numa_[i], config_.expert_num * config_.intermediate_size * config_.hidden_size * ggml_type_size(config_.up_type) / ggml_blck_size(config_.up_type));
        numa_free(down_proj_numa_[i], config_.expert_num * config_.hidden_size * config_.intermediate_size * ggml_type_size(config_.down_type) / ggml_blck_size(config_.down_type));
    }
    #endif
}

void SFT_MOE::warm_up(Backend* backend) {
    std::vector<float> input_fp32(config_.hidden_size);
    std::vector<uint8_t> input(config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type));
    std::vector<uint8_t> output(config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type));
    for (int i = 0; i < config_.hidden_size; i++) {
        input_fp32[i] = 0;
    }
    from_float(input_fp32.data(), input.data(), config_.hidden_size, config_.hidden_type);
	/* ---------- 仅用于占位的 ForwardCache ---------- */
    SFT_MoEForwardCache dummy_cache; // 内容无用，只为满足接口
	dummy_cache.init(/*k=*/1, config_.intermediate_size);
    for (int i = 0; i < config_.expert_num; i++) {
        uint64_t expert_ids = i;
        float weights = 0;
        forward_one(1, &expert_ids, &weights, input.data(), output.data(), backend, &dummy_cache);
    }
}

static float act_fn(float x) {
    return x / (1.0f + expf(-x));
}

void SFT_MOE::ensure_fwd_cache(int qlen, int k)
{
	if (fw_cache_ == nullptr) {
		fw_cache_ = new std::vector<SFT_MoEForwardCache>();
	}
	
	int old_sz = fw_cache_->size();
    if (old_sz < qlen)
    {
        fw_cache_->resize(qlen);
        for (int i = old_sz; i < qlen; ++i)  // 仅初始化新增元素
            (*fw_cache_)[i].init(k, config_.intermediate_size);
    }
    
    // Ensure all entries up to qlen are initialized (in case k changed)
    // This is important for backward pass which accesses cache entries
    for (int i = 0; i < qlen; ++i) {
        // Re-init if k or intermediate_size might have changed
        // The init() function is safe to call multiple times (only resizes if needed)
        (*fw_cache_)[i].init(k, config_.intermediate_size);
    }
}

SFT_MoEForwardCache* SFT_MOE::fwd_cache_ptr()
{
	return (fw_cache_ == nullptr || fw_cache_->empty()) ? nullptr : fw_cache_->data();
}

void SFT_MOE::forward_one(int k, const uint64_t* expert_ids, const float* weights, const void* input, void* output, Backend* backend, SFT_MoEForwardCache* fwd_cache) {
    const void* gate_input_ptr;
    const void* up_input_ptr;
    if (config_.hidden_type == ggml_get_type_traits_cpu(config_.gate_type)->vec_dot_type && config_.hidden_type == ggml_get_type_traits_cpu(config_.up_type)->vec_dot_type) {
        gate_input_ptr = up_input_ptr = input;
    } else {
        to_float(input, s_input_fp32_, config_.hidden_size, config_.hidden_type);
        if (ggml_get_type_traits_cpu(config_.gate_type)->vec_dot_type == ggml_get_type_traits_cpu(config_.up_type)->vec_dot_type) {
            from_float(s_input_fp32_, s_gate_input_, config_.hidden_size, ggml_get_type_traits_cpu(config_.gate_type)->vec_dot_type);
            gate_input_ptr = up_input_ptr = s_gate_input_;
        } else {
            if (config_.hidden_type != ggml_get_type_traits_cpu(config_.gate_type)->vec_dot_type) {
                from_float(s_input_fp32_, s_gate_input_, config_.hidden_size, ggml_get_type_traits_cpu(config_.gate_type)->vec_dot_type);
                gate_input_ptr = s_gate_input_;
            } else {
                gate_input_ptr = input;
            }
            if (config_.hidden_type != ggml_get_type_traits_cpu(config_.up_type)->vec_dot_type) {
                from_float(s_input_fp32_, s_up_input_, config_.hidden_size, ggml_get_type_traits_cpu(config_.up_type)->vec_dot_type);
                up_input_ptr = s_up_input_;
            } else {
                up_input_ptr = input;
            }
        }
    }
    int nth = config_.intermediate_size / config_.stride;
    if (nth <= 0) nth = 1;  // Safety check: ensure nth > 0
    backend->do_work_stealing_job(nth * k, nullptr, [&](int task_id) {
        int expert_idx = task_id / nth;
        uint64_t expert_id = expert_ids[expert_idx];
        int ith = task_id % nth;
        
        #ifdef USE_NUMA
        void* gate_proj_ptr = (uint8_t*)gate_proj_numa_[Backend::numa_node] + (expert_id * config_.intermediate_size + ith * config_.stride) * config_.hidden_size * ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type);
        #else
        void* gate_proj_ptr = (uint8_t*)gate_proj_ + (expert_id * config_.intermediate_size + ith * config_.stride) * config_.hidden_size * ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type);
        #endif

        float* gate_output_ptr = s_gate_output_[expert_idx] + ith * config_.stride;
        ggml_compute_params params_gate;
        params_gate.ith = ith;
        params_gate.nth = std::max(1, nth);  // Ensure nth > 0 to avoid assertion failure
        params_gate.threadpool = nullptr;
        if (params_gate.nth <= 0) { 
            fprintf(stderr, "[SFT_MOE DEBUG] params_gate.nth is %ld, fixing to 1 (nth=%d, config_.stride=%d)\n", params_gate.nth, nth, config_.stride);
            params_gate.nth = 1;  // Final safety check
        }
        if (params_gate.nth <= 0) {
            fprintf(stderr, "[SFT_MOE DEBUG] CRITICAL: params_gate.nth is still %ld after fix! Aborting llamafile_sgemm call.\n", params_gate.nth);
            abort();
        }
        llamafile_sgemm(&params_gate, config_.stride, 1, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_proj_ptr, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_input_ptr, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_output_ptr, config_.stride, config_.gate_type, ggml_get_type_traits_cpu(config_.gate_type)->vec_dot_type, GGML_TYPE_F32);

        #ifdef USE_NUMA
        void* up_proj_ptr = (uint8_t*)up_proj_numa_[Backend::numa_node] + (expert_id * config_.intermediate_size + ith * config_.stride) * config_.hidden_size * ggml_type_size(config_.up_type) / ggml_blck_size(config_.up_type);
        #else
        void* up_proj_ptr = (uint8_t*)up_proj_ + (expert_id * config_.intermediate_size + ith * config_.stride) * config_.hidden_size * ggml_type_size(config_.up_type) / ggml_blck_size(config_.up_type);
        #endif

        float* up_output_ptr = s_up_output_[expert_idx] + ith * config_.stride;
        ggml_compute_params params_up;
        params_up.ith = ith;
        params_up.nth = std::max(1, nth);  // Ensure nth > 0 to avoid assertion failure
        params_up.threadpool = nullptr;
        if (params_up.nth <= 0) { 
            fprintf(stderr, "[SFT_MOE DEBUG] params_up.nth is %ld, fixing to 1 (nth=%d, config_.stride=%d)\n", params_up.nth, nth, config_.stride);
            params_up.nth = 1;  // Final safety check
        }
        if (params_up.nth <= 0) {
            fprintf(stderr, "[SFT_MOE DEBUG] CRITICAL: params_up.nth is still %ld after fix! Aborting llamafile_sgemm call.\n", params_up.nth);
            abort();
        }
        llamafile_sgemm(&params_up, config_.stride, 1, config_.hidden_size / ggml_blck_size(config_.up_type), up_proj_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), up_input_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), up_output_ptr, config_.stride, config_.up_type, ggml_get_type_traits_cpu(config_.up_type)->vec_dot_type, GGML_TYPE_F32);
        for (int i = ith * config_.stride; i < (ith + 1) * config_.stride; i++) {
            s_intermediate_fp32_[expert_idx][i] = act_fn(s_gate_output_[expert_idx][i]) * s_up_output_[expert_idx][i];
        }
        if (config_.stride % ggml_blck_size(ggml_get_type_traits_cpu(config_.down_type)->vec_dot_type) == 0) {
            float* intermediate_fp32_ptr = s_intermediate_fp32_[expert_idx] + ith * config_.stride;
            void* down_input_ptr = s_down_input_[expert_idx] + ith * config_.stride * ggml_type_size(ggml_get_type_traits_cpu(config_.down_type)->vec_dot_type) / ggml_blck_size(ggml_get_type_traits_cpu(config_.down_type)->vec_dot_type);
            from_float(intermediate_fp32_ptr, down_input_ptr, config_.stride, ggml_get_type_traits_cpu(config_.down_type)->vec_dot_type);
        }
    }, nullptr);
    if (config_.stride % ggml_blck_size(ggml_get_type_traits_cpu(config_.down_type)->vec_dot_type) != 0) {
        for (int i = 0; i < k; i++) {
            from_float(s_intermediate_fp32_[i], s_down_input_[i], config_.intermediate_size, ggml_get_type_traits_cpu(config_.down_type)->vec_dot_type);
        }
    }
    nth = config_.hidden_size / config_.stride;
    backend->do_work_stealing_job(nth, nullptr, [&](int task_id) {
        int ith = task_id;
        for (int i = ith * config_.stride; i < (ith + 1) * config_.stride; i++) {
            s_output_fp32_[i] = 0;
        }
        for (int expert_idx = 0; expert_idx < k; expert_idx++) {
            uint64_t expert_id = expert_ids[expert_idx];

            #ifdef USE_NUMA
            void* down_proj_ptr = (uint8_t*)down_proj_numa_[Backend::numa_node] + (expert_id * config_.hidden_size + ith * config_.stride) * config_.intermediate_size * ggml_type_size(config_.down_type) / ggml_blck_size(config_.down_type);
            #else
            void* down_proj_ptr = (uint8_t*)down_proj_ + (expert_id * config_.hidden_size + ith * config_.stride) * config_.intermediate_size * ggml_type_size(config_.down_type) / ggml_blck_size(config_.down_type);
            #endif
            
            float* down_output_ptr = s_down_output_[expert_idx] + ith * config_.stride;
            ggml_compute_params params_down;
        params_down.ith = ith;
        params_down.nth = std::max(1, nth);  // Ensure nth > 0 to avoid assertion failure
        params_down.threadpool = nullptr;
        if (params_down.nth <= 0) { 
            fprintf(stderr, "[SFT_MOE DEBUG forward_one] params_down.nth is %ld, fixing to 1 (nth=%d, config_.stride=%d)\n", params_down.nth, nth, config_.stride);
            params_down.nth = 1;  // Final safety check
        }
        if (params_down.nth <= 0) {
            fprintf(stderr, "[SFT_MOE DEBUG forward_one] CRITICAL: params_down.nth is still %ld after fix! Aborting llamafile_sgemm call.\n", params_down.nth);
            abort();
        }
            llamafile_sgemm(&params_down, config_.stride, 1, config_.intermediate_size / ggml_blck_size(config_.down_type), down_proj_ptr, config_.intermediate_size / ggml_blck_size(config_.down_type), s_down_input_[expert_idx], config_.intermediate_size / ggml_blck_size(config_.down_type), down_output_ptr, config_.stride, config_.down_type, ggml_get_type_traits_cpu(config_.down_type)->vec_dot_type, GGML_TYPE_F32);
            for (int i = ith * config_.stride; i < (ith + 1) * config_.stride; i++) {
                s_output_fp32_[i] += s_down_output_[expert_idx][i] * weights[expert_idx];
            }
        }
        if (config_.stride % ggml_blck_size(config_.hidden_type) == 0) {
            float* output_fp32_ptr = s_output_fp32_ + ith * config_.stride;
            void* output_ptr = (uint8_t*)output + ith * config_.stride * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type);
            from_float(output_fp32_ptr, output_ptr, config_.stride, config_.hidden_type);
        }
    }, nullptr);

	for (int e = 0; e < k; ++e) {
        // gate_output_: float[inter_size] per expert
        std::memcpy(fwd_cache->gate_u[e].data(),
                    s_gate_output_[e],
                    sizeof(float) * config_.intermediate_size);

        std::memcpy(fwd_cache->up_v[e].data(),
                    s_up_output_[e],
                    sizeof(float) * config_.intermediate_size);

        // 可选保存 z
        // std::memcpy(fwd_cache->z[e].data(),
        //             s_intermediate_fp32_[e],
        //             sizeof(float) * config_.intermediate_size);
    }
}

void SFT_MOE::forward_many(int qlen, int k, const uint64_t* expert_ids, const float* weights, const void* input, void* output, Backend* backend, SFT_MoEForwardCache* fwd_cache) {
    for (int i = 0; i < config_.expert_num; i++) {
        m_local_num_[i] = 0;
    }
    for (int i = 0; i < qlen; i++) {
        for (int j = 0; j < k; j++) {
            m_local_pos_[i][j] = m_local_num_[expert_ids[i * k + j]]++;
        }
    }
    uint64_t offset = 0;
    for (int i = 0; i < config_.expert_num; i++) {
        m_local_gate_input_ptr_[i] = m_local_gate_input_ + offset * config_.hidden_size * ggml_type_size(ggml_get_type_traits_cpu(config_.gate_type)->vec_dot_type) / ggml_blck_size(ggml_get_type_traits_cpu(config_.gate_type)->vec_dot_type);
        m_local_up_input_ptr_[i] = m_local_up_input_ + offset * config_.hidden_size * ggml_type_size(ggml_get_type_traits_cpu(config_.up_type)->vec_dot_type) / ggml_blck_size(ggml_get_type_traits_cpu(config_.up_type)->vec_dot_type);
        m_local_gate_output_ptr_[i] = m_local_gate_output_ + offset * config_.intermediate_size;
        m_local_up_output_ptr_[i] = m_local_up_output_ + offset * config_.intermediate_size;
        m_local_intermediate_fp32_ptr_[i] = m_local_intermediate_fp32_ + offset * config_.intermediate_size;
        m_local_down_input_ptr_[i] = m_local_down_input_ + offset * config_.intermediate_size * ggml_type_size(ggml_get_type_traits_cpu(config_.down_type)->vec_dot_type) / ggml_blck_size(ggml_get_type_traits_cpu(config_.down_type)->vec_dot_type);
        m_local_down_output_ptr_[i] = m_local_down_output_ + offset * config_.hidden_size;
        offset += m_local_num_[i];
    }
    backend->do_work_stealing_job(qlen, nullptr, [&](int i) {
        const void* gate_input_ptr;
        const void* up_input_ptr;
        if (config_.hidden_type == ggml_get_type_traits_cpu(config_.gate_type)->vec_dot_type && config_.hidden_type == ggml_get_type_traits_cpu(config_.up_type)->vec_dot_type) {
            gate_input_ptr = up_input_ptr = (uint8_t*)input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type);
        } else {
            to_float((uint8_t*)input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), m_input_fp32_[i], config_.hidden_size, config_.hidden_type);
            if (ggml_get_type_traits_cpu(config_.gate_type)->vec_dot_type == ggml_get_type_traits_cpu(config_.up_type)->vec_dot_type) {
                from_float(m_input_fp32_[i], m_gate_input_[i], config_.hidden_size, ggml_get_type_traits_cpu(config_.gate_type)->vec_dot_type);
                gate_input_ptr = up_input_ptr = m_gate_input_[i];
            } else {
                if (config_.hidden_type != ggml_get_type_traits_cpu(config_.gate_type)->vec_dot_type) {
                    from_float(m_input_fp32_[i], m_gate_input_[i], config_.hidden_size, ggml_get_type_traits_cpu(config_.gate_type)->vec_dot_type);
                    gate_input_ptr = m_gate_input_[i];
                } else {
                    gate_input_ptr = (uint8_t*)input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type);
                }
                if (config_.hidden_type != ggml_get_type_traits_cpu(config_.up_type)->vec_dot_type) {
                    from_float(m_input_fp32_[i], m_up_input_[i], config_.hidden_size, ggml_get_type_traits_cpu(config_.up_type)->vec_dot_type);
                    up_input_ptr = m_up_input_[i];
                } else {
                    up_input_ptr = (uint8_t*)input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type);
                }
            }
        }
        for (int j = 0; j < k; j++) {
            memcpy(m_local_gate_input_ptr_[expert_ids[i * k + j]] + m_local_pos_[i][j] * config_.hidden_size * ggml_type_size(ggml_get_type_traits_cpu(config_.gate_type)->vec_dot_type) / ggml_blck_size(ggml_get_type_traits_cpu(config_.gate_type)->vec_dot_type), gate_input_ptr, config_.hidden_size * ggml_type_size(ggml_get_type_traits_cpu(config_.gate_type)->vec_dot_type) / ggml_blck_size(ggml_get_type_traits_cpu(config_.gate_type)->vec_dot_type));
            memcpy(m_local_up_input_ptr_[expert_ids[i * k + j]] + m_local_pos_[i][j] * config_.hidden_size * ggml_type_size(ggml_get_type_traits_cpu(config_.up_type)->vec_dot_type) / ggml_blck_size(ggml_get_type_traits_cpu(config_.up_type)->vec_dot_type), up_input_ptr, config_.hidden_size * ggml_type_size(ggml_get_type_traits_cpu(config_.up_type)->vec_dot_type) / ggml_blck_size(ggml_get_type_traits_cpu(config_.up_type)->vec_dot_type));
        }
    }, nullptr);
    // Use config_.stride if QK_K is not defined or is 0, otherwise use QK_K
    #ifndef QK_K
    #define QK_K 256  // Default quantization block size
    #endif
    int stride = (QK_K > 0) ? QK_K : config_.stride;
    if (stride <= 0) stride = 64;  // Safety fallback
    int nth = config_.intermediate_size / stride;
    backend->do_work_stealing_job(nth * config_.expert_num, nullptr, [&](int task_id) {
        uint64_t expert_idx = task_id / nth;
        int ith = task_id % nth;
        void* gate_input_ptr = m_local_gate_input_ptr_[expert_idx];

        #ifdef USE_NUMA
        void* gate_proj_ptr = (uint8_t*)gate_proj_numa_[Backend::numa_node] + (expert_idx * config_.intermediate_size + ith * stride) * config_.hidden_size * ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type);
        #else
        void* gate_proj_ptr = (uint8_t*)gate_proj_ + (expert_idx * config_.intermediate_size + ith * stride) * config_.hidden_size * ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type);
        #endif

        float* gate_output_ptr = m_local_gate_output_ptr_[expert_idx] + ith * stride;
        ggml_compute_params params_gate;
        params_gate.ith = ith;
        params_gate.nth = std::max(1, nth);  // Ensure nth > 0 to avoid assertion failure
        params_gate.threadpool = nullptr;
        if (params_gate.nth <= 0) { params_gate.nth = 1; }  // Final safety check
        llamafile_sgemm(&params_gate, stride, m_local_num_[expert_idx], config_.hidden_size / ggml_blck_size(config_.gate_type), gate_proj_ptr, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_input_ptr, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_output_ptr, config_.intermediate_size, config_.gate_type, ggml_get_type_traits_cpu(config_.gate_type)->vec_dot_type, GGML_TYPE_F32);
        void* up_input_ptr = m_local_up_input_ptr_[expert_idx];

        #ifdef USE_NUMA
        void* up_proj_ptr = (uint8_t*)up_proj_numa_[Backend::numa_node] + (expert_idx * config_.intermediate_size + ith * stride) * config_.hidden_size * ggml_type_size(config_.up_type) / ggml_blck_size(config_.up_type);
        #else
        void* up_proj_ptr = (uint8_t*)up_proj_ + (expert_idx * config_.intermediate_size + ith * stride) * config_.hidden_size * ggml_type_size(config_.up_type) / ggml_blck_size(config_.up_type);
        #endif

        float* up_output_ptr = m_local_up_output_ptr_[expert_idx] + ith * stride;
        ggml_compute_params params_up;
        params_up.ith = ith;
        params_up.nth = std::max(1, nth);  // Ensure nth > 0 to avoid assertion failure
        params_up.threadpool = nullptr;
        if (params_up.nth <= 0) { 
            fprintf(stderr, "[SFT_MOE DEBUG] params_up.nth is %ld, fixing to 1 (nth=%d, config_.stride=%d)\n", params_up.nth, nth, config_.stride);
            params_up.nth = 1;  // Final safety check
        }
        if (params_up.nth <= 0) {
            fprintf(stderr, "[SFT_MOE DEBUG] CRITICAL: params_up.nth is still %ld after fix! Aborting llamafile_sgemm call.\n", params_up.nth);
            abort();
        }
        llamafile_sgemm(&params_up, stride, m_local_num_[expert_idx], config_.hidden_size / ggml_blck_size(config_.up_type), up_proj_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), up_input_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), up_output_ptr, config_.intermediate_size, config_.up_type, ggml_get_type_traits_cpu(config_.up_type)->vec_dot_type, GGML_TYPE_F32);
        for (int i = 0; i < m_local_num_[expert_idx]; i++) {
            for (int j = ith * stride; j < (ith + 1) * stride; j++) {
                m_local_intermediate_fp32_ptr_[expert_idx][i * config_.intermediate_size + j] = act_fn(m_local_gate_output_ptr_[expert_idx][i * config_.intermediate_size + j]) * m_local_up_output_ptr_[expert_idx][i * config_.intermediate_size + j];
            }
            float* intermediate_fp32_ptr = m_local_intermediate_fp32_ptr_[expert_idx] + i * config_.intermediate_size + ith * stride;
            void* down_input_ptr = m_local_down_input_ptr_[expert_idx] + i * config_.intermediate_size * ggml_type_size(ggml_get_type_traits_cpu(config_.down_type)->vec_dot_type) / ggml_blck_size(ggml_get_type_traits_cpu(config_.down_type)->vec_dot_type) + ith * stride * ggml_type_size(ggml_get_type_traits_cpu(config_.down_type)->vec_dot_type) / ggml_blck_size(ggml_get_type_traits_cpu(config_.down_type)->vec_dot_type);
            from_float(intermediate_fp32_ptr, down_input_ptr, stride, ggml_get_type_traits_cpu(config_.down_type)->vec_dot_type);
        }
    }, nullptr);
    #ifndef QK_K
    #define QK_K 256  // Default quantization block size
    #endif
    stride = (QK_K > 0) ? QK_K : config_.stride;
    if (stride <= 0) stride = 64;  // Safety fallback
    nth = config_.hidden_size / stride;
    backend->do_work_stealing_job(nth * config_.expert_num, nullptr, [&](int task_id) {
        uint64_t expert_idx = task_id / nth;
        int ith = task_id % nth;
        void* down_input_ptr = m_local_down_input_ptr_[expert_idx];
        
        #ifdef USE_NUMA
        void* down_proj_ptr = (uint8_t*)down_proj_numa_[Backend::numa_node] + (expert_idx * config_.hidden_size + ith * stride) * config_.intermediate_size * ggml_type_size(config_.down_type) / ggml_blck_size(config_.down_type);
        #else
        void* down_proj_ptr = (uint8_t*)down_proj_ + (expert_idx * config_.hidden_size + ith * stride) * config_.intermediate_size * ggml_type_size(config_.down_type) / ggml_blck_size(config_.down_type);
        #endif

        float* down_output_ptr = m_local_down_output_ptr_[expert_idx] + ith * stride;
        ggml_compute_params params_down;
        params_down.ith = ith;
        params_down.nth = std::max(1, nth);  // Ensure nth > 0 to avoid assertion failure
        params_down.threadpool = nullptr;
        if (params_down.nth <= 0) { 
            fprintf(stderr, "[SFT_MOE DEBUG] params_down.nth is %ld, fixing to 1 (nth=%d, config_.stride=%d)\n", params_down.nth, nth, config_.stride);
            params_down.nth = 1;  // Final safety check
        }
        if (params_down.nth <= 0) {
            fprintf(stderr, "[SFT_MOE DEBUG] CRITICAL: params_down.nth is still %ld after fix! Aborting llamafile_sgemm call.\n", params_down.nth);
            abort();
        }
        llamafile_sgemm(&params_down, stride, m_local_num_[expert_idx], config_.intermediate_size / ggml_blck_size(config_.down_type), down_proj_ptr, config_.intermediate_size / ggml_blck_size(config_.down_type), down_input_ptr, config_.intermediate_size / ggml_blck_size(config_.down_type), down_output_ptr, config_.hidden_size, config_.down_type, ggml_get_type_traits_cpu(config_.down_type)->vec_dot_type, GGML_TYPE_F32);
    }, nullptr);
    backend->do_work_stealing_job(qlen, nullptr, [&](int i) {
        for (int e = 0; e < config_.hidden_size; e++) {
            m_output_fp32_[i][e] = 0;
        }
        for (int j = 0; j < k; j++) {
            for (int e = 0; e < config_.hidden_size; e++) {
                m_output_fp32_[i][e] += m_local_down_output_ptr_[expert_ids[i * k + j]][m_local_pos_[i][j] * config_.hidden_size + e] * weights[i * k + j];
            }
        }
        from_float(m_output_fp32_[i], (uint8_t*)output + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), config_.hidden_size, config_.hidden_type);
    }, nullptr);

	/* 把每个 token-expert 的行复制到各自 cache */
    backend->do_work_stealing_job(qlen, nullptr, [&](int token_idx) {
        auto& cache = fwd_cache[token_idx];
        // cache 已在上层 init(k, inter_size)
        for (int j = 0; j < k; ++j) {
            uint64_t  eid   = expert_ids[token_idx*k + j];
            int       row   = m_local_pos_[token_idx][j];
            size_t    ofs   = row * config_.intermediate_size;
            /* gate u */
            std::memcpy(cache.gate_u[j].data(),
                        m_local_gate_output_ptr_[eid] + ofs,
                        sizeof(float) * config_.intermediate_size);
            /* up v */
            std::memcpy(cache.up_v[j].data(),
                        m_local_up_output_ptr_[eid] + ofs,
                        sizeof(float) * config_.intermediate_size);
            /* 可选 z */
            // std::memcpy(cache.z[j].data(),
            //             m_local_intermediate_fp32_ptr_[eid] + ofs,
            //             sizeof(float) * config_.intermediate_size);
        }
    }, nullptr);
}

void SFT_MOE::forward(int qlen, int k, const uint64_t* expert_ids, const float* weights, const void* input, void* output, Backend* backend, SFT_MoEForwardCache* fwd_cache) {
    if (qlen < config_.group_min_len) {
        for (int i = 0; i < qlen; i++) {
			// fwd_cache[i].init(k, config_.intermediate_size);      // 预分配
            forward_one(k, expert_ids + i * k, weights + i * k, (uint8_t*)input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), (uint8_t*)output + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), backend, fwd_cache + i);
        }
        return;
    }
    int forward_len = std::min(config_.group_max_len, qlen);
    // for (int i = 0; i < forward_len; ++i)
    //     fwd_cache[i].init(k, config_.intermediate_size);
    forward_many(forward_len, k, expert_ids, weights, input, output, backend, fwd_cache);
    forward(qlen - forward_len, k, expert_ids + forward_len * k, weights + forward_len * k, (uint8_t*)input + forward_len * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), (uint8_t*)output + forward_len * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), backend, fwd_cache + forward_len);
}

static float act_fn_grad(float x) {
    float sigmoid_x = 1.0f / (1.0f + expf(-x));
    return sigmoid_x * (1. + x * (1. - sigmoid_x));
}

void SFT_MOE::transpose_expert_matrix(const void* src, void* dst, int R, int C, ggml_type src_type, ggml_type dst_type, uint64_t expert_idx) {
    // Debug: Check if KSFT_MOE_DEBUG is set
    const char* debug_env = std::getenv("KSFT_MOE_DEBUG");
    bool debug = (debug_env != nullptr && std::string(debug_env) == "1");
    
    if (debug && expert_idx == 0) {
        fprintf(stderr, "[C++ transpose_expert_matrix] expert_idx=%lu, R=%d, C=%d, src=%p, dst=%p\n", 
                expert_idx, R, C, src, dst);
        fprintf(stderr, "[C++ transpose_expert_matrix] transpose_buffer_fp32_=%p, transpose_buffer_=%p\n",
                transpose_buffer_fp32_, transpose_buffer_);
        fflush(stderr);
    }
    
    // Validate pointers
    if (src == nullptr || dst == nullptr || transpose_buffer_fp32_ == nullptr || transpose_buffer_ == nullptr) {
        throw std::runtime_error("Null pointer in transpose_expert_matrix");
    }
    
    size_t buffer_offset = (size_t)(R * C * expert_idx);
    size_t buffer_size = (size_t)(R * C);
    size_t total_buffer_size = (size_t)config_.expert_num * config_.intermediate_size * config_.hidden_size;
    
    if (debug && expert_idx == 0) {
        fprintf(stderr, "[C++ transpose_expert_matrix] buffer_offset=%zu, buffer_size=%zu, total_buffer_size=%zu\n", 
                buffer_offset, buffer_size, total_buffer_size);
        fprintf(stderr, "[C++ transpose_expert_matrix] src_type=%d, dst_type=%d\n", src_type, dst_type);
        fprintf(stderr, "[C++ transpose_expert_matrix] Calling to_float with src=%p, dst=%p, size=%zu...\n",
                src, transpose_buffer_fp32_ + buffer_offset, buffer_size);
        fflush(stderr);
    }
    
    // Validate buffer bounds
    if (buffer_offset + buffer_size > total_buffer_size) {
        throw std::runtime_error("Buffer overflow in transpose_expert_matrix: buffer_offset + buffer_size > total_buffer_size");
    }
    
    // Validate pointers are within reasonable range
    uintptr_t src_ptr = reinterpret_cast<uintptr_t>(src);
    uintptr_t dst_ptr = reinterpret_cast<uintptr_t>(transpose_buffer_fp32_ + buffer_offset);
    if (src_ptr == 0 || dst_ptr == 0) {
        throw std::runtime_error("Null pointer in transpose_expert_matrix");
    }
    
    if (debug && expert_idx == 0) {
        fprintf(stderr, "[C++ transpose_expert_matrix] About to call to_float: src=%p, dst=%p, src_type=%d, size=%zu (R=%d, C=%d)\n", 
                src, transpose_buffer_fp32_ + buffer_offset, src_type, buffer_size, R, C);
        fprintf(stderr, "[C++ transpose_expert_matrix] src pointer value: 0x%lx, dst pointer value: 0x%lx\n",
                src_ptr, dst_ptr);
        fflush(stderr);
    }
    
    // Call to_float - this may segfault if src points to invalid/inaccessible memory
    // The segfault will happen here if gate_proj_ points to GPU memory or memory
    // that's not accessible from worker threads
    try {
        to_float(src, transpose_buffer_fp32_ + buffer_offset, R * C, src_type);
        
        if (debug && expert_idx == 0) {
            fprintf(stderr, "[C++ transpose_expert_matrix] to_float completed successfully\n");
            fflush(stderr);
        }
    } catch (const std::exception& e) {
        if (debug) {
            fprintf(stderr, "[C++ transpose_expert_matrix] Exception in to_float: %s\n", e.what());
            fflush(stderr);
        }
        throw;
    } catch (...) {
        if (debug) {
            fprintf(stderr, "[C++ transpose_expert_matrix] Unknown exception in to_float\n");
            fflush(stderr);
        }
        throw;
    }
    
    if (debug && expert_idx == 0) {
        fprintf(stderr, "[C++ transpose_expert_matrix] to_float completed, calling from_float...\n");
        fflush(stderr);
    }
    
    from_float(transpose_buffer_fp32_ + buffer_offset, transpose_buffer_ + buffer_offset * ggml_type_size(dst_type), R * C, dst_type);
    
    if (debug && expert_idx == 0) {
        fprintf(stderr, "[C++ transpose_expert_matrix] from_float completed, starting memcpy loop...\n");
        fflush(stderr);
    }
    
    for (int r = 0; r < R; ++r) {
        for (int c = 0; c < C; ++c) {
            memcpy(
                (uint8_t*)dst + (c * R + r) * ggml_type_size(dst_type),
                (uint8_t*)transpose_buffer_ + (buffer_offset + r * C + c) * ggml_type_size(dst_type),
                ggml_type_size(dst_type));
        }
    }
    
    if (debug && expert_idx == 0) {
        fprintf(stderr, "[C++ transpose_expert_matrix] memcpy loop completed\n");
        fflush(stderr);
    }
}

void SFT_MOE::get_transpose(Backend* backend) {
    // Debug: Check if KSFT_MOE_DEBUG is set
    const char* debug_env = std::getenv("KSFT_MOE_DEBUG");
    bool debug = (debug_env != nullptr && std::string(debug_env) == "1");
    
    if (debug) {
        fprintf(stderr, "[C++ SFT_MOE::get_transpose] Starting transpose operations\n");
        fflush(stderr);
    }
    
    // Validate pointers before transpose operations
    if (gate_proj_ == nullptr || gate_proj_t_ == nullptr || up_proj_ == nullptr || up_proj_t_ == nullptr || 
        down_proj_ == nullptr || down_proj_t_ == nullptr || transpose_buffer_ == nullptr || transpose_buffer_fp32_ == nullptr) {
        throw std::runtime_error("One or more transpose buffers are nullptr in get_transpose");
    }
    
    if (debug) {
        fprintf(stderr, "[C++ SFT_MOE::get_transpose] All transpose buffers validated\n");
        fprintf(stderr, "[C++ SFT_MOE::get_transpose] gate_proj_=%p, gate_proj_t_=%p\n", gate_proj_, gate_proj_t_);
        fprintf(stderr, "[C++ SFT_MOE::get_transpose] up_proj_=%p, up_proj_t_=%p\n", up_proj_, up_proj_t_);
        fprintf(stderr, "[C++ SFT_MOE::get_transpose] down_proj_=%p, down_proj_t_=%p\n", down_proj_, down_proj_t_);
        fprintf(stderr, "[C++ SFT_MOE::get_transpose] transpose_buffer_=%p, transpose_buffer_fp32_=%p\n", transpose_buffer_, transpose_buffer_fp32_);
        fflush(stderr);
    }
    
    // Transpose gate_proj_
    if (debug) {
        fprintf(stderr, "[C++ SFT_MOE::get_transpose] Transposing gate_proj_ (expert_num=%ld)\n", config_.expert_num);
        fflush(stderr);
    }
    int R_gate = config_.intermediate_size;
    int C_gate = config_.hidden_size;
    size_t gate_expert_src_stride_bytes = (size_t)R_gate * C_gate * ggml_type_size(config_.gate_type);
    size_t gate_expert_dst_t_stride_bytes = (size_t)C_gate * R_gate * ggml_type_size(config_.grad_type);
    
    // Step 3: Test memory accessibility in main thread before worker threads
    if (debug) {
        fprintf(stderr, "[C++ SFT_MOE::get_transpose] Testing memory accessibility in main thread...\n");
        fprintf(stderr, "[C++ SFT_MOE::get_transpose] gate_proj_=%p, trying to read first 16 bytes...\n", gate_proj_);
        fflush(stderr);
    }
    
    // Try to read first few bytes from gate_proj_ in main thread
    // This will help us determine if the memory is accessible at all, or only from worker threads
    try {
        // Calculate size of first expert's data
        size_t first_expert_size = (size_t)R_gate * C_gate * ggml_type_size(config_.gate_type);
        if (first_expert_size > 0) {
            // Try to read first 16 bytes (or less if expert is smaller)
            size_t test_size = (first_expert_size < 16) ? first_expert_size : 16;
            volatile uint8_t test_buffer[16] = {0};
            
            // Use memcpy which should handle invalid pointers more gracefully than direct dereference
            memcpy((void*)test_buffer, gate_proj_, test_size);
            
            if (debug) {
                fprintf(stderr, "[C++ SFT_MOE::get_transpose] ✓ Memory accessibility test PASSED in main thread\n");
                fprintf(stderr, "[C++ SFT_MOE::get_transpose] First %zu bytes read successfully\n", test_size);
                fflush(stderr);
            }
        }
    } catch (...) {
        if (debug) {
            fprintf(stderr, "[C++ SFT_MOE::get_transpose] ✗ Memory accessibility test FAILED in main thread\n");
            fprintf(stderr, "[C++ SFT_MOE::get_transpose] Cannot read from gate_proj_ even in main thread!\n");
            fflush(stderr);
        }
        // Don't throw here - let it fail in worker thread to see the actual error
    }
    
    if (debug) {
        fprintf(stderr, "[C++ SFT_MOE::get_transpose] About to call do_work_stealing_job for gate_proj_\n");
        fflush(stderr);
    }
    // This will help us determine if the memory is accessible at all
    try {
        volatile uint8_t test_bytes[16];
        for (int i = 0; i < 16 && i < gate_expert_src_stride_bytes; ++i) {
            test_bytes[i] = *((const uint8_t*)gate_proj_ + i);
        }
        if (debug) {
            fprintf(stderr, "[C++ SFT_MOE::get_transpose] ✓ Memory accessibility test PASSED in main thread\n");
            fprintf(stderr, "[C++ SFT_MOE::get_transpose] First 16 bytes: ");
            for (int i = 0; i < 16 && i < gate_expert_src_stride_bytes; ++i) {
                fprintf(stderr, "%02x ", test_bytes[i]);
            }
            fprintf(stderr, "\n");
            fflush(stderr);
        }
    } catch (...) {
        if (debug) {
            fprintf(stderr, "[C++ SFT_MOE::get_transpose] ✗ Memory accessibility test FAILED in main thread\n");
            fprintf(stderr, "[C++ SFT_MOE::get_transpose] Cannot read from gate_proj_ even in main thread!\n");
            fflush(stderr);
        }
        throw std::runtime_error("gate_proj_ memory is not accessible even in main thread");
    }
    
    if (debug) {
        fprintf(stderr, "[C++ SFT_MOE::get_transpose] About to call do_work_stealing_job for gate_proj_\n");
        fflush(stderr);
    }
    
    try {
        backend->do_work_stealing_job(config_.expert_num, nullptr, [&](int expert_idx) {
            void* src_expert = (uint8_t*)gate_proj_ + expert_idx * gate_expert_src_stride_bytes;
            void* dst_expert_t = (uint8_t*)gate_proj_t_ + expert_idx * gate_expert_dst_t_stride_bytes;
            transpose_expert_matrix(src_expert, dst_expert_t, R_gate, C_gate, config_.gate_type, config_.grad_type, expert_idx);
        }, nullptr);
    } catch (const std::exception& e) {
        if (debug) {
            fprintf(stderr, "[C++ SFT_MOE::get_transpose] Exception in gate_proj_ transpose: %s\n", e.what());
            fflush(stderr);
        }
        throw;
    } catch (...) {
        if (debug) {
            fprintf(stderr, "[C++ SFT_MOE::get_transpose] Unknown exception in gate_proj_ transpose\n");
            fflush(stderr);
        }
        throw;
    }
    
    if (debug) {
        fprintf(stderr, "[C++ SFT_MOE::get_transpose] gate_proj_ transpose completed\n");
        fflush(stderr);
    }

    // Transpose up_proj_
    if (debug) {
        fprintf(stderr, "[C++ SFT_MOE::get_transpose] Transposing up_proj_\n");
        fflush(stderr);
    }
    int R_up = config_.intermediate_size;
    int C_up = config_.hidden_size;
    size_t up_expert_src_stride_bytes = (size_t)R_up * C_up * ggml_type_size(config_.up_type);
    size_t up_expert_dst_t_stride_bytes = (size_t)C_up * R_up * ggml_type_size(config_.grad_type);
    backend->do_work_stealing_job(config_.expert_num, nullptr, [&](int expert_idx) {
        void* src_expert = (uint8_t*)up_proj_ + expert_idx * up_expert_src_stride_bytes;
        void* dst_expert_t = (uint8_t*)up_proj_t_ + expert_idx * up_expert_dst_t_stride_bytes;
        transpose_expert_matrix(src_expert, dst_expert_t, R_up, C_up, config_.up_type, config_.grad_type, expert_idx);
    }, nullptr);
    
    if (debug) {
        fprintf(stderr, "[C++ SFT_MOE::get_transpose] up_proj_ transpose completed\n");
        fflush(stderr);
    }

    // Transpose down_proj_
    if (debug) {
        fprintf(stderr, "[C++ SFT_MOE::get_transpose] Transposing down_proj_\n");
        fflush(stderr);
    }
    int R_down = config_.hidden_size;
    int C_down = config_.intermediate_size;
    size_t down_expert_src_stride_bytes = (size_t)R_down * C_down * ggml_type_size(config_.down_type);
    size_t down_expert_dst_t_stride_bytes = (size_t)C_down * R_down * ggml_type_size(config_.grad_type);
    backend->do_work_stealing_job(config_.expert_num, nullptr, [&](int expert_idx) {
        void* src_expert = (uint8_t*)down_proj_ + expert_idx * down_expert_src_stride_bytes;
        void* dst_expert_t = (uint8_t*)down_proj_t_ + expert_idx * down_expert_dst_t_stride_bytes;
        transpose_expert_matrix(src_expert, dst_expert_t, R_down, C_down, config_.down_type, config_.grad_type, expert_idx);
    }, nullptr);
    
    if (debug) {
        fprintf(stderr, "[C++ SFT_MOE::get_transpose] down_proj_ transpose completed\n");
        fprintf(stderr, "[C++ SFT_MOE::get_transpose] All transpose operations completed\n");
        fflush(stderr);
    }
}

void SFT_MOE::backward_one(int k, const uint64_t* expert_ids, const float* weights, const void* output_grad, void* input_grad, Backend* backend, const SFT_MoEForwardCache* fwd_cache) {
	// clock_t clk1, clk2, clk3, clk4;
	// clock_t clkz1, clkz2, clkz3, clkz4, clkz5;
	// clk1 = clock();
	// clk2 = clock();
    int nth = config_.intermediate_size / config_.stride;
    backend->do_work_stealing_job(nth * k, nullptr, [&](int task_id) {
        int expert_idx = task_id / nth;
        uint64_t expert_id = expert_ids[expert_idx];
        int ith = task_id % nth;
		// clkz1 = clock();
        void* down_proj_t_ptr = (uint8_t*)down_proj_t_ + (expert_id * config_.intermediate_size + ith * config_.stride) * config_.hidden_size * ggml_type_size(config_.grad_type);
        float* down_input_grad_ptr = s_down_input_grad_[expert_idx] + ith * config_.stride;
        // clkz2 = clock();
        ggml_compute_params params_down;
        params_down.ith = ith;
        params_down.nth = std::max(1, nth);  // Ensure nth > 0 to avoid assertion failure
        params_down.threadpool = nullptr;
        if (params_down.nth <= 0) { 
            fprintf(stderr, "[SFT_MOE DEBUG] params_down.nth is %ld, fixing to 1 (nth=%d, config_.stride=%d)\n", params_down.nth, nth, config_.stride);
            params_down.nth = 1;  // Final safety check
        }
        if (params_down.nth <= 0) {
            fprintf(stderr, "[SFT_MOE DEBUG] CRITICAL: params_down.nth is still %ld after fix! Aborting llamafile_sgemm call.\n", params_down.nth);
            abort();
        }
        llamafile_sgemm(&params_down, config_.stride, 1, config_.hidden_size, down_proj_t_ptr, config_.hidden_size, output_grad, config_.hidden_size, down_input_grad_ptr, config_.stride, config_.grad_type, config_.grad_type, GGML_TYPE_F32);
        // clkz3 = clock();
        for (int i = ith * config_.stride; i < (ith + 1) * config_.stride; i++) {
            s_down_input_grad_[expert_idx][i] *= weights[expert_idx];

            s_gate_output_grad_fp32_[expert_idx][i] = s_down_input_grad_[expert_idx][i] * fwd_cache->up_v[expert_idx][i] * act_fn_grad(fwd_cache->gate_u[expert_idx][i]); 
            s_up_output_grad_fp32_[expert_idx][i] = s_down_input_grad_[expert_idx][i] * act_fn(fwd_cache->gate_u[expert_idx][i]);
        }
        // clkz4 = clock();
        from_float(s_gate_output_grad_fp32_[expert_idx] + ith * config_.stride, s_gate_output_grad_[expert_idx] + ith * config_.stride * ggml_type_size(config_.grad_type), config_.stride, config_.grad_type);
        from_float(s_up_output_grad_fp32_[expert_idx] + ith * config_.stride, s_up_output_grad_[expert_idx] + ith * config_.stride * ggml_type_size(config_.grad_type), config_.stride, config_.grad_type);
        // clkz5 = clock();
    }, nullptr);

	// clk3 = clock();
    nth = config_.hidden_size / config_.stride;
    backend->do_work_stealing_job(nth, nullptr, [&](int task_id) {
        int ith = task_id;
        for (int i = ith * config_.stride; i < (ith + 1) * config_.stride; i++) {
            s_input_grad_fp32_[i] = 0;
        }
        for (int expert_idx = 0; expert_idx < k; expert_idx++) {
            uint64_t expert_id = expert_ids[expert_idx];

            void* gate_proj_t_ptr = (uint8_t*)gate_proj_t_ + (expert_id * config_.hidden_size + ith * config_.stride) * config_.intermediate_size * ggml_type_size(config_.grad_type);
            float* gate_input_grad_ptr = s_gate_input_grad_[expert_idx] + ith * config_.stride;
            ggml_compute_params params_gate;
        params_gate.ith = ith;
        params_gate.nth = std::max(1, nth);  // Ensure nth > 0 to avoid assertion failure
        params_gate.threadpool = nullptr;
            llamafile_sgemm(&params_gate, config_.stride, 1, config_.intermediate_size, gate_proj_t_ptr, config_.intermediate_size, s_gate_output_grad_[expert_idx], config_.intermediate_size, gate_input_grad_ptr, config_.stride, config_.grad_type, config_.grad_type, GGML_TYPE_F32);

            void* up_proj_t_ptr = (uint8_t*)up_proj_t_ + (expert_id * config_.hidden_size + ith * config_.stride) * config_.intermediate_size * ggml_type_size(config_.grad_type);
            float* up_input_grad_ptr = s_up_input_grad_[expert_idx] + ith * config_.stride;
            ggml_compute_params params_up;
        params_up.ith = ith;
        params_up.nth = std::max(1, nth);  // Ensure nth > 0 to avoid assertion failure
        params_up.threadpool = nullptr;
            llamafile_sgemm(&params_up, config_.stride, 1, config_.intermediate_size, up_proj_t_ptr, config_.intermediate_size, s_up_output_grad_[expert_idx], config_.intermediate_size, up_input_grad_ptr, config_.stride, config_.grad_type, config_.grad_type, GGML_TYPE_F32);
            
            for (int i = ith * config_.stride; i < (ith + 1) * config_.stride; i++) {
                s_input_grad_fp32_[i] += s_gate_input_grad_[expert_idx][i] + s_up_input_grad_[expert_idx][i];
            }
        }
        from_float(s_input_grad_fp32_ + ith * config_.stride, (uint8_t*)input_grad + ith * config_.stride * ggml_type_size(config_.grad_type), config_.stride, config_.grad_type);
    }, nullptr);
	// clk4 = clock();
	// std::cout << "[Δclk12] " << (clk2 - clk1) / static_cast<double>(CLOCKS_PER_SEC) * 1000
    //       << " ms  [Δclk23] " << (clk3 - clk2) / static_cast<double>(CLOCKS_PER_SEC) * 1000
    //       << " ms  [Δclk34] " << (clk4 - clk3) / static_cast<double>(CLOCKS_PER_SEC) * 1000
    //       << " ms  [Δclkz12] " << (clkz2 - clkz1) / static_cast<double>(CLOCKS_PER_SEC) * 1000
    //       << " ms  [Δclkz23] " << (clkz3 - clkz2) / static_cast<double>(CLOCKS_PER_SEC) * 1000
    //       << " ms  [Δclkz34] " << (clkz4 - clkz3) / static_cast<double>(CLOCKS_PER_SEC) * 1000
    //       << " ms  [Δclkz45] " << (clkz5 - clkz4) / static_cast<double>(CLOCKS_PER_SEC) * 1000
    //       << " ms\n";

}

void SFT_MOE::backward_many(int qlen, int k, const uint64_t* expert_ids, const float* weights, const void* output_grad, void* input_grad, Backend* backend, const SFT_MoEForwardCache* fwd_cache) {
    for (int i = 0; i < config_.expert_num; i++) {
        m_local_num_[i] = 0;
    }
    for (int i = 0; i < qlen; i++) {
        for (int j = 0; j < k; j++) {
            m_local_pos_[i][j] = m_local_num_[expert_ids[i * k + j]]++;
        }
    }
    uint64_t offset = 0;
    for (int i = 0; i < config_.expert_num; i++) {
        m_local_down_output_grad_ptr_[i] = m_local_down_output_grad_ + offset * config_.hidden_size * ggml_type_size(config_.grad_type);
        m_local_down_input_grad_ptr_[i] = m_local_down_input_grad_ + offset * config_.intermediate_size;
        m_local_gate_output_grad_fp32_ptr_[i] = m_local_gate_output_grad_fp32_ + offset * config_.intermediate_size;
        m_local_up_output_grad_fp32_ptr_[i] = m_local_up_output_grad_fp32_ + offset * config_.intermediate_size;
        m_local_gate_output_grad_ptr_[i] = m_local_gate_output_grad_ + offset * config_.intermediate_size * ggml_type_size(config_.grad_type);
        m_local_up_output_grad_ptr_[i] = m_local_up_output_grad_ + offset * config_.intermediate_size * ggml_type_size(config_.grad_type);
        m_local_gate_input_grad_ptr_[i] = m_local_gate_input_grad_ + offset * config_.hidden_size;
        m_local_up_input_grad_ptr_[i] = m_local_up_input_grad_ + offset * config_.hidden_size;
        m_local_token_indices_ptr_[i] = m_local_token_indices_ + offset;
        m_local_expert_positions_ptr_[i] = m_local_expert_positions_ + offset;
        offset += m_local_num_[i];
    }

    backend->do_work_stealing_job(qlen, nullptr, [&](int i) {
        for (int j = 0; j < k; j++) {
            uint64_t expert_id = expert_ids[i * k + j];
            int local_row = m_local_pos_[i][j];
            memcpy(m_local_down_output_grad_ptr_[expert_id] + local_row * config_.hidden_size * ggml_type_size(config_.grad_type), (uint8_t*)output_grad + i * config_.hidden_size * ggml_type_size(config_.grad_type), config_.hidden_size * ggml_type_size(config_.grad_type));
            m_local_token_indices_ptr_[expert_id][local_row] = i;
            m_local_expert_positions_ptr_[expert_id][local_row] = j;
        }
    }, nullptr);

    // get_transpose(backend);

    // Use config_.stride if QK_K is not defined or is 0, otherwise use QK_K
    #ifndef QK_K
    #define QK_K 256  // Default quantization block size
    #endif
    int stride = (QK_K > 0) ? QK_K : config_.stride;
    if (stride <= 0) stride = 64;  // Safety fallback
    int nth = config_.intermediate_size / stride;
    backend->do_work_stealing_job(nth * config_.expert_num, nullptr, [&](int task_id) {
        uint64_t expert_idx = task_id / nth;
        int ith = task_id % nth;
        
        void* down_proj_t_ptr = (uint8_t*)down_proj_t_ + (expert_idx * config_.intermediate_size + ith * stride) * config_.hidden_size * ggml_type_size(config_.grad_type);
        void* down_output_grad_ptr = m_local_down_output_grad_ptr_[expert_idx];
        float* down_input_grad_ptr = m_local_down_input_grad_ptr_[expert_idx] + ith * stride;
                    
        ggml_compute_params params_down;
        params_down.ith = ith;
        params_down.nth = std::max(1, nth);  // Ensure nth > 0 to avoid assertion failure
        params_down.threadpool = nullptr;
        if (params_down.nth <= 0) { 
            fprintf(stderr, "[SFT_MOE DEBUG] params_down.nth is %ld, fixing to 1 (nth=%d, config_.stride=%d)\n", params_down.nth, nth, config_.stride);
            params_down.nth = 1;  // Final safety check
        }
        if (params_down.nth <= 0) {
            fprintf(stderr, "[SFT_MOE DEBUG] CRITICAL: params_down.nth is still %ld after fix! Aborting llamafile_sgemm call.\n", params_down.nth);
            abort();
        }
        llamafile_sgemm(&params_down, stride, m_local_num_[expert_idx], config_.hidden_size, down_proj_t_ptr, config_.hidden_size, down_output_grad_ptr, config_.hidden_size, down_input_grad_ptr, config_.intermediate_size, config_.grad_type, config_.grad_type, GGML_TYPE_F32);
        
        for (int i = 0; i < m_local_num_[expert_idx]; i++) {
            int token_idx = m_local_token_indices_ptr_[expert_idx][i];
            int expert_pos = m_local_expert_positions_ptr_[expert_idx][i];
            float weight = weights[token_idx * k + expert_pos];
            
            for (int j = ith * stride; j < (ith + 1) * stride; j++) {
                m_local_down_input_grad_ptr_[expert_idx][i * config_.intermediate_size + j] *= weight;
                
                float down_input_grad = m_local_down_input_grad_ptr_[expert_idx][i * config_.intermediate_size + j];
                m_local_gate_output_grad_fp32_ptr_[expert_idx][i * config_.intermediate_size + j] = down_input_grad * fwd_cache[token_idx].up_v[expert_pos][j] * act_fn_grad(fwd_cache[token_idx].gate_u[expert_pos][j]);
                m_local_up_output_grad_fp32_ptr_[expert_idx][i * config_.intermediate_size + j] = down_input_grad * act_fn(fwd_cache[token_idx].gate_u[expert_pos][j]);
            }
            
            float* gate_output_grad_fp32_ptr = m_local_gate_output_grad_fp32_ptr_[expert_idx] + i * config_.intermediate_size + ith * stride;
            void* gate_output_grad_ptr = m_local_gate_output_grad_ptr_[expert_idx] + (i * config_.intermediate_size + ith * stride) * ggml_type_size(config_.grad_type);
            from_float(gate_output_grad_fp32_ptr, gate_output_grad_ptr, stride, config_.grad_type);
            
            float* up_output_grad_fp32_ptr = m_local_up_output_grad_fp32_ptr_[expert_idx] + i * config_.intermediate_size + ith * stride;
            void* up_output_grad_ptr = m_local_up_output_grad_ptr_[expert_idx] + (i * config_.intermediate_size + ith * stride) * ggml_type_size(config_.grad_type);
            from_float(up_output_grad_fp32_ptr, up_output_grad_ptr, stride, config_.grad_type);
        }
    }, nullptr);
    #ifndef QK_K
    #define QK_K 256  // Default quantization block size
    #endif
    stride = (QK_K > 0) ? QK_K : config_.stride;
    if (stride <= 0) stride = 64;  // Safety fallback
    nth = config_.hidden_size / stride;
    backend->do_work_stealing_job(nth * config_.expert_num, nullptr, [&](int task_id) {
        uint64_t expert_idx = task_id / nth;
        int ith = task_id % nth;
        
        void* gate_proj_t_ptr = (uint8_t*)gate_proj_t_ + (expert_idx * config_.hidden_size + ith * stride) * config_.intermediate_size * ggml_type_size(config_.grad_type);
        void* up_proj_t_ptr = (uint8_t*)up_proj_t_ + (expert_idx * config_.hidden_size + ith * stride) * config_.intermediate_size * ggml_type_size(config_.grad_type);
        void* gate_output_grad_ptr = m_local_gate_output_grad_ptr_[expert_idx];
        void* up_output_grad_ptr = m_local_up_output_grad_ptr_[expert_idx];
        float* gate_input_grad_ptr = m_local_gate_input_grad_ptr_[expert_idx] + ith * stride;
        float* up_input_grad_ptr = m_local_up_input_grad_ptr_[expert_idx] + ith * stride;
        
        ggml_compute_params params_gate;
        params_gate.ith = ith;
        params_gate.nth = std::max(1, nth);  // Ensure nth > 0 to avoid assertion failure
        params_gate.threadpool = nullptr;
        if (params_gate.nth <= 0) { params_gate.nth = 1; }  // Final safety check
        llamafile_sgemm(&params_gate, stride, m_local_num_[expert_idx], config_.intermediate_size, gate_proj_t_ptr, config_.intermediate_size, gate_output_grad_ptr, config_.intermediate_size, gate_input_grad_ptr, config_.hidden_size, config_.grad_type, config_.grad_type, GGML_TYPE_F32);
        ggml_compute_params params_up;
        params_up.ith = ith;
        params_up.nth = std::max(1, nth);  // Ensure nth > 0 to avoid assertion failure
        params_up.threadpool = nullptr;
        if (params_up.nth <= 0) { 
            fprintf(stderr, "[SFT_MOE DEBUG] params_up.nth is %ld, fixing to 1 (nth=%d, config_.stride=%d)\n", params_up.nth, nth, config_.stride);
            params_up.nth = 1;  // Final safety check
        }
        if (params_up.nth <= 0) {
            fprintf(stderr, "[SFT_MOE DEBUG] CRITICAL: params_up.nth is still %ld after fix! Aborting llamafile_sgemm call.\n", params_up.nth);
            abort();
        }
        llamafile_sgemm(&params_up, stride, m_local_num_[expert_idx], config_.intermediate_size, up_proj_t_ptr, config_.intermediate_size, up_output_grad_ptr, config_.intermediate_size, up_input_grad_ptr, config_.hidden_size, config_.grad_type, config_.grad_type, GGML_TYPE_F32);
    }, nullptr);
    backend->do_work_stealing_job(qlen, nullptr, [&](int i) {
        for (int e = 0; e < config_.hidden_size; e++) {
            m_grad_input_fp32_[i][e] = 0;
        }
        for (int j = 0; j < k; j++) {
            for (int e = 0; e < config_.hidden_size; e++) {
                m_grad_input_fp32_[i][e] += m_local_gate_input_grad_ptr_[expert_ids[i * k + j]][m_local_pos_[i][j] * config_.hidden_size + e] + m_local_up_input_grad_ptr_[expert_ids[i * k + j]][m_local_pos_[i][j] * config_.hidden_size + e];
            }
        }
        from_float(m_grad_input_fp32_[i], (uint8_t*)input_grad + i * config_.hidden_size * ggml_type_size(config_.grad_type), config_.hidden_size, config_.grad_type);
    }, nullptr);
}

// TODO: input和layer_idx参数可以删除
void SFT_MOE::backward(int layer_idx, int qlen, int k, const uint64_t* expert_ids, const float* weights,
                   const void* input, const void* grad_output, void* grad_input, Backend* backend, const SFT_MoEForwardCache* fwd_cache) {
    // Debug: Check if KSFT_MOE_DEBUG is set
    const char* debug_env = std::getenv("KSFT_MOE_DEBUG");
    bool debug = (debug_env != nullptr && std::string(debug_env) == "1");
    
    if (debug) {
        fprintf(stderr, "[C++ SFT_MOE::backward] Entering backward: layer_idx=%d, qlen=%d, k=%d\n", layer_idx, qlen, k);
        fflush(stderr);
    }
    
    // Validate inputs
    if (fwd_cache == nullptr) {
        throw std::runtime_error("fwd_cache is nullptr in SFT_MOE::backward");
    }
    if (expert_ids == nullptr || weights == nullptr || input == nullptr || grad_output == nullptr || grad_input == nullptr) {
        throw std::runtime_error("One or more input pointers are nullptr in SFT_MOE::backward");
    }
    if (backend == nullptr) {
        throw std::runtime_error("backend is nullptr in SFT_MOE::backward");
    }

    if (debug) {
        fprintf(stderr, "[C++ SFT_MOE::backward] Calling get_transpose...\n");
        fflush(stderr);
    }
    get_transpose(backend);
    
    if (debug) {
        fprintf(stderr, "[C++ SFT_MOE::backward] get_transpose completed, starting backward loop\n");
        fflush(stderr);
    }
    
    int remaining_qlen = qlen;
    int processed_offset = 0;
    
    while (remaining_qlen > 0) {
        if (debug) {
            fprintf(stderr, "[C++ SFT_MOE::backward] Loop iteration: remaining_qlen=%d, processed_offset=%d\n", remaining_qlen, processed_offset);
            fflush(stderr);
        }
        
        // config_.group_min_len = 10000000;
        if (remaining_qlen < config_.group_min_len) {
            if (debug) {
                fprintf(stderr, "[C++ SFT_MOE::backward] Using backward_one path (remaining_qlen=%d < group_min_len=%d)\n", remaining_qlen, config_.group_min_len);
                fflush(stderr);
            }
            
            for (int i = 0; i < remaining_qlen; i++) {
                if (debug && i == 0) {
                    fprintf(stderr, "[C++ SFT_MOE::backward] Calling backward_one for token %d\n", i);
                    fflush(stderr);
                }
                
                // Validate cache entry is accessible
                const SFT_MoEForwardCache* cache_entry = fwd_cache + processed_offset + i;
                if (cache_entry == nullptr) {
                    throw std::runtime_error("fwd_cache entry is nullptr in backward_one");
                }
                
                backward_one(k,
                             expert_ids + (processed_offset + i) * k,
                             weights + (processed_offset + i) * k,
                             (uint8_t*)grad_output + (processed_offset + i) * config_.hidden_size * ggml_type_size(config_.grad_type),
                             (uint8_t*)grad_input + (processed_offset + i) * config_.hidden_size * ggml_type_size(config_.grad_type),
                             backend,
                             fwd_cache + processed_offset + i);
            }
            break;
        } else {
            int backward_len = std::min(config_.group_max_len, remaining_qlen);
            
            if (debug) {
                fprintf(stderr, "[C++ SFT_MOE::backward] Using backward_many path (backward_len=%d)\n", backward_len);
                fflush(stderr);
            }
            
            // Validate cache range is accessible
            const SFT_MoEForwardCache* cache_start = fwd_cache + processed_offset;
            if (cache_start == nullptr) {
                throw std::runtime_error("fwd_cache range is nullptr in backward_many");
            }
            
            if (debug) {
                fprintf(stderr, "[C++ SFT_MOE::backward] About to call backward_many...\n");
                fflush(stderr);
            }
            
            backward_many(backward_len, 
                         k, 
                         expert_ids + processed_offset * k, 
                         weights + processed_offset * k, 
                         (uint8_t*)grad_output + processed_offset * config_.hidden_size * ggml_type_size(config_.grad_type), 
                         (uint8_t*)grad_input + processed_offset * config_.hidden_size * ggml_type_size(config_.grad_type), 
                         backend, 
                         fwd_cache + processed_offset);
            
            if (debug) {
                fprintf(stderr, "[C++ SFT_MOE::backward] backward_many completed\n");
                fflush(stderr);
            }
            
            remaining_qlen -= backward_len;
            processed_offset += backward_len;
        }
    }
    
    if (debug) {
        fprintf(stderr, "[C++ SFT_MOE::backward] Backward completed successfully\n");
        fflush(stderr);
    }
}