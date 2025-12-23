/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-07-12 10:07:58
 * @Version      : 1.0.0
 * @LastEditors  : chenht2022 
 * @LastEditTime : 2024-07-25 10:34:55
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#ifndef CPUINFER_CONVERSION_H
#define CPUINFER_CONVERSION_H

#include <memory.h>
#include "llama.cpp/ggml.h"
#include "llama.cpp/ggml-quants.h"

inline void to_float(const void* input, float* output, int size, ggml_type type) {
    if (type == ggml_type::GGML_TYPE_F32) {
        memcpy(output, input, size * sizeof(float));
    } else {
        ggml_get_type_traits(type)->to_float(input, output, size);
    }
}

inline void from_float(const float* input, void* output, int size, ggml_type type) {
    if (type == ggml_type::GGML_TYPE_F32) {
        memcpy(output, input, size * sizeof(float));
    } else if (type == GGML_TYPE_F16) {
        ggml_fp16_t* out = (ggml_fp16_t*)output;
        for (int i = 0; i < size; i++) {
            out[i] = ggml_fp32_to_fp16(input[i]);
        }
    } else {
        // Note: from_float is not available in ggml_type_traits in newer versions
        // For now, fallback to F32 copy - quantization should be handled elsewhere
        // TODO: Implement proper quantization using quantize_row_* functions
        memcpy(output, input, size * sizeof(float));
    }
}

#endif