/**
 * Fix for ggml_compute_params incomplete type issue
 * This header ensures ggml_compute_params is fully defined
 */

#ifndef GGML_COMPUTE_PARAMS_FIX_H
#define GGML_COMPUTE_PARAMS_FIX_H

#include "llama.cpp/ggml.h"

// Ensure ggml_compute_params is fully defined
// If it's forward-declared only, we need to include the implementation header
#ifndef GGML_COMPUTE_PARAMS_DEFINED
// Try to get the full definition from ggml-impl.h
#include "llama.cpp/ggml-impl.h"
#endif

#endif

