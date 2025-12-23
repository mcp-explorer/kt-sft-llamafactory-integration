#!/usr/bin/env python3
"""
Fix llamafile_sgemm calls to use new signature with ggml_compute_params
Old: llamafile_sgemm(m, n, k, A, lda, B, ldb, C, ldc, ith, nth, GGML_TASK_COMPUTE, type_a, type_b, type_c, PREC)
New: ggml_compute_params params = {ith, nth, nullptr}; llamafile_sgemm(&params, m, n, k, A, lda, B, ldb, C, ldc, type_a, type_b, type_c)
"""

import re
import sys

def fix_llamafile_sgemm_call(content):
    """Fix llamafile_sgemm calls in content"""
    
    # Pattern to match llamafile_sgemm calls with old signature
    # llamafile_sgemm(m, n, k, A, lda, B, ldb, C, ldc, ith, nth, GGML_TASK_COMPUTE, type_a, type_b, type_c, PREC)
    pattern = r'llamafile_sgemm\s*\(\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*GGML_TASK_COMPUTE\s*,\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*([^)]+)\)'
    
    def replace_call(match):
        m = match.group(1).strip()
        n = match.group(2).strip()
        k = match.group(3).strip()
        A = match.group(4).strip()
        lda = match.group(5).strip()
        B = match.group(6).strip()
        ldb = match.group(7).strip()
        C = match.group(8).strip()
        ldc = match.group(9).strip()
        ith = match.group(10).strip()
        nth = match.group(11).strip()
        type_a = match.group(12).strip()
        type_b = match.group(13).strip()
        type_c = match.group(14).strip()
        # PREC is ignored in new signature
        
        # Create params struct before the call
        params_line = f"        ggml_compute_params params = {{{ith}, {nth}, nullptr}};\n"
        new_call = f"llamafile_sgemm(&params, {m}, {n}, {k}, {A}, {lda}, {B}, {ldb}, {C}, {ldc}, {type_a}, {type_b}, {type_c})"
        
        return params_line + "        " + new_call
    
    # Replace all matches
    new_content = re.sub(pattern, replace_call, content)
    
    return new_content

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 fix_llamafile_sgemm.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    new_content = fix_llamafile_sgemm_call(content)
    
    with open(file_path, 'w') as f:
        f.write(new_content)
    
    print(f"Fixed llamafile_sgemm calls in {file_path}")

