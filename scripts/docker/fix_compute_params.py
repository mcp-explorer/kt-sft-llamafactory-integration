#!/usr/bin/env python3
"""
Fix ggml_compute_params initialization to use field-by-field assignment
instead of aggregate initialization to avoid incomplete type errors
"""

import re
import sys

def fix_compute_params(content):
    """Fix ggml_compute_params initialization"""
    
    # Pattern: ggml_compute_params params = {ith, nth, nullptr};
    pattern = r'ggml_compute_params\s+(\w+)\s*=\s*\{([^}]+)\};'
    
    def replace_init(match):
        var_name = match.group(1)
        args = [a.strip() for a in match.group(2).split(',')]
        
        if len(args) >= 2:
            ith = args[0]
            nth = args[1]
            threadpool = args[2] if len(args) > 2 else 'nullptr'
            
            replacement = f"""ggml_compute_params {var_name};
        {var_name}.ith = {ith};
        {var_name}.nth = {nth};
        {var_name}.threadpool = {threadpool};"""
            return replacement
        return match.group(0)
    
    new_content = re.sub(pattern, replace_init, content)
    return new_content

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 fix_compute_params.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    new_content = fix_compute_params(content)
    
    with open(file_path, 'w') as f:
        f.write(new_content)
    
    print(f"Fixed ggml_compute_params initialization in {file_path}")

