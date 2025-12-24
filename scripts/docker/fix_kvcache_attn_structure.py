#!/usr/bin/env python3
"""
Fix code structure issues in kvcache_attn.cpp

This script fixes:
1. Missing closing braces
2. Orphaned code blocks
3. Missing for loop bodies
4. Premature lambda closings
"""

import sys
import re

def fix_kvcache_attn_structure(file_path):
    """Fix structure issues in kvcache_attn.cpp"""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    original_count = len(lines)
    new_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        line_num = i + 1
        
        # Fix 1: Remove premature lambda closing at line 1318
        if line_num == 1318 and line.strip() == '});':
            # Skip this line - it's a premature closing
            i += 1
            continue
        
        # Fix 2: Remove orphaned statement at line 1319
        if line_num == 1319 and 'thread_local_attn_lse_[thread_id][i];' in line and not line.strip().startswith('for'):
            # Skip this orphaned statement
            i += 1
            continue
        
        # Fix 3: Add missing closing brace after else block
        if line_num == 1318 and 'nullptr, nullptr);' in line:
            new_lines.append(line)
            # Check if next line starts a for loop (should be after else closes)
            if i + 1 < len(lines) and lines[i + 1].strip().startswith('for'):
                new_lines.append('            }\n')  # Close the else block
            i += 1
            continue
        
        # Fix 4: Add missing for loop body
        if line.strip() == 'for (int i = 0; i < n_gqa_; i++) {' and i + 1 < len(lines):
            next_line = lines[i + 1]
            # If next line doesn't start with proper indentation for loop body, add it
            if not next_line.strip().startswith('block_lse_') and not next_line.strip().startswith('int'):
                new_lines.append(line)
                new_lines.append('                block_lse_[batch_id][block_idx][head_id * n_gqa_ + i] =\n')
                new_lines.append('                    thread_local_attn_lse_[thread_id][i];\n')
                new_lines.append('            }\n')
                i += 1
                continue
        
        new_lines.append(line)
        i += 1
    
    with open(file_path, 'w') as f:
        f.writelines(new_lines)
    
    print(f"Fixed structure issues in {file_path}")
    print(f"Original: {original_count} lines, New: {len(new_lines)} lines")
    return len(new_lines) != original_count

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python3 fix_kvcache_attn_structure.py <path_to_kvcache_attn.cpp>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    fix_kvcache_attn_structure(file_path)

