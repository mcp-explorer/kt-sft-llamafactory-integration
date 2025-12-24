#!/usr/bin/env python3
"""
Check function signatures in kvcache_attn.cpp against header declarations

This script helps identify function signature mismatches.
"""

import sys
import re
import os

def extract_function_declarations(header_path):
    """Extract function declarations from header file"""
    if not os.path.exists(header_path):
        print(f"Warning: Header file not found: {header_path}")
        return {}
    
    with open(header_path, 'r') as f:
        content = f.read()
    
    # Pattern to match function declarations
    # e.g., void KVCache::function_name(type1 arg1, type2 arg2);
    pattern = r'(?:void|int|bool|float|double|const\s+\w+\*?)\s+KVCache::(\w+)\s*\(([^)]*)\)'
    
    declarations = {}
    for match in re.finditer(pattern, content):
        func_name = match.group(1)
        params = match.group(2).strip()
        declarations[func_name] = params
    
    return declarations

def extract_function_calls(cpp_path, func_name):
    """Extract all calls to a specific function"""
    with open(cpp_path, 'r') as f:
        lines = f.readlines()
    
    calls = []
    pattern = re.compile(rf'{func_name}\s*\(')
    
    for i, line in enumerate(lines, 1):
        if pattern.search(line):
            # Try to extract the full call (may span multiple lines)
            call_lines = [line]
            j = i
            paren_count = line.count('(') - line.count(')')
            
            while paren_count > 0 and j < len(lines):
                j += 1
                if j < len(lines):
                    call_lines.append(lines[j - 1])
                    paren_count += lines[j - 1].count('(') - lines[j - 1].count(')')
            
            calls.append((i, ''.join(call_lines)))
    
    return calls

def check_signature_mismatch(header_path, cpp_path, func_name):
    """Check if function calls match the declaration"""
    declarations = extract_function_declarations(header_path)
    
    if func_name not in declarations:
        print(f"Function {func_name} not found in header file")
        return
    
    expected_params = declarations[func_name]
    calls = extract_function_calls(cpp_path, func_name)
    
    print(f"\nFunction: {func_name}")
    print(f"Declaration: {func_name}({expected_params})")
    print(f"Found {len(calls)} calls:")
    
    for line_num, call in calls[:5]:  # Show first 5
        # Extract arguments from call
        # This is simplified - actual parsing would be more complex
        match = re.search(rf'{func_name}\s*\((.*)\)', call, re.DOTALL)
        if match:
            args = match.group(1)
            arg_count = len([a for a in args.split(',') if a.strip()])
            expected_count = len([p for p in expected_params.split(',') if p.strip()]) if expected_params else 0
            
            if arg_count != expected_count:
                print(f"  Line {line_num}: Mismatch - Expected {expected_count} args, got {arg_count}")
                print(f"    Call: {call[:100]}...")

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 check_function_signatures.py <header_file> <cpp_file> [function_name]")
        sys.exit(1)
    
    header_path = sys.argv[1]
    cpp_path = sys.argv[2]
    
    if len(sys.argv) == 4:
        func_name = sys.argv[3]
        check_signature_mismatch(header_path, cpp_path, func_name)
    else:
        # Check all functions
        declarations = extract_function_declarations(header_path)
        print(f"Found {len(declarations)} function declarations")
        
        # Check common problematic functions
        problematic_funcs = [
            'calculate_block_similarity_kvhead_',
            'select_block_kvhead_',
            'calculate_sparsity_layer_',
            'calculate_sparsity_kvhead_',
            'attn_with_kvcache_one_block_',
        ]
        
        for func_name in problematic_funcs:
            if func_name in declarations:
                check_signature_mismatch(header_path, cpp_path, func_name)

if __name__ == '__main__':
    main()

