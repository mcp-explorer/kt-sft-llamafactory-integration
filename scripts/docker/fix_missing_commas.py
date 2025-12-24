#!/usr/bin/env python3
"""
Fix missing commas in function calls in kvcache_attn.cpp

This script identifies and fixes missing commas between function arguments.
"""

import sys
import re

def find_missing_commas(file_path):
    """Find lines with potential missing commas"""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    issues = []
    
    # Common patterns that indicate missing commas
    patterns = [
        (r'(\w+)\s+(\w+)\s*\)', 'Possible missing comma between arguments'),
        (r'(\w+)\s+\(', 'Possible missing comma before opening paren'),
        (r'\)\s+(\w+)', 'Possible missing comma after closing paren'),
    ]
    
    for i, line in enumerate(lines, 1):
        # Check for common error patterns
        if 'expected primary-expression before' in line.lower() or \
           'expected \')\' before' in line.lower():
            # Look at the actual source line
            if i - 1 < len(lines):
                source_line = lines[i - 1]
                # Check for patterns that suggest missing comma
                if re.search(r'\w+\s+\w+\s*[,)]', source_line):
                    issues.append((i, source_line.strip()))
    
    return issues

def fix_missing_commas_manual(file_path, line_num, fix):
    """Manually fix a specific missing comma"""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    if line_num < 1 or line_num > len(lines):
        print(f"Invalid line number: {line_num}")
        return False
    
    line = lines[line_num - 1]
    new_line = fix(line)
    
    if new_line != line:
        lines[line_num - 1] = new_line
        with open(file_path, 'w') as f:
            f.writelines(lines)
        print(f"Fixed line {line_num}")
        return True
    
    return False

def suggest_fixes(file_path):
    """Suggest fixes for missing commas"""
    issues = find_missing_commas(file_path)
    
    if not issues:
        print("No obvious missing comma issues found.")
        print("You may need to check compilation errors manually.")
        return
    
    print(f"Found {len(issues)} potential missing comma issues:")
    for line_num, line in issues[:10]:  # Show first 10
        print(f"  Line {line_num}: {line[:80]}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 fix_missing_commas.py <path_to_kvcache_attn.cpp> [line_num]")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    if len(sys.argv) == 3:
        # Manual fix mode
        line_num = int(sys.argv[2])
        # This would need the actual fix - would need to be provided
        print(f"Manual fix mode for line {line_num} not implemented yet.")
        print("Please check the compilation error and fix manually.")
    else:
        suggest_fixes(file_path)

