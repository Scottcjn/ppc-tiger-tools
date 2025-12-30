#!/usr/bin/env python
"""
Fix GGUF bool size for PowerPC/Tiger where sizeof(bool) == 4.
GGUF spec requires 1-byte bools.

Two fixes needed:
1. GGUF_TYPE_SIZE table entry for bool
2. gguf_fread_el call for GGUF_TYPE_BOOL case
"""

import os
import sys

def patch_gguf_bool():
    """Fix bool size in GGUF type size table and read code."""
    if not os.path.exists('ggml.c'):
        print("ggml.c not found")
        return False

    f = open('ggml.c', 'r')
    content = f.read()
    f.close()

    # Backup
    f = open('ggml.c.bool-bak', 'w')
    f.write(content)
    f.close()

    modified = False

    # Fix 1: Change sizeof(bool) to 1 in the GGUF type size array
    old_bool_size = '[GGUF_TYPE_BOOL]    = sizeof(bool),'
    new_bool_size = '[GGUF_TYPE_BOOL]    = 1,  /* GGUF spec: always 1 byte, not sizeof(bool) */'

    if old_bool_size in content:
        content = content.replace(old_bool_size, new_bool_size)
        print("Fixed GGUF_TYPE_BOOL size to 1 byte in type table")
        modified = True
    else:
        print("Type table already patched or not found")

    # Fix 2: Change the bool read to use 1 byte explicitly
    # Original: case GGUF_TYPE_BOOL:    ok = ok && gguf_fread_el (file, &kv->value.bool_,   sizeof(kv->value.bool_),   &offset); break;
    # Fixed:   Read 1 byte into a temp, then assign to bool_
    old_bool_read = 'case GGUF_TYPE_BOOL:    ok = ok && gguf_fread_el (file, &kv->value.bool_,   sizeof(kv->value.bool_),   &offset); break;'
    new_bool_read = 'case GGUF_TYPE_BOOL:    { uint8_t b = 0; ok = ok && gguf_fread_el(file, &b, 1, &offset); kv->value.bool_ = (bool)b; } break;'

    if old_bool_read in content:
        content = content.replace(old_bool_read, new_bool_read)
        print("Fixed GGUF_TYPE_BOOL read to use 1 byte")
        modified = True
    else:
        # Try with different spacing
        old_bool_read2 = 'case GGUF_TYPE_BOOL:'
        if old_bool_read2 in content and 'uint8_t b = 0' not in content:
            # Need to find and replace the full line
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'case GGUF_TYPE_BOOL:' in line and 'sizeof(kv->value.bool_)' in line:
                    lines[i] = '                case GGUF_TYPE_BOOL:    { uint8_t b = 0; ok = ok && gguf_fread_el(file, &b, 1, &offset); kv->value.bool_ = (bool)b; } break;'
                    print("Fixed GGUF_TYPE_BOOL read to use 1 byte (alternate match)")
                    modified = True
                    break
            content = '\n'.join(lines)
        else:
            print("Bool read already patched or not found")

    if modified:
        f = open('ggml.c', 'w')
        f.write(content)
        f.close()

    return modified


if __name__ == '__main__':
    if not os.path.exists('ggml.c'):
        print("Error: Run from llama.cpp directory")
        sys.exit(1)
    if patch_gguf_bool():
        print("Patch applied. Recompile with:")
        print("  /usr/local/bin/gmake CC=/usr/local/bin/gcc-7 CXX=/usr/local/bin/g++-7 LLAMA_NO_METAL=1 LLAMA_NO_ACCELERATE=1 -j2 main")
    else:
        print("No changes made.")
