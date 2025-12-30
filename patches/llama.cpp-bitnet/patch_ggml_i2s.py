#!/usr/bin/env python3
"""
Patch ggml.c to add I2_S (BitNet) support.
Run from llama.cpp-b2000 directory.
"""

import sys
import os

def patch_ggml():
    with open('ggml.c', 'r') as f:
        content = f.read()

    # Check if already patched
    if 'GGML_TYPE_I2_S' in content:
        print("I2_S already present in ggml.c")
        return True

    # 1. Add include for ggml-i2s-patch.h after ggml-impl.h
    if '#include "ggml-i2s-patch.h"' not in content:
        content = content.replace(
            '#include "ggml-impl.h"',
            '#include "ggml-impl.h"\n#include "ggml-i2s-patch.h"'
        )
        print("Added ggml-i2s-patch.h include")

    # 2. Add I2_S entry to type_traits array after Q8_K
    i2s_entry = '''    [GGML_TYPE_I2_S] = {
        .type_name                = "i2_s",
        .blck_size                = 256,
        .type_size                = 64,
        .is_quantized             = true,
        .to_float                 = NULL,
        .from_float               = NULL,
        .from_float_reference     = NULL,
        .vec_dot                  = NULL,
        .vec_dot_type             = GGML_TYPE_Q8_K,
    },
'''

    # Find the Q8_K entry end and insert I2_S after it
    q8k_pattern = '.type_name                = "q8_K"'
    if q8k_pattern in content:
        # Find closing brace of Q8_K entry
        idx = content.find(q8k_pattern)
        # Look for next "}," after Q8_K
        brace_idx = content.find('},', idx)
        if brace_idx != -1:
            insert_pos = brace_idx + 2  # After "},"
            content = content[:insert_pos] + '\n' + i2s_entry + content[insert_pos:]
            print("Added I2_S entry to type_traits")

    # Backup and write
    with open('ggml.c.bak', 'w') as f:
        f.write(open('ggml.c').read())

    with open('ggml.c', 'w') as f:
        f.write(content)

    print("Patched ggml.c successfully")
    return True

if __name__ == '__main__':
    if not os.path.exists('ggml.c'):
        print("Error: ggml.c not found. Run from llama.cpp directory.")
        sys.exit(1)
    patch_ggml()
