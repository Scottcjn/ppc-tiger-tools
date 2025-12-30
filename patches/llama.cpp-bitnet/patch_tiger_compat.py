#!/usr/bin/env python3
"""
Patch llama.cpp b2000 for Tiger (GCC 4.0.1) compatibility.
Fixes:
1. deprecated(message) -> deprecated (GCC 4.0.1 doesn't support message)
2. restrict keyword -> empty (not in C89)
3. Add I2_S support
"""

import sys
import os
import re

def patch_ggml_h():
    """Patch ggml.h for Tiger compatibility."""
    if not os.path.exists('ggml.h'):
        print("ggml.h not found")
        return False

    with open('ggml.h', 'r') as f:
        content = f.read()

    # Backup
    with open('ggml.h.tiger-bak', 'w') as f:
        f.write(content)

    # 1. Fix GGML_DEPRECATED - use deprecated without message for older GCC
    # Change:
    #   #ifdef __GNUC__
    #   #    define GGML_DEPRECATED(func, hint) func __attribute__((deprecated(hint)))
    # To check GCC version and use simple deprecated for old GCC
    old_deprecated = '#ifdef __GNUC__\n#    define GGML_DEPRECATED(func, hint) func __attribute__((deprecated(hint)))'
    new_deprecated = '''#ifdef __GNUC__
#  if __GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 5)
#    define GGML_DEPRECATED(func, hint) func __attribute__((deprecated))
#  else
#    define GGML_DEPRECATED(func, hint) func __attribute__((deprecated(hint)))
#  endif'''

    if old_deprecated in content:
        content = content.replace(old_deprecated, new_deprecated)
        print("Fixed GGML_DEPRECATED for old GCC")
    else:
        print("GGML_DEPRECATED already patched or not found")

    # 2. Fix GGML_RESTRICT - add check for C89/C90
    # Change:
    #   #else
    #   #define GGML_RESTRICT restrict
    # To check for C99
    old_restrict = '''#ifdef  __cplusplus
// restrict not standard in C++
#define GGML_RESTRICT
#else
#define GGML_RESTRICT restrict
#endif'''

    new_restrict = '''#ifdef  __cplusplus
// restrict not standard in C++
#define GGML_RESTRICT
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
// C99 or later - restrict is available
#define GGML_RESTRICT restrict
#else
// C89/C90 - restrict not available
#define GGML_RESTRICT
#endif'''

    if old_restrict in content:
        content = content.replace(old_restrict, new_restrict)
        print("Fixed GGML_RESTRICT for C89")
    else:
        print("GGML_RESTRICT already patched or not found")

    with open('ggml.h', 'w') as f:
        f.write(content)

    return True


def patch_ggml_quants_h():
    """Patch ggml-quants.h for Tiger compatibility."""
    if not os.path.exists('ggml-quants.h'):
        print("ggml-quants.h not found")
        return False

    with open('ggml-quants.h', 'r') as f:
        content = f.read()

    # Backup
    with open('ggml-quants.h.tiger-bak', 'w') as f:
        f.write(content)

    modified = False

    # 1. Add restrict compat for C89 at the top of the file
    tiger_compat = '''/* Tiger/C89 compatibility - restrict and static_assert not available */
#if !defined(__STDC_VERSION__) || __STDC_VERSION__ < 199901L
#  ifndef restrict
#    define restrict
#  endif
#endif

#ifndef static_assert
#  define static_assert(cond, msg)
#endif

'''
    if '/* Tiger/C89 compatibility' not in content:
        # Find first #include or #ifndef after any initial comments
        lines = content.split('\n')
        insert_idx = 0
        for i, line in enumerate(lines):
            if line.startswith('#ifndef') or line.startswith('#include'):
                insert_idx = i
                break
        if insert_idx > 0:
            lines.insert(insert_idx, tiger_compat)
            content = '\n'.join(lines)
            modified = True
            print("Added restrict/static_assert compat to ggml-quants.h")

    if modified:
        with open('ggml-quants.h', 'w') as f:
            f.write(content)

    return True


def patch_ggml_c():
    """Patch ggml.c to add I2_S include and entry."""
    if not os.path.exists('ggml.c'):
        print("ggml.c not found")
        return False

    with open('ggml.c', 'r') as f:
        content = f.read()

    modified = False

    # 1. Add I2_S include if not present
    if '#include "ggml-i2s-patch.h"' not in content:
        # Add after ggml-impl.h include
        content = content.replace(
            '#include "ggml-impl.h"',
            '#include "ggml-impl.h"\n#include "ggml-i2s-patch.h"'
        )
        print("Added ggml-i2s-patch.h include")
        modified = True

    # 2. Add I2_S entry to type_traits if not present
    if 'GGML_TYPE_I2_S' not in content:
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
        # Find the type_traits array - look for Q8_K entry WITHIN it
        # The array starts with: static const ggml_type_traits_t type_traits[GGML_TYPE_COUNT]
        array_start = content.find('type_traits[GGML_TYPE_COUNT]')
        if array_start == -1:
            print("Could not find type_traits array")
            return False

        # Find Q8_K entry after array start
        q8k_pattern = '[GGML_TYPE_Q8_K]'
        q8k_idx = content.find(q8k_pattern, array_start)
        if q8k_idx == -1:
            print("Could not find Q8_K entry in type_traits")
            return False

        # Q8_K is the last entry - ends with "}\n};" not "},"
        # Find the closing "}" for Q8_K entry followed by array end "};"
        # Look for pattern: "    }\n};"
        array_end_pattern = '    }\n};'
        array_end_idx = content.find(array_end_pattern, q8k_idx)
        if array_end_idx == -1:
            # Try with different indentation
            array_end_pattern = '}\n};'
            array_end_idx = content.find(array_end_pattern, q8k_idx)

        if array_end_idx == -1:
            print("Could not find end of type_traits array after Q8_K")
            return False

        # Insert after the "}" of Q8_K, before the "};" of the array
        # We need to add a comma to Q8_K entry and insert I2_S
        # Replace "    }\n};" with "    },\n    [I2_S]...\n};"
        old_ending = content[array_end_idx:array_end_idx+len(array_end_pattern)]
        new_ending = '    },\n' + i2s_entry + '};'
        content = content[:array_end_idx] + new_ending + content[array_end_idx+len(array_end_pattern):]
        print("Added I2_S entry to type_traits (after Q8_K)")
        modified = True

    if modified:
        # Backup
        with open('ggml.c.tiger-bak', 'w') as f:
            with open('ggml.c', 'r') as orig:
                f.write(orig.read())

        with open('ggml.c', 'w') as f:
            f.write(content)

    return True


def main():
    if not os.path.exists('ggml.h'):
        print("Error: Run this from llama.cpp directory")
        return 1

    print("=== Tiger Compatibility Patch for llama.cpp b2000 ===\n")

    patch_ggml_h()
    patch_ggml_quants_h()
    patch_ggml_c()

    print("\n=== Patching complete ===")
    print("Try compiling with: make LLAMA_NO_METAL=1")
    return 0


if __name__ == '__main__':
    sys.exit(main())
