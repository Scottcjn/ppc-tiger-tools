#!/usr/bin/env python
"""
Add debug output to GGUF reader to diagnose loading issues.
Python 2 compatible version.
"""

import os
import sys

def patch_gguf_debug():
    """Add debug prints to gguf_init_from_file."""
    if not os.path.exists('ggml.c'):
        print("ggml.c not found")
        return False

    f = open('ggml.c', 'r')
    content = f.read()
    f.close()

    # Add debug after reading header
    old_header = '''        ok = ok && gguf_fread_scalar(file, &ctx->header.version,   sizeof(ctx->header.version),   &offset);
        ok = ok && gguf_fread_scalar(file, &ctx->header.n_tensors, sizeof(ctx->header.n_tensors), &offset);
        ok = ok && gguf_fread_scalar(file, &ctx->header.n_kv,      sizeof(ctx->header.n_kv),      &offset);'''

    new_header = '''        ok = ok && gguf_fread_scalar(file, &ctx->header.version,   sizeof(ctx->header.version),   &offset);
        ok = ok && gguf_fread_scalar(file, &ctx->header.n_tensors, sizeof(ctx->header.n_tensors), &offset);
        ok = ok && gguf_fread_scalar(file, &ctx->header.n_kv,      sizeof(ctx->header.n_kv),      &offset);

        fprintf(stderr, "DEBUG: GGUF header - version=%u, n_tensors=%llu, n_kv=%llu\\n",
                ctx->header.version, (unsigned long long)ctx->header.n_tensors,
                (unsigned long long)ctx->header.n_kv);'''

    if old_header in content and 'DEBUG: GGUF header' not in content:
        content = content.replace(old_header, new_header)
        print("Added debug output to header reading")
    else:
        print("Header debug already patched or not found")

    # Add debug for KV reading
    old_kv = '''            ok = ok && gguf_fread_str(file, &kv->key,                    &offset);
            ok = ok && gguf_fread_scalar(file, &kv->type, sizeof(kv->type), &offset);'''

    new_kv = '''            ok = ok && gguf_fread_str(file, &kv->key,                    &offset);
            ok = ok && gguf_fread_scalar(file, &kv->type, sizeof(kv->type), &offset);

            fprintf(stderr, "DEBUG: KV[%llu] key='%s' type=%u\\n", i,
                    kv->key.data ? kv->key.data : "(null)", kv->type);'''

    if old_kv in content and 'DEBUG: KV[' not in content:
        content = content.replace(old_kv, new_kv)
        print("Added debug output to KV reading")
    else:
        print("KV debug already patched or not found")

    f = open('ggml.c', 'w')
    f.write(content)
    f.close()

    return True


if __name__ == '__main__':
    if not os.path.exists('ggml.c'):
        print("Error: Run from llama.cpp directory")
        sys.exit(1)
    patch_gguf_debug()
    print("Debug patches applied. Recompile and run.")
