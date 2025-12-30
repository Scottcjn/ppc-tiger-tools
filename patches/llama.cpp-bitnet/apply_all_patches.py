#!/usr/bin/env python3
"""
Apply all BitNet I2_S patches to llama.cpp.
Run from the llama.cpp directory.

Usage:
    cd llama.cpp
    python3 /path/to/apply_all_patches.py
"""

import os
import sys
import shutil

def backup_file(filename):
    """Create backup of file."""
    if os.path.exists(filename):
        backup = filename + '.orig'
        if not os.path.exists(backup):
            shutil.copy2(filename, backup)
            print(f"  Backed up {filename}")

def patch_ggml_h():
    """Add GGML_TYPE_I2_S to ggml.h"""
    filename = 'ggml.h'
    if not os.path.exists(filename):
        print(f"  {filename} not found, skipping")
        return False

    with open(filename, 'r') as f:
        content = f.read()

    if 'GGML_TYPE_I2_S' in content:
        print(f"  {filename}: I2_S already present")
        return True

    backup_file(filename)

    # Add I2_S type after Q8_K
    old = 'GGML_TYPE_Q8_K  = 15,'
    new = '''GGML_TYPE_Q8_K  = 15,
        GGML_TYPE_I2_S  = 36,  // BitNet ternary 2-bit'''

    if old in content:
        content = content.replace(old, new)
        with open(filename, 'w') as f:
            f.write(content)
        print(f"  {filename}: Added GGML_TYPE_I2_S")
        return True
    else:
        print(f"  {filename}: Could not find insertion point")
        return False

def patch_ggml_c():
    """Add I2_S type traits and bool fix to ggml.c"""
    filename = 'ggml.c'
    if not os.path.exists(filename):
        print(f"  {filename} not found")
        return False

    with open(filename, 'r') as f:
        content = f.read()

    backup_file(filename)
    modified = False

    # Fix 1: GGUF bool size (for big-endian platforms where sizeof(bool)==4)
    old_bool = '[GGUF_TYPE_BOOL]    = sizeof(bool),'
    new_bool = '[GGUF_TYPE_BOOL]    = 1,  /* GGUF spec: always 1 byte */'
    if old_bool in content:
        content = content.replace(old_bool, new_bool)
        print(f"  {filename}: Fixed GGUF bool size")
        modified = True

    # Fix 2: Bool read to use 1 byte
    old_bool_read = 'case GGUF_TYPE_BOOL:    ok = ok && gguf_fread_el (file, &kv->value.bool_,   sizeof(kv->value.bool_),   &offset); break;'
    new_bool_read = 'case GGUF_TYPE_BOOL:    { uint8_t b = 0; ok = ok && gguf_fread_el(file, &b, 1, &offset); kv->value.bool_ = (bool)b; } break;'
    if old_bool_read in content:
        content = content.replace(old_bool_read, new_bool_read)
        print(f"  {filename}: Fixed GGUF bool read")
        modified = True

    # Fix 3: Add I2_S to type_traits (if not present)
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
        # Find Q8_K entry and add after it
        q8k_end = content.find('[GGML_TYPE_Q8_K]')
        if q8k_end != -1:
            # Find the closing }, of Q8_K entry
            brace_count = 0
            i = q8k_end
            while i < len(content):
                if content[i] == '{':
                    brace_count += 1
                elif content[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        # Insert after the },
                        insert_pos = content.find(',', i) + 1
                        content = content[:insert_pos] + '\n' + i2s_entry + content[insert_pos:]
                        print(f"  {filename}: Added I2_S type_traits")
                        modified = True
                        break
                i += 1

    if modified:
        with open(filename, 'w') as f:
            f.write(content)

    return modified

def patch_llama_cpp():
    """Add BitNet architecture support to llama.cpp"""
    filename = 'llama.cpp'
    if not os.path.exists(filename):
        print(f"  {filename} not found")
        return False

    with open(filename, 'r') as f:
        lines = f.readlines()

    backup_file(filename)
    modified = False

    # Check if already patched
    content = ''.join(lines)
    if 'LLM_ARCH_BITNET' in content:
        print(f"  {filename}: BitNet already present")
        return True

    # 1. Add LLM_ARCH_BITNET to enum
    for i, line in enumerate(lines):
        if 'LLM_ARCH_ORION,' in line and 'LLM_ARCH_BITNET' not in ''.join(lines[i:i+5]):
            lines.insert(i+1, '    LLM_ARCH_BITNET,\n')
            print(f"  {filename}: Added LLM_ARCH_BITNET enum")
            modified = True
            break

    # 2. Add architecture name mapping
    for i, line in enumerate(lines):
        if '"orion"' in line and 'LLM_ARCH_ORION' in line:
            lines.insert(i+1, '    { LLM_ARCH_BITNET,          "bitnet-b1.58" },\n')
            print(f"  {filename}: Added bitnet-b1.58 name")
            modified = True
            break

    # 3. Add tensor enums
    for i, line in enumerate(lines):
        if 'LLM_TENSOR_ATTN_K_NORM,' in line and 'LLM_TENSOR_ATTN_SUB_NORM' not in ''.join(lines[i:i+5]):
            lines.insert(i+1, '    LLM_TENSOR_ATTN_SUB_NORM,\n')
            lines.insert(i+2, '    LLM_TENSOR_FFN_SUB_NORM,\n')
            print(f"  {filename}: Added sub-norm tensor enums")
            modified = True
            break

    # 4. Add struct fields for sub-norms
    for i, line in enumerate(lines):
        if 'struct ggml_tensor * ffn_norm_b;' in line and 'attn_sub_norm' not in ''.join(lines[i:i+5]):
            lines.insert(i+1, '    struct ggml_tensor * attn_sub_norm;  // BitNet\n')
            lines.insert(i+2, '    struct ggml_tensor * ffn_sub_norm;   // BitNet\n')
            print(f"  {filename}: Added sub-norm struct fields")
            modified = True
            break

    # 5. Add BitNet tensor mappings
    in_tensor_names = False
    for i, line in enumerate(lines):
        if 'LLM_TENSOR_NAMES' in line:
            in_tensor_names = True
        if in_tensor_names and 'LLM_ARCH_ORION' in line:
            # Find end of ORION block
            brace_depth = 0
            for j in range(i, min(i+50, len(lines))):
                brace_depth += lines[j].count('{') - lines[j].count('}')
                if brace_depth <= 0 and '},' in lines[j]:
                    bitnet_tensors = '''    {
        LLM_ARCH_BITNET,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_SUB_NORM,   "blk.%d.attn_sub_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_SUB_NORM,    "blk.%d.ffn_sub_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
'''
                    lines.insert(j+1, bitnet_tensors)
                    print(f"  {filename}: Added BitNet tensor mappings")
                    modified = True
                    break
            break

    # 6. Add hyperparameter loading
    for i, line in enumerate(lines):
        if 'case LLM_ARCH_ORION:' in line and 'hparams' in ''.join(lines[i:i+20]):
            for j in range(i, min(i+30, len(lines))):
                if '} break;' in lines[j] and 'LLM_ARCH_BITNET' not in ''.join(lines[j:j+10]):
                    bitnet_hparams = '''        case LLM_ARCH_BITNET:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 26: model.type = e_model::MODEL_3B; break;
                    case 32: model.type = e_model::MODEL_7B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
'''
                    lines.insert(j+1, bitnet_hparams)
                    print(f"  {filename}: Added BitNet hparams loading")
                    modified = True
                    break
            break

    # 7. Add tensor loading
    for i, line in enumerate(lines):
        if 'case LLM_ARCH_ORION:' in line and 'ml.create_tensor' in ''.join(lines[i:i+50]):
            brace_depth = 0
            for j in range(i, min(i+100, len(lines))):
                brace_depth += lines[j].count('{') - lines[j].count('}')
                if '} break;' in lines[j] and brace_depth <= 1:
                    if 'LLM_ARCH_BITNET' not in ''.join(lines[j:j+10]):
                        bitnet_tensors_load = '''            case LLM_ARCH_BITNET:
                {
                    model.tok_embd = ml.create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});
                    {
                        model.output_norm = ml.create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
                        model.output = ml.create_tensor(ctx_output_split, tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, false);
                        if (!model.output) {
                            model.output = ml.create_tensor(ctx_output_split, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, false);
                            ml.n_created--;
                        }
                    }
                    for (int i = 0; i < n_layer; ++i) {
                        ggml_context * ctx_layer = ctx_for_layer(i);
                        ggml_context * ctx_split = ctx_for_layer_split(i);
                        auto & layer = model.layers[i];
                        layer.attn_norm = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});
                        layer.attn_sub_norm = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_SUB_NORM, "weight", i), {n_embd}, false);
                        layer.wq = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q, "weight", i), {n_embd, n_embd});
                        layer.wk = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K, "weight", i), {n_embd, n_embd_gqa});
                        layer.wv = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V, "weight", i), {n_embd, n_embd_gqa});
                        layer.wo = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});
                        layer.ffn_norm = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});
                        layer.ffn_sub_norm = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_SUB_NORM, "weight", i), {n_ff}, false);
                        layer.ffn_gate = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff});
                        layer.ffn_down = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd});
                        layer.ffn_up = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP, "weight", i), {n_embd, n_ff});
                    }
                } break;
'''
                        lines.insert(j+1, bitnet_tensors_load)
                        print(f"  {filename}: Added BitNet tensor loading")
                        modified = True
                    break
            break

    # 8. Add build_bitnet switch case
    for i, line in enumerate(lines):
        if 'llm.build_orion()' in line:
            for j in range(i, min(i+5, len(lines))):
                if '} break;' in lines[j] and 'build_bitnet' not in ''.join(lines[j:j+10]):
                    bitnet_case = '''        case LLM_ARCH_BITNET:
            {
                result = llm.build_bitnet();
            } break;
'''
                    lines.insert(j+1, bitnet_case)
                    print(f"  {filename}: Added build_bitnet switch case")
                    modified = True
                    break
            break

    # 9. Add build_bitnet function
    for i, line in enumerate(lines):
        if 'struct ggml_cgraph * build_orion()' in line:
            brace_depth = 0
            for j in range(i, min(i+200, len(lines))):
                brace_depth += lines[j].count('{') - lines[j].count('}')
                if brace_depth == 0 and lines[j].strip() == '}':
                    if 'build_bitnet' not in ''.join(lines[j:j+10]):
                        build_bitnet_func = '''
    struct ggml_cgraph * build_bitnet() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, LLAMA_MAX_NODES, false);

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = llm_build_inp_embd(ctx0, hparams, batch, model.tok_embd, lctx.inp_tokens, lctx.inp_embd, cb);
        cb(inpL, "inp_embd", -1);

        struct ggml_tensor * inp_pos = ggml_view_1d(ctx0, lctx.inp_pos, n_tokens, 0);
        cb(inp_pos, "inp_pos", -1);

        struct ggml_tensor * KQ_mask = ggml_view_2d(ctx0, lctx.inp_KQ_mask, n_kv, n_tokens, n_kv*ggml_type_size(lctx.inp_KQ_mask->type), 0);
        cb(KQ_mask, "KQ_mask", -1);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            cur = llm_build_norm(ctx0, inpL, hparams,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, cb, il);
            cb(cur, "attn_norm", il);

            {
                struct ggml_tensor * Qcur = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                struct ggml_tensor * Kcur = ggml_mul_mat(ctx0, model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                struct ggml_tensor * Vcur = ggml_mul_mat(ctx0, model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_rope_custom(ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos,
                    hparams.n_rot, 0, 0, n_orig_ctx, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
                cb(Qcur, "Qcur", il);
                Kcur = ggml_rope_custom(ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos,
                    hparams.n_rot, 0, 0, n_orig_ctx, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
                cb(Kcur, "Kcur", il);

                cur = llm_build_kv(ctx0, model, hparams, kv_self, gf,
                        model.layers[il].wo, NULL, Kcur, Vcur, Qcur, KQ_mask,
                        n_ctx, n_tokens, kv_head, n_kv, -1.0f, 1.0f/sqrtf(float(n_embd_head)), cb, il);
                cb(cur, "kqv_out", il);
            }

            if (model.layers[il].attn_sub_norm) {
                cur = llm_build_norm(ctx0, cur, hparams, model.layers[il].attn_sub_norm, NULL, LLM_NORM_RMS, cb, il);
                cb(cur, "attn_sub_norm", il);
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            cur = llm_build_norm(ctx0, ffn_inp, hparams, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, cb, il);
            cb(cur, "ffn_norm", il);

            cur = llm_build_ffn(ctx0, cur, model.layers[il].ffn_up, NULL, model.layers[il].ffn_gate, NULL,
                    model.layers[il].ffn_down, NULL, NULL, LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            cb(cur, "ffn_out", il);

            if (model.layers[il].ffn_sub_norm) {
                cur = llm_build_norm(ctx0, cur, hparams, model.layers[il].ffn_sub_norm, NULL, LLM_NORM_RMS, cb, il);
                cb(cur, "ffn_sub_norm", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "l_out", il);
            inpL = cur;
        }

        cur = inpL;
        cur = llm_build_norm(ctx0, cur, hparams, model.output_norm, NULL, LLM_NORM_RMS, cb, -1);
        cb(cur, "result_norm", -1);
        cur = ggml_mul_mat(ctx0, model.output, cur);
        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);
        return gf;
    }
'''
                        lines.insert(j+1, build_bitnet_func)
                        print(f"  {filename}: Added build_bitnet function")
                        modified = True
                    break
            break

    if modified:
        with open(filename, 'w') as f:
            f.writelines(lines)

    return modified

def copy_header_files(patch_dir):
    """Copy I2_S header files to llama.cpp directory."""
    headers = ['ggml-i2s-patch.h', 'ggml-i2s-altivec.h']
    for header in headers:
        src = os.path.join(patch_dir, header)
        if os.path.exists(src):
            shutil.copy2(src, header)
            print(f"  Copied {header}")
        else:
            print(f"  Warning: {header} not found in patch directory")

def main():
    print("=" * 60)
    print("BitNet I2_S Patch for llama.cpp")
    print("=" * 60)
    print()

    # Check we're in llama.cpp directory
    if not os.path.exists('ggml.c') or not os.path.exists('llama.cpp'):
        print("Error: Run this script from the llama.cpp directory")
        print("  cd llama.cpp")
        print("  python3 /path/to/apply_all_patches.py")
        sys.exit(1)

    patch_dir = os.path.dirname(os.path.abspath(__file__))

    print("[1/4] Patching ggml.h...")
    patch_ggml_h()

    print("\n[2/4] Patching ggml.c...")
    patch_ggml_c()

    print("\n[3/4] Patching llama.cpp...")
    patch_llama_cpp()

    print("\n[4/4] Copying header files...")
    copy_header_files(patch_dir)

    print()
    print("=" * 60)
    print("Patching complete!")
    print("=" * 60)
    print()
    print("Build with:")
    print("  make -j4")
    print()
    print("For PowerPC (Tiger/Leopard):")
    print("  gmake CC=gcc-7 CXX=g++-7 LLAMA_NO_METAL=1 LLAMA_NO_ACCELERATE=1 -j2")
    print()

if __name__ == '__main__':
    main()
