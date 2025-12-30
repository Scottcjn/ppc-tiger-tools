#!/usr/bin/env python
"""
Add BitNet (bitnet-b1.58) architecture support to llama.cpp.

BitNet uses:
- I2_S ternary weights for Q, K, V, O, ffn_gate, ffn_up, ffn_down
- F32 for norms (attn_norm, ffn_norm, attn_sub_norm, ffn_sub_norm)
- RMSNorm after attention and in FFN (sub-layer norms)

This patch adds:
1. LLM_ARCH_BITNET enum
2. Architecture name mapping
3. New tensor types for sub-norms
4. Tensor name mappings
5. Model hyperparameter loading
6. Tensor loading
7. Compute graph building (build_bitnet)
"""

import os
import sys
import re

def backup_file(filename):
    """Create backup of file."""
    if os.path.exists(filename):
        f = open(filename, 'r')
        content = f.read()
        f.close()
        f = open(filename + '.bitnet-bak', 'w')
        f.write(content)
        f.close()

def patch_llama_cpp():
    """Apply BitNet architecture patches to llama.cpp."""
    if not os.path.exists('llama.cpp'):
        print("llama.cpp not found")
        return False

    f = open('llama.cpp', 'r')
    content = f.read()
    f.close()

    backup_file('llama.cpp')

    modified = False

    # 1. Add LLM_ARCH_BITNET to enum (before UNKNOWN)
    old_arch_enum = '    LLM_ARCH_ORION,\n    LLM_ARCH_UNKNOWN,'
    new_arch_enum = '    LLM_ARCH_ORION,\n    LLM_ARCH_BITNET,\n    LLM_ARCH_UNKNOWN,'

    if old_arch_enum in content and 'LLM_ARCH_BITNET' not in content:
        content = content.replace(old_arch_enum, new_arch_enum)
        print("Added LLM_ARCH_BITNET to enum")
        modified = True

    # 2. Add architecture name mapping
    old_arch_name = '    { LLM_ARCH_ORION,           "orion"     },'
    new_arch_name = '    { LLM_ARCH_ORION,           "orion"     },\n    { LLM_ARCH_BITNET,          "bitnet-b1.58" },'

    if old_arch_name in content and '"bitnet-b1.58"' not in content:
        content = content.replace(old_arch_name, new_arch_name)
        print("Added bitnet-b1.58 name mapping")
        modified = True

    # 3. Add new tensor types for sub-norms (after ATTN_K_NORM)
    old_tensor_enum = '    LLM_TENSOR_ATTN_K_NORM,'
    new_tensor_enum = '    LLM_TENSOR_ATTN_K_NORM,\n    LLM_TENSOR_ATTN_SUB_NORM,\n    LLM_TENSOR_FFN_SUB_NORM,'

    if old_tensor_enum in content and 'LLM_TENSOR_ATTN_SUB_NORM' not in content:
        content = content.replace(old_tensor_enum, new_tensor_enum)
        print("Added LLM_TENSOR_ATTN_SUB_NORM and LLM_TENSOR_FFN_SUB_NORM")
        modified = True

    # 4. Add tensor name mappings for BitNet
    # Find the end of the ORION tensor mapping block
    orion_tensor_end = '''        },
    },
};'''

    bitnet_tensor_mapping = '''        },
    },
    {
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
};'''

    if 'LLM_ARCH_BITNET' in content and 'LLM_ARCH_BITNET,' not in content.split('LLM_TENSOR_NAMES')[1][:2000]:
        # Need to add tensor mapping
        # Find the last closing of LLM_TENSOR_NAMES
        idx = content.find('LLM_TENSOR_NAMES')
        if idx != -1:
            # Find the closing }; of the map
            end_idx = content.find('};', idx)
            while end_idx != -1:
                # Check if this is followed by another map entry or end of array
                next_chunk = content[end_idx:end_idx+100]
                if '    {' not in next_chunk or 'LLM_ARCH' not in next_chunk:
                    # This is the end of the tensor names map
                    break
                end_idx = content.find('};', end_idx + 1)

            if end_idx != -1:
                # Find where to insert - look for pattern
                # Actually let's just search for the ORION block end
                pass

    # Simpler approach: insert after ORION tensor mappings
    if 'LLM_ARCH_BITNET' in content:
        # Find ORION in tensor names
        orion_idx = content.find('LLM_ARCH_ORION,', content.find('LLM_TENSOR_NAMES'))
        if orion_idx != -1:
            # Find the closing }, for ORION
            brace_count = 0
            i = orion_idx
            while i < len(content):
                if content[i] == '{':
                    brace_count += 1
                elif content[i] == '}':
                    brace_count -= 1
                    if brace_count == -1:
                        # Found the end of ORION block
                        # Check if BITNET already there
                        if 'LLM_ARCH_BITNET' not in content[i:i+500]:
                            # Insert BitNet mapping
                            bitnet_block = ''',
    {
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
    }'''
                            content = content[:i+1] + bitnet_block + content[i+1:]
                            print("Added BitNet tensor name mappings")
                            modified = True
                        break
                i += 1

    # 5. Add ffn_sub_norm to llama_layer struct
    old_struct = '    struct ggml_tensor * ffn_norm_b;'
    new_struct = '    struct ggml_tensor * ffn_norm_b;\n    struct ggml_tensor * attn_sub_norm;  // BitNet\n    struct ggml_tensor * ffn_sub_norm;   // BitNet'

    if old_struct in content and 'ffn_sub_norm' not in content:
        content = content.replace(old_struct, new_struct)
        print("Added attn_sub_norm and ffn_sub_norm to llama_layer struct")
        modified = True

    # 6. Add hyperparameter loading for BitNet (after ORION case)
    old_orion_case = '''        case LLM_ARCH_ORION:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

                switch (hparams.n_layer) {
                    case 40: model.type = e_model::MODEL_14B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;'''

    new_orion_and_bitnet = '''        case LLM_ARCH_ORION:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

                switch (hparams.n_layer) {
                    case 40: model.type = e_model::MODEL_14B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_BITNET:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 26: model.type = e_model::MODEL_3B; break;
                    case 32: model.type = e_model::MODEL_7B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;'''

    if old_orion_case in content and 'LLM_ARCH_BITNET:' not in content.split('hparams.f_norm')[0]:
        content = content.replace(old_orion_case, new_orion_and_bitnet)
        print("Added BitNet hyperparameter loading")
        modified = True

    # 7. Add tensor loading for BitNet (large block - insert after ORION tensor loading)
    # Find ORION tensor loading block
    orion_tensor_load = '            case LLM_ARCH_ORION:'

    bitnet_tensor_load = '''            case LLM_ARCH_BITNET:
                {
                    model.tok_embd = ml.create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

                    // output
                    {
                        model.output_norm = ml.create_tensor(ctx_output,       tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
                        model.output      = ml.create_tensor(ctx_output_split, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, false);
                        if (!model.output) {
                            model.output = ml.create_tensor(ctx_output_split, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, false);
                            ml.n_created--; // reset for next check
                        }
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        ggml_context * ctx_layer = ctx_for_layer(i);
                        ggml_context * ctx_split = ctx_for_layer_split(i);

                        auto & layer = model.layers[i];

                        layer.attn_norm = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});
                        layer.attn_sub_norm = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_SUB_NORM, "weight", i), {n_embd}, false);

                        layer.wq = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd});
                        layer.wk = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa});
                        layer.wv = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa});
                        layer.wo = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});

                        layer.ffn_norm = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});
                        layer.ffn_sub_norm = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_SUB_NORM, "weight", i), {n_ff}, false);

                        layer.ffn_gate = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff});
                        layer.ffn_down = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd});
                        layer.ffn_up   = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff});
                    }
                } break;
'''

    # Find where ORION tensor loading ends and insert BitNet before default
    orion_match = content.find(orion_tensor_load, content.find('tn = LLM_TN'))
    if orion_match != -1 and 'case LLM_ARCH_BITNET:' not in content[orion_match:orion_match+5000]:
        # Find the end of ORION block
        # Look for "} break;" pattern after ORION
        i = orion_match
        brace_depth = 0
        found_start = False
        while i < len(content):
            if content[i] == '{':
                brace_depth += 1
                found_start = True
            elif content[i] == '}':
                brace_depth -= 1
                if found_start and brace_depth == 0:
                    # Look for "break;" after this
                    end_search = content[i:i+50]
                    if 'break;' in end_search:
                        break_idx = i + end_search.find('break;') + 6
                        # Insert BitNet tensor loading after ORION
                        content = content[:break_idx] + '\n' + bitnet_tensor_load + content[break_idx:]
                        print("Added BitNet tensor loading")
                        modified = True
                    break
            i += 1

    # 8. Add build_bitnet() case in switch (after ORION)
    old_graph_orion = '''        case LLM_ARCH_ORION:
            {
                result = llm.build_orion();
            } break;'''

    new_graph_orion_bitnet = '''        case LLM_ARCH_ORION:
            {
                result = llm.build_orion();
            } break;
        case LLM_ARCH_BITNET:
            {
                result = llm.build_bitnet();
            } break;'''

    if old_graph_orion in content and 'build_bitnet' not in content:
        content = content.replace(old_graph_orion, new_graph_orion_bitnet)
        print("Added build_bitnet() switch case")
        modified = True

    # 9. Add build_bitnet() function declaration and implementation
    # Find build_orion declaration and add build_bitnet after it
    orion_decl = '    struct ggml_cgraph * build_orion() {'
    if orion_decl in content and 'build_bitnet()' not in content:
        # Find the end of build_orion function
        orion_idx = content.find(orion_decl)
        if orion_idx != -1:
            # Find matching closing brace
            brace_depth = 0
            i = orion_idx
            while i < len(content):
                if content[i] == '{':
                    brace_depth += 1
                elif content[i] == '}':
                    brace_depth -= 1
                    if brace_depth == 0:
                        # Found end of build_orion
                        # Insert build_bitnet after
                        bitnet_build = '''

    struct ggml_cgraph * build_bitnet() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, LLAMA_MAX_NODES, false);

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = llm_build_inp_embd(ctx0, hparams, batch, model.tok_embd, lctx.inp_tokens, lctx.inp_embd, cb);
        cb(inpL, "inp_embd", -1);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = ggml_view_1d(ctx0, lctx.inp_pos, n_tokens, 0);
        cb(inp_pos, "inp_pos", -1);

        // KQ_mask (mask for 1 head, it will be broadcast to all heads)
        struct ggml_tensor * KQ_mask = ggml_view_2d(ctx0, lctx.inp_KQ_mask, n_kv, n_tokens, n_kv*ggml_type_size(lctx.inp_KQ_mask->type), 0);
        cb(KQ_mask, "KQ_mask", -1);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            // norm
            cur = llm_build_norm(ctx0, inpL, hparams,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, cb, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor * Qcur = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);

                struct ggml_tensor * Kcur = ggml_mul_mat(ctx0, model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);

                struct ggml_tensor * Vcur = ggml_mul_mat(ctx0, model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_rope_custom(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos,
                    hparams.n_rot, 0, 0, n_orig_ctx, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_custom(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos,
                    hparams.n_rot, 0, 0, n_orig_ctx, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = llm_build_kv(ctx0, model, hparams, kv_self, gf,
                        model.layers[il].wo, NULL,
                        Kcur, Vcur, Qcur, KQ_mask, nullptr, n_ctx, n_tokens, kv_head, n_kv, 1.0f/sqrtf(float(n_embd_head)), cb, il);
                cb(cur, "kqv_out", il);
            }

            // BitNet: sub-layer norm after attention
            if (model.layers[il].attn_sub_norm) {
                cur = llm_build_norm(ctx0, cur, hparams,
                        model.layers[il].attn_sub_norm, NULL,
                        LLM_NORM_RMS, cb, il);
                cb(cur, "attn_sub_norm", il);
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            cur = llm_build_norm(ctx0, ffn_inp, hparams,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, cb, il);
            cb(cur, "ffn_norm", il);

            cur = llm_build_ffn(ctx0, cur,
                    model.layers[il].ffn_up,   NULL,
                    model.layers[il].ffn_gate, NULL,
                    model.layers[il].ffn_down, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            cb(cur, "ffn_out", il);

            // BitNet: sub-layer norm after FFN gate/up
            if (model.layers[il].ffn_sub_norm) {
                cur = llm_build_norm(ctx0, cur, hparams,
                        model.layers[il].ffn_sub_norm, NULL,
                        LLM_NORM_RMS, cb, il);
                cb(cur, "ffn_sub_norm", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = llm_build_norm(ctx0, cur, hparams,
                model.output_norm, NULL,
                LLM_NORM_RMS, cb, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = ggml_mul_mat(ctx0, model.output, cur);
        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }
'''
                        content = content[:i+1] + bitnet_build + content[i+1:]
                        print("Added build_bitnet() function")
                        modified = True
                        break
                i += 1

    if modified:
        f = open('llama.cpp', 'w')
        f.write(content)
        f.close()

    return modified


if __name__ == '__main__':
    if not os.path.exists('llama.cpp'):
        print("Error: Run from llama.cpp directory")
        sys.exit(1)

    print("=== BitNet Architecture Patch for llama.cpp ===")
    print()

    if patch_llama_cpp():
        print()
        print("=== Patch applied successfully ===")
        print("Recompile with:")
        print("  /usr/local/bin/gmake CC=/usr/local/bin/gcc-7 CXX=/usr/local/bin/g++-7 LLAMA_NO_METAL=1 LLAMA_NO_ACCELERATE=1 -j2 main")
    else:
        print("No changes made or patch failed.")
