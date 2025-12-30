#!/usr/bin/env python
"""
BitNet architecture patch v2 - More careful insertion handling.
"""

import os
import sys

def patch_llama_cpp():
    if not os.path.exists('llama.cpp'):
        print("llama.cpp not found")
        return False

    f = open('llama.cpp', 'r')
    lines = f.readlines()
    f.close()

    # Backup
    f = open('llama.cpp.bitnet-bak2', 'w')
    f.writelines(lines)
    f.close()

    modified = False

    # 1. Add LLM_ARCH_BITNET to enum
    for i, line in enumerate(lines):
        if 'LLM_ARCH_ORION,' in line and 'LLM_ARCH_BITNET' not in ''.join(lines[i:i+5]):
            lines.insert(i+1, '    LLM_ARCH_BITNET,\n')
            print("Added LLM_ARCH_BITNET to enum at line %d" % (i+2))
            modified = True
            break

    # 2. Add architecture name mapping
    for i, line in enumerate(lines):
        if '"orion"' in line and 'LLM_ARCH_ORION' in line and 'bitnet-b1.58' not in ''.join(lines[i:i+3]):
            lines.insert(i+1, '    { LLM_ARCH_BITNET,          "bitnet-b1.58" },\n')
            print("Added bitnet-b1.58 name at line %d" % (i+2))
            modified = True
            break

    # 3. Add tensor enums (after ATTN_K_NORM)
    for i, line in enumerate(lines):
        if 'LLM_TENSOR_ATTN_K_NORM,' in line and 'LLM_TENSOR_ATTN_SUB_NORM' not in ''.join(lines[i:i+5]):
            lines.insert(i+1, '    LLM_TENSOR_ATTN_SUB_NORM,\n')
            lines.insert(i+2, '    LLM_TENSOR_FFN_SUB_NORM,\n')
            print("Added tensor enums at line %d" % (i+2))
            modified = True
            break

    # 4. Add struct fields (after ffn_norm_b)
    for i, line in enumerate(lines):
        if 'struct ggml_tensor * ffn_norm_b;' in line and 'attn_sub_norm' not in ''.join(lines[i:i+5]):
            lines.insert(i+1, '    struct ggml_tensor * attn_sub_norm;  // BitNet\n')
            lines.insert(i+2, '    struct ggml_tensor * ffn_sub_norm;   // BitNet\n')
            print("Added struct fields at line %d" % (i+2))
            modified = True
            break

    # 5. Add BitNet tensor name mappings (after ORION block in LLM_TENSOR_NAMES)
    # Find the ORION entry in tensor names and add after its closing }
    in_tensor_names = False
    orion_start = -1
    brace_depth = 0
    for i, line in enumerate(lines):
        if 'LLM_TENSOR_NAMES' in line:
            in_tensor_names = True
        if in_tensor_names and 'LLM_ARCH_ORION' in line:
            orion_start = i
            brace_depth = 0
        if orion_start >= 0:
            brace_depth += line.count('{') - line.count('}')
            if brace_depth <= 0 and '},' in line:
                # This is the end of ORION tensor block
                if 'LLM_ARCH_BITNET' not in ''.join(lines[i:i+10]):
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
                    lines.insert(i+1, bitnet_tensors)
                    print("Added BitNet tensor mappings at line %d" % (i+2))
                    modified = True
                break

    # 6. Add hyperparameter loading (after ORION case)
    for i, line in enumerate(lines):
        if 'case LLM_ARCH_ORION:' in line and 'hparams' in ''.join(lines[i:i+20]):
            # Find the break statement for this case
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
                    print("Added BitNet hparams at line %d" % (j+2))
                    modified = True
                    break
            break

    # 7. Add tensor loading (after ORION tensor loading case)
    # Search for tensor loading section
    for i, line in enumerate(lines):
        if 'case LLM_ARCH_ORION:' in line and 'ml.create_tensor' in ''.join(lines[i:i+50]):
            # Find the break for this case
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
                        print("Added BitNet tensor loading at line %d" % (j+2))
                        modified = True
                    break
            break

    # 8. Add build_bitnet switch case (after ORION)
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
                    print("Added build_bitnet switch case at line %d" % (j+2))
                    modified = True
                    break
            break

    # 9. Add build_bitnet function (after build_orion function, before the }; closing the struct)
    # Find build_orion function and its closing }
    for i, line in enumerate(lines):
        if 'struct ggml_cgraph * build_orion()' in line:
            brace_depth = 0
            for j in range(i, min(i+200, len(lines))):
                brace_depth += lines[j].count('{') - lines[j].count('}')
                if brace_depth == 0 and lines[j].strip() == '}':
                    # This is the end of build_orion
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

                Qcur = ggml_rope_custom(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos,
                    hparams.n_rot, 0, 0, n_orig_ctx, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_custom(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos,
                    hparams.n_rot, 0, 0, n_orig_ctx, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
                cb(Kcur, "Kcur", il);

                cur = llm_build_kv(ctx0, model, hparams, kv_self, gf,
                        model.layers[il].wo, NULL,
                        Kcur, Vcur, Qcur, KQ_mask, nullptr, n_ctx, n_tokens, kv_head, n_kv, 1.0f/sqrtf(float(n_embd_head)), cb, il);
                cb(cur, "kqv_out", il);
            }

            if (model.layers[il].attn_sub_norm) {
                cur = llm_build_norm(ctx0, cur, hparams,
                        model.layers[il].attn_sub_norm, NULL,
                        LLM_NORM_RMS, cb, il);
                cb(cur, "attn_sub_norm", il);
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            cur = llm_build_norm(ctx0, ffn_inp, hparams,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, cb, il);
            cb(cur, "ffn_norm", il);

            cur = llm_build_ffn(ctx0, cur,
                    model.layers[il].ffn_up, NULL,
                    model.layers[il].ffn_gate, NULL,
                    model.layers[il].ffn_down, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            cb(cur, "ffn_out", il);

            if (model.layers[il].ffn_sub_norm) {
                cur = llm_build_norm(ctx0, cur, hparams,
                        model.layers[il].ffn_sub_norm, NULL,
                        LLM_NORM_RMS, cb, il);
                cb(cur, "ffn_sub_norm", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "l_out", il);

            inpL = cur;
        }

        cur = inpL;

        cur = llm_build_norm(ctx0, cur, hparams,
                model.output_norm, NULL,
                LLM_NORM_RMS, cb, -1);
        cb(cur, "result_norm", -1);

        cur = ggml_mul_mat(ctx0, model.output, cur);
        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }
'''
                        lines.insert(j+1, build_bitnet_func)
                        print("Added build_bitnet function at line %d" % (j+2))
                        modified = True
                    break
            break

    if modified:
        f = open('llama.cpp', 'w')
        f.writelines(lines)
        f.close()

    return modified


if __name__ == '__main__':
    if not os.path.exists('llama.cpp'):
        print("Error: Run from llama.cpp directory")
        sys.exit(1)

    print("=== BitNet Architecture Patch v2 ===")
    if patch_llama_cpp():
        print("\n=== Patch applied ===")
    else:
        print("No changes made")
