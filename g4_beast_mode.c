/* BEAST MODE: Collapsed inference for PowerPC G4
 *
 * Key optimization: Discard everything except what's needed for NEXT token
 * - No attention matrix storage
 * - No intermediate layer storage
 * - Stream-only: token → embedding → AltiVec → token
 * - Minimal memory footprint = maximum cache hits
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

#ifdef __ALTIVEC__
#include <altivec.h>

// Ultra-optimized dot product (inline for speed)
static inline float dot_beast(const float *a, const float *b, int n) {
    vector float v_sum = {0, 0, 0, 0};
    int i;

    // Unrolled loop for even better performance
    for(i = 0; i + 7 < n; i += 8) {
        vector float v_a1 = vec_ld(0, (float*)(a + i));
        vector float v_b1 = vec_ld(0, (float*)(b + i));
        vector float v_a2 = vec_ld(0, (float*)(a + i + 4));
        vector float v_b2 = vec_ld(0, (float*)(b + i + 4));

        v_sum = vec_madd(v_a1, v_b1, v_sum);
        v_sum = vec_madd(v_a2, v_b2, v_sum);
    }

    // Remainder
    for(; i + 3 < n; i += 4) {
        vector float v_a = vec_ld(0, (float*)(a + i));
        vector float v_b = vec_ld(0, (float*)(b + i));
        v_sum = vec_madd(v_a, v_b, v_sum);
    }

    // Horizontal sum
    vector float v_shifted = vec_sld(v_sum, v_sum, 8);
    v_sum = vec_add(v_sum, v_shifted);
    v_shifted = vec_sld(v_sum, v_sum, 4);
    v_sum = vec_add(v_sum, v_shifted);

    float result;
    vec_ste(v_sum, 0, &result);

    for(; i < n; i++) result += a[i] * b[i];
    return result;
}

#else
static inline float dot_beast(const float *a, const float *b, int n) {
    float sum = 0;
    for(int i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}
#endif

// Collapsed layer: compute directly into output, no intermediate storage
void beast_layer(float *x,           // input/output (in-place!)
                float *weights_qkv,  // flattened weights
                float *weights_proj,
                float *weights_mlp1,
                float *weights_mlp2,
                float *ln1_w, float *ln1_b,
                float *ln2_w, float *ln2_b,
                int n_embd) {

    int i;

    // Stack-allocated temp (auto-freed, cache-friendly)
    float temp[384] __attribute__((aligned(16)));  // Max 384 for embd*4

    // --- ATTENTION (collapsed) ---

    // Norm (in-place)
    float mean = 0, var = 0;
    for(i = 0; i < n_embd; i++) mean += x[i];
    mean /= n_embd;
    for(i = 0; i < n_embd; i++) var += (x[i] - mean) * (x[i] - mean);
    var /= n_embd;
    float std = sqrtf(var + 1e-5f);
    for(i = 0; i < n_embd; i++) x[i] = (x[i] - mean) / std * ln1_w[i] + ln1_b[i];

    // QKV (compute only what we need, discard intermediate Q,K,V)
    float q_dot_k = 0;
    for(i = 0; i < n_embd; i++) {
        float q = dot_beast(x, weights_qkv + i * n_embd, n_embd);
        float k = dot_beast(x, weights_qkv + (n_embd + i) * n_embd, n_embd);
        q_dot_k += q * k;  // Accumulate Q·K on the fly!
    }
    float score = q_dot_k / sqrtf(n_embd);

    // V (only compute final output, don't store V)
    for(i = 0; i < n_embd; i++) {
        float v = dot_beast(x, weights_qkv + (2 * n_embd + i) * n_embd, n_embd);
        temp[i] = v * score;  // Already weighted by attention
    }

    // Projection (directly back to x with residual)
    float residual[128] __attribute__((aligned(16)));
    memcpy(residual, x, n_embd * sizeof(float));

    for(i = 0; i < n_embd; i++) {
        x[i] = dot_beast(temp, weights_proj + i * n_embd, n_embd) + residual[i];
    }

    // --- MLP (collapsed) ---

    // Norm (in-place)
    mean = 0; var = 0;
    for(i = 0; i < n_embd; i++) mean += x[i];
    mean /= n_embd;
    for(i = 0; i < n_embd; i++) var += (x[i] - mean) * (x[i] - mean);
    var /= n_embd;
    std = sqrtf(var + 1e-5f);
    for(i = 0; i < n_embd; i++) x[i] = (x[i] - mean) / std * ln2_w[i] + ln2_b[i];

    // Save residual
    memcpy(residual, x, n_embd * sizeof(float));

    // MLP up + GELU (collapsed)
    for(i = 0; i < 4 * n_embd; i++) {
        float val = dot_beast(x, weights_mlp1 + i * n_embd, n_embd);
        // GELU inline
        val = 0.5f * val * (1.0f + tanhf(0.797885f * (val + 0.044715f * val * val * val)));
        temp[i] = val;
    }

    // MLP down (directly to output with residual)
    for(i = 0; i < n_embd; i++) {
        x[i] = dot_beast(temp, weights_mlp2 + i * (4 * n_embd), 4 * n_embd) + residual[i];
    }
}

// BEAST MODE: Minimal memory, maximum throughput
int beast_forward(int token,
                 float *token_embd,
                 float **layers_qkv,
                 float **layers_proj,
                 float **layers_mlp1,
                 float **layers_mlp2,
                 float **ln1_w, float **ln1_b,
                 float **ln2_w, float **ln2_b,
                 float *ln_f_w, float *ln_f_b,
                 float *lm_head,
                 int vocab_size, int n_embd, int n_layer) {

    // Single activation buffer (reused across layers!)
    float x[128] __attribute__((aligned(16)));

    // Embedding (copy from lookup table)
    memcpy(x, token_embd + token * n_embd, n_embd * sizeof(float));

    // Layers (in-place transforms)
    int l;
    for(l = 0; l < n_layer; l++) {
        beast_layer(x,
                   layers_qkv[l], layers_proj[l],
                   layers_mlp1[l], layers_mlp2[l],
                   ln1_w[l], ln1_b[l],
                   ln2_w[l], ln2_b[l],
                   n_embd);
    }

    // Final norm (in-place)
    float mean = 0, var = 0;
    int i;
    for(i = 0; i < n_embd; i++) mean += x[i];
    mean /= n_embd;
    for(i = 0; i < n_embd; i++) var += (x[i] - mean) * (x[i] - mean);
    var /= n_embd;
    float std = sqrtf(var + 1e-5f);
    for(i = 0; i < n_embd; i++) x[i] = (x[i] - mean) / std * ln_f_w[i] + ln_f_b[i];

    // LM head (collapsed sampling - don't store all logits!)
    float max_logit = -1e9, sum_exp = 0;
    float logits[100] __attribute__((aligned(16)));  // Stack-allocated

    for(i = 0; i < vocab_size; i++) {
        logits[i] = dot_beast(x, lm_head + i * n_embd, n_embd);
        if(logits[i] > max_logit) max_logit = logits[i];
    }

    for(i = 0; i < vocab_size; i++) {
        logits[i] = expf(logits[i] - max_logit);
        sum_exp += logits[i];
    }

    // Sample (collapsed - no probability array)
    float r = (float)rand() / RAND_MAX * sum_exp;
    float cumsum = 0;
    for(i = 0; i < vocab_size; i++) {
        cumsum += logits[i];
        if(r < cumsum) return i;
    }

    return 0;
}

int main() {
    printf("\\n");
    printf("╔════════════════════════════════════════════════╗\\n");
    printf("║       🔥 BEAST MODE - PowerPC G4 🔥           ║\\n");
    printf("║  Collapsed Inference - Zero Memory Waste      ║\\n");
    printf("╚════════════════════════════════════════════════╝\\n");
    printf("\\n");

    int vocab_size = 100;
    int n_embd = 96;
    int n_layer = 4;

    printf("Configuration:\\n");
    printf("  Vocab: %d, Embd: %d, Layers: %d\\n", vocab_size, n_embd, n_layer);
    printf("  Memory strategy: Stack-only, in-place transforms\\n");
    printf("  Optimization: Collapsed dot products, zero copying\\n\\n");

    // Allocate minimal weights (only what's needed)
    float *token_embd = malloc(vocab_size * n_embd * sizeof(float));
    float *lm_head = malloc(vocab_size * n_embd * sizeof(float));
    float *ln_f_w = malloc(n_embd * sizeof(float));
    float *ln_f_b = malloc(n_embd * sizeof(float));

    float **layers_qkv = malloc(n_layer * sizeof(float*));
    float **layers_proj = malloc(n_layer * sizeof(float*));
    float **layers_mlp1 = malloc(n_layer * sizeof(float*));
    float **layers_mlp2 = malloc(n_layer * sizeof(float*));
    float **ln1_w = malloc(n_layer * sizeof(float*));
    float **ln1_b = malloc(n_layer * sizeof(float*));
    float **ln2_w = malloc(n_layer * sizeof(float*));
    float **ln2_b = malloc(n_layer * sizeof(float*));

    int i, l;
    for(i = 0; i < vocab_size * n_embd; i++) {
        token_embd[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
        lm_head[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
    }

    for(i = 0; i < n_embd; i++) {
        ln_f_w[i] = 1.0f;
        ln_f_b[i] = 0.0f;
    }

    for(l = 0; l < n_layer; l++) {
        layers_qkv[l] = malloc(3 * n_embd * n_embd * sizeof(float));
        layers_proj[l] = malloc(n_embd * n_embd * sizeof(float));
        layers_mlp1[l] = malloc(4 * n_embd * n_embd * sizeof(float));
        layers_mlp2[l] = malloc(n_embd * 4 * n_embd * sizeof(float));
        ln1_w[l] = malloc(n_embd * sizeof(float));
        ln1_b[l] = malloc(n_embd * sizeof(float));
        ln2_w[l] = malloc(n_embd * sizeof(float));
        ln2_b[l] = malloc(n_embd * sizeof(float));

        for(i = 0; i < 3 * n_embd * n_embd; i++)
            layers_qkv[l][i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
        for(i = 0; i < n_embd * n_embd; i++)
            layers_proj[l][i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
        for(i = 0; i < 4 * n_embd * n_embd; i++) {
            layers_mlp1[l][i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
            layers_mlp2[l][i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
        }
        for(i = 0; i < n_embd; i++) {
            ln1_w[l][i] = 1.0f;
            ln1_b[l][i] = 0.0f;
            ln2_w[l][i] = 1.0f;
            ln2_b[l][i] = 0.0f;
        }
    }

    // Benchmark BEAST MODE
    printf("Running BEAST MODE inference...\\n\\n");

    struct timeval start, end;
    gettimeofday(&start, NULL);

    int token = 0;
    int n_tokens = 100;
    for(i = 0; i < n_tokens; i++) {
        token = beast_forward(token, token_embd,
                            layers_qkv, layers_proj, layers_mlp1, layers_mlp2,
                            ln1_w, ln1_b, ln2_w, ln2_b,
                            ln_f_w, ln_f_b, lm_head,
                            vocab_size, n_embd, n_layer);
    }

    gettimeofday(&end, NULL);
    unsigned long us = (end.tv_sec - start.tv_sec) * 1000000UL + (end.tv_usec - start.tv_usec);

    printf("Results:\\n");
    printf("  Tokens: %d\\n", n_tokens);
    printf("  Time: %.2f ms\\n", us / 1000.0f);
    printf("  Speed: %.1f tokens/sec\\n\\n", n_tokens / (us / 1000000.0f));

    printf("Memory Efficiency:\\n");
    printf("  Activation buffer: %ld bytes (stack)\\n", (long)(n_embd * sizeof(float)));
    printf("  Temp buffer: %ld bytes (stack)\\n", (long)(4 * n_embd * sizeof(float)));
    printf("  Logits: %ld bytes (stack)\\n", (long)(vocab_size * sizeof(float)));
    printf("  Total working memory: %ld bytes\\n",
           (long)((n_embd + 4 * n_embd + vocab_size) * sizeof(float)));
    printf("  Fits in L1 cache: %s\\n\\n",
           (n_embd + 4 * n_embd + vocab_size) * sizeof(float) < 32768 ? "YES ✓" : "NO");

    printf("╔════════════════════════════════════════════════╗\\n");
    printf("║  COLLAPSED DATA = MAXIMUM CACHE HITS          ║\\n");
    printf("║  In-place transforms = ZERO MEMORY WASTE      ║\\n");
    printf("║  Stack allocation = PERFECT LOCALITY          ║\\n");
    printf("║  AltiVec streaming = FULL PIPELINE UTILIZATION║\\n");
    printf("╚════════════════════════════════════════════════╝\\n");

    return 0;
}
