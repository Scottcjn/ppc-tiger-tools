/* PowerPC G4 LLM Demo
 * Streaming dot product architecture
 * Text → AltiVec vmaddfp → Text
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

#ifdef __ALTIVEC__
#include <altivec.h>

float dot_altivec(const float *a, const float *b, int n) {
    vector float v_sum = {0, 0, 0, 0};
    int i;
    for(i = 0; i + 3 < n; i += 4) {
        vector float v_a = vec_ld(0, (float*)(a + i));
        vector float v_b = vec_ld(0, (float*)(b + i));
        v_sum = vec_madd(v_a, v_b, v_sum);
    }
    vector float v_shifted = vec_sld(v_sum, v_sum, 8);
    v_sum = vec_add(v_sum, v_shifted);
    v_shifted = vec_sld(v_sum, v_sum, 4);
    v_sum = vec_add(v_sum, v_shifted);
    float result;
    vec_ste(v_sum, 0, &result);
    for(; i < n; i++) result += a[i] * b[i];
    return result;
}
#define dot_product dot_altivec
#else
float dot_product(const float *a, const float *b, int n) {
    float sum = 0;
    for(int i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}
#endif

void layer_norm(float *x, float *w, float *b, int n) {
    float mean = 0, var = 0;
    int i;
    for(i = 0; i < n; i++) mean += x[i];
    mean /= n;
    for(i = 0; i < n; i++) var += (x[i] - mean) * (x[i] - mean);
    var /= n;
    float std = sqrtf(var + 1e-5f);
    for(i = 0; i < n; i++) x[i] = (x[i] - mean) / std * w[i] + b[i];
}

void gelu(float *x, int n) {
    int i;
    for(i = 0; i < n; i++) {
        float v = x[i];
        x[i] = 0.5f * v * (1.0f + tanhf(0.797885f * (v + 0.044715f * v * v * v)));
    }
}

void softmax(float *x, int n) {
    float max_val = x[0];
    int i;
    for(i = 1; i < n; i++) if(x[i] > max_val) max_val = x[i];
    float sum = 0;
    for(i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for(i = 0; i < n; i++) x[i] /= sum;
}

typedef struct {
    int vocab_size;
    int n_embd;
    int n_layer;
    float *token_embd;  // vocab x embd (row-major, each row is a weight vector)
    float **ln1_w, **ln1_b;
    float **qkv_w;      // layer x (3*embd*embd), stored as embd rows of 3*embd
    float **qkv_b;
    float **proj_w;     // layer x (embd*embd), stored as embd rows of embd
    float **proj_b;
    float **ln2_w, **ln2_b;
    float **mlp_w;      // layer x (4*embd*embd)
    float **mlp_b;
    float **mlp_proj_w;
    float **mlp_proj_b;
    float *ln_f_w, *ln_f_b;
    float *lm_head_w;   // vocab x embd (row-major)
} Model;

void forward_streaming(Model *m, int token, float *logits) {
    int i, j, l;
    int n_embd = m->n_embd;

    // Embedding (just copy the row)
    float *x = malloc(n_embd * sizeof(float));
    memcpy(x, m->token_embd + token * n_embd, n_embd * sizeof(float));

    float *residual = malloc(n_embd * sizeof(float));
    float *qkv = malloc(3 * n_embd * sizeof(float));
    float *temp = malloc(4 * n_embd * sizeof(float));

    // Layers
    for(l = 0; l < m->n_layer; l++) {
        memcpy(residual, x, n_embd * sizeof(float));
        layer_norm(x, m->ln1_w[l], m->ln1_b[l], n_embd);

        // QKV projection - stream through dot products!
        for(i = 0; i < 3 * n_embd; i++) {
            qkv[i] = dot_product(x, m->qkv_w[l] + i * n_embd, n_embd);
            qkv[i] += m->qkv_b[l][i];
        }

        // Simplified attention (single token)
        float *q = qkv;
        float *k = qkv + n_embd;
        float *v = qkv + 2 * n_embd;

        float score = dot_product(q, k, n_embd) / sqrtf(n_embd);
        for(i = 0; i < n_embd; i++) temp[i] = v[i] * score;

        // Attention projection - stream!
        for(i = 0; i < n_embd; i++) {
            x[i] = dot_product(temp, m->proj_w[l] + i * n_embd, n_embd);
            x[i] += m->proj_b[l][i] + residual[i];
        }

        // MLP
        memcpy(residual, x, n_embd * sizeof(float));
        layer_norm(x, m->ln2_w[l], m->ln2_b[l], n_embd);

        // MLP up - stream!
        for(i = 0; i < 4 * n_embd; i++) {
            temp[i] = dot_product(x, m->mlp_w[l] + i * n_embd, n_embd);
            temp[i] += m->mlp_b[l][i];
        }
        gelu(temp, 4 * n_embd);

        // MLP down - stream!
        for(i = 0; i < n_embd; i++) {
            x[i] = dot_product(temp, m->mlp_proj_w[l] + i * (4 * n_embd), 4 * n_embd);
            x[i] += m->mlp_proj_b[l][i] + residual[i];
        }
    }

    // Final
    layer_norm(x, m->ln_f_w, m->ln_f_b, n_embd);

    // LM head - stream through vocabulary!
    for(i = 0; i < m->vocab_size; i++) {
        logits[i] = dot_product(x, m->lm_head_w + i * n_embd, n_embd);
    }

    free(x); free(residual); free(qkv); free(temp);
}

int main() {
    printf("\\n");
    printf("╔════════════════════════════════════════════╗\\n");
    printf("║  PowerPC G4 LLM - Streaming Dot Products  ║\\n");
    printf("║  Text → AltiVec vmaddfp → Text            ║\\n");
    printf("╚════════════════════════════════════════════╝\\n");
    printf("\\n");

    // Test model
    Model m;
    m.vocab_size = 100;
    m.n_embd = 128;
    m.n_layer = 4;

    long params = m.vocab_size * m.n_embd;
    params += m.n_layer * (
        m.n_embd * 2 + 3 * m.n_embd * m.n_embd + 3 * m.n_embd +
        m.n_embd * m.n_embd + m.n_embd +
        m.n_embd * 2 + 4 * m.n_embd * m.n_embd + 4 * m.n_embd +
        m.n_embd * 4 * m.n_embd + m.n_embd
    );
    params += m.n_embd * 2 + m.vocab_size * m.n_embd;

    printf("Model Configuration:\\n");
    printf("  Vocabulary: %d tokens\\n", m.vocab_size);
    printf("  Embedding: %d dimensions\\n", m.n_embd);
    printf("  Layers: %d\\n", m.n_layer);
    printf("  Parameters: %ld (~%.1fM)\\n\\n", params, params / 1e6);

    // Allocate with random weights
    m.token_embd = malloc(m.vocab_size * m.n_embd * sizeof(float));
    m.lm_head_w = malloc(m.vocab_size * m.n_embd * sizeof(float));
    m.ln_f_w = malloc(m.n_embd * sizeof(float));
    m.ln_f_b = malloc(m.n_embd * sizeof(float));

    int i, j, l;
    for(i = 0; i < m.vocab_size * m.n_embd; i++) {
        m.token_embd[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
        m.lm_head_w[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
    }
    for(i = 0; i < m.n_embd; i++) {
        m.ln_f_w[i] = 1.0f;
        m.ln_f_b[i] = 0.0f;
    }

    m.ln1_w = malloc(m.n_layer * sizeof(float*));
    m.ln1_b = malloc(m.n_layer * sizeof(float*));
    m.ln2_w = malloc(m.n_layer * sizeof(float*));
    m.ln2_b = malloc(m.n_layer * sizeof(float*));
    m.qkv_w = malloc(m.n_layer * sizeof(float*));
    m.qkv_b = malloc(m.n_layer * sizeof(float*));
    m.proj_w = malloc(m.n_layer * sizeof(float*));
    m.proj_b = malloc(m.n_layer * sizeof(float*));
    m.mlp_w = malloc(m.n_layer * sizeof(float*));
    m.mlp_b = malloc(m.n_layer * sizeof(float*));
    m.mlp_proj_w = malloc(m.n_layer * sizeof(float*));
    m.mlp_proj_b = malloc(m.n_layer * sizeof(float*));

    for(l = 0; l < m.n_layer; l++) {
        m.ln1_w[l] = malloc(m.n_embd * sizeof(float));
        m.ln1_b[l] = malloc(m.n_embd * sizeof(float));
        m.ln2_w[l] = malloc(m.n_embd * sizeof(float));
        m.ln2_b[l] = malloc(m.n_embd * sizeof(float));
        for(i = 0; i < m.n_embd; i++) {
            m.ln1_w[l][i] = 1.0f;
            m.ln1_b[l][i] = 0.0f;
            m.ln2_w[l][i] = 1.0f;
            m.ln2_b[l][i] = 0.0f;
        }

        m.qkv_w[l] = malloc(3 * m.n_embd * m.n_embd * sizeof(float));
        m.qkv_b[l] = malloc(3 * m.n_embd * sizeof(float));
        m.proj_w[l] = malloc(m.n_embd * m.n_embd * sizeof(float));
        m.proj_b[l] = malloc(m.n_embd * sizeof(float));
        m.mlp_w[l] = malloc(4 * m.n_embd * m.n_embd * sizeof(float));
        m.mlp_b[l] = malloc(4 * m.n_embd * sizeof(float));
        m.mlp_proj_w[l] = malloc(m.n_embd * 4 * m.n_embd * sizeof(float));
        m.mlp_proj_b[l] = malloc(m.n_embd * sizeof(float));

        for(i = 0; i < 3 * m.n_embd * m.n_embd; i++)
            m.qkv_w[l][i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
        for(i = 0; i < m.n_embd * m.n_embd; i++)
            m.proj_w[l][i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
        for(i = 0; i < 4 * m.n_embd * m.n_embd; i++) {
            m.mlp_w[l][i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
            m.mlp_proj_w[l][i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
        }
        for(i = 0; i < 3 * m.n_embd; i++) m.qkv_b[l][i] = 0;
        for(i = 0; i < m.n_embd; i++) m.proj_b[l][i] = 0;
        for(i = 0; i < 4 * m.n_embd; i++) m.mlp_b[l][i] = 0;
        for(i = 0; i < m.n_embd; i++) m.mlp_proj_b[l][i] = 0;
    }

    // Benchmark
    printf("Running inference (streaming dot products)...\\n\\n");
    float *logits = malloc(m.vocab_size * sizeof(float));

    struct timeval start, end;
    gettimeofday(&start, NULL);

    int token = 0;
    int n_tokens = 50;
    for(i = 0; i < n_tokens; i++) {
        forward_streaming(&m, token, logits);
        softmax(logits, m.vocab_size);

        // Sample
        float r = (float)rand() / RAND_MAX, c = 0;
        for(j = 0; j < m.vocab_size; j++) {
            c += logits[j];
            if(r < c) { token = j; break; }
        }
    }

    gettimeofday(&end, NULL);
    unsigned long us = (end.tv_sec - start.tv_sec) * 1000000UL + (end.tv_usec - start.tv_usec);

    printf("Performance:\\n");
    printf("  Tokens generated: %d\\n", n_tokens);
    printf("  Time: %.2f ms\\n", us / 1000.0f);
    printf("  Speed: %.1f tokens/sec\\n\\n", n_tokens / (us / 1000000.0f));

    printf("Architecture Analysis:\\n");
    printf("  - Each forward pass = %d dot products\\n",
           m.n_layer * (3 * m.n_embd + m.n_embd + 4 * m.n_embd + m.n_embd) + m.vocab_size);
    printf("  - AltiVec vmaddfp: 4 floats per cycle\\n");
    printf("  - Every weight access streams through SIMD pipeline\\n");
    printf("  - Binary transform: Text→Embedding→Dot Products→Logits→Text\\n\\n");

    printf("╔════════════════════════════════════════════╗\\n");
    printf("║  🔥 TRANSFORMER ON 1999 HARDWARE 🔥       ║\\n");
    printf("║  AltiVec dot products = Perfect match!    ║\\n");
    printf("╚════════════════════════════════════════════╝\\n\\n");

    return 0;
}
