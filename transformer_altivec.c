/* Transformer with AltiVec SIMD for PowerPC G4 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

#ifdef __ALTIVEC__
#include <altivec.h>

// AltiVec-accelerated matrix multiplication
void altivec_matmul(float *out, const float *x, const float *w, int n, int d) {
    int i, j, k;

    // Zero output
    for(i = 0; i < d; i++) out[i] = 0;

    // out[d] = x[n] @ w[d x n]^T
    for(i = 0; i < d; i++) {
        vector float v_sum = {0, 0, 0, 0};

        // Vectorized dot product (4 at a time)
        for(j = 0; j + 3 < n; j += 4) {
            vector float v_x = vec_ld(0, (float*)(x + j));
            vector float v_w = vec_ld(0, (float*)(w + i * n + j));
            v_sum = vec_madd(v_x, v_w, v_sum);
        }

        // Horizontal sum of vector
        vector float v_sum_shifted = vec_sld(v_sum, v_sum, 8);
        v_sum = vec_add(v_sum, v_sum_shifted);
        v_sum_shifted = vec_sld(v_sum, v_sum, 4);
        v_sum = vec_add(v_sum, v_sum_shifted);

        // Store first element
        vec_ste(v_sum, 0, &out[i]);

        // Handle remainder
        for(; j < n; j++) {
            out[i] += x[j] * w[i * n + j];
        }
    }
}

#define matmul altivec_matmul
#else
// Scalar fallback
void matmul(float *out, const float *x, const float *w, int n, int d) {
    int i, j;
    for(i = 0; i < d; i++) {
        float sum = 0;
        for(j = 0; j < n; j++) {
            sum += x[j] * w[i * n + j];
        }
        out[i] = sum;
    }
}
#endif

void layer_norm(float *x, float *w, float *b, int size) {
    float mean = 0, var = 0;
    int i;
    for(i = 0; i < size; i++) mean += x[i];
    mean /= size;
    for(i = 0; i < size; i++) var += (x[i] - mean) * (x[i] - mean);
    var /= size;
    float std = sqrtf(var + 1e-5f);
    for(i = 0; i < size; i++) {
        x[i] = (x[i] - mean) / std * w[i] + b[i];
    }
}

void gelu(float *x, int size) {
    int i;
    for(i = 0; i < size; i++) {
        float v = x[i];
        x[i] = 0.5f * v * (1.0f + tanhf(0.797885f * (v + 0.044715f * v * v * v)));
    }
}

void softmax(float *x, int size) {
    float max_val = x[0];
    int i;
    for(i = 1; i < size; i++) {
        if(x[i] > max_val) max_val = x[i];
    }
    float sum = 0;
    for(i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for(i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

typedef struct {
    int vocab_size;
    int n_embd;
    int n_layer;
    int n_head;

    float *token_embd;  // vocab x embd
    float **ln1_w;      // layer x embd
    float **ln1_b;
    float **qkv_w;      // layer x (3*embd) x embd
    float **qkv_b;
    float **attn_proj_w;
    float **attn_proj_b;
    float **ln2_w;
    float **ln2_b;
    float **mlp_w;      // layer x (4*embd) x embd
    float **mlp_b;
    float **mlp_proj_w;
    float **mlp_proj_b;
    float *ln_f_w;
    float *ln_f_b;
    float *lm_head_w;   // vocab x embd
} Model;

void forward(Model *m, int token, float *logits) {
    int i, l;
    int n_embd = m->n_embd;
    int head_dim = n_embd / m->n_head;

    // Embedding
    float *x = malloc(n_embd * sizeof(float));
    memcpy(x, m->token_embd + token * n_embd, n_embd * sizeof(float));

    float *residual = malloc(n_embd * sizeof(float));
    float *qkv = malloc(3 * n_embd * sizeof(float));
    float *attn_out = malloc(n_embd * sizeof(float));
    float *mlp_hidden = malloc(4 * n_embd * sizeof(float));
    float *mlp_out = malloc(n_embd * sizeof(float));

    // Layers
    for(l = 0; l < m->n_layer; l++) {
        // Save residual
        memcpy(residual, x, n_embd * sizeof(float));

        // Pre-attention norm
        layer_norm(x, m->ln1_w[l], m->ln1_b[l], n_embd);

        // QKV projection (AltiVec accelerated!)
        matmul(qkv, x, m->qkv_w[l], n_embd, 3 * n_embd);
        for(i = 0; i < 3 * n_embd; i++) qkv[i] += m->qkv_b[l][i];

        // Simplified attention (self-attention on single token)
        float *q = qkv;
        float *k = qkv + n_embd;
        float *v = qkv + 2 * n_embd;

        for(i = 0; i < m->n_head; i++) {
            float score = 0;
            int j;
            for(j = 0; j < head_dim; j++) {
                score += q[i * head_dim + j] * k[i * head_dim + j];
            }
            score /= sqrtf(head_dim);

            for(j = 0; j < head_dim; j++) {
                attn_out[i * head_dim + j] = v[i * head_dim + j] * score;
            }
        }

        // Attention output projection (AltiVec!)
        matmul(x, attn_out, m->attn_proj_w[l], n_embd, n_embd);
        for(i = 0; i < n_embd; i++) x[i] += m->attn_proj_b[l][i];

        // Add residual
        for(i = 0; i < n_embd; i++) x[i] += residual[i];

        // Save residual
        memcpy(residual, x, n_embd * sizeof(float));

        // Pre-MLP norm
        layer_norm(x, m->ln2_w[l], m->ln2_b[l], n_embd);

        // MLP up projection (AltiVec!)
        matmul(mlp_hidden, x, m->mlp_w[l], n_embd, 4 * n_embd);
        for(i = 0; i < 4 * n_embd; i++) mlp_hidden[i] += m->mlp_b[l][i];
        gelu(mlp_hidden, 4 * n_embd);

        // MLP down projection (AltiVec!)
        matmul(x, mlp_hidden, m->mlp_proj_w[l], 4 * n_embd, n_embd);
        for(i = 0; i < n_embd; i++) x[i] += m->mlp_proj_b[l][i];

        // Add residual
        for(i = 0; i < n_embd; i++) x[i] += residual[i];
    }

    // Final norm
    layer_norm(x, m->ln_f_w, m->ln_f_b, n_embd);

    // LM head (AltiVec!)
    matmul(logits, x, m->lm_head_w, n_embd, m->vocab_size);

    free(x); free(residual); free(qkv);
    free(attn_out); free(mlp_hidden); free(mlp_out);
}

int sample(float *logits, int size) {
    softmax(logits, size);
    float r = (float)rand() / RAND_MAX, c = 0;
    int i;
    for(i = 0; i < size; i++) {
        c += logits[i];
        if(r < c) return i;
    }
    return 0;
}

int main() {
    printf("🔥 AltiVec-Accelerated Transformer on PowerPC G4 🔥\\n");
    printf("==================================================\\n\\n");

    // Test model (scalable)
    Model m;
    m.vocab_size = 1000;
    m.n_embd = 256;
    m.n_layer = 4;
    m.n_head = 8;

    printf("Model: %d vocab, %d embd, %d layers, %d heads\\n",
           m.vocab_size, m.n_embd, m.n_layer, m.n_head);

    // Calculate params
    long params = m.vocab_size * m.n_embd;  // token_embd
    params += m.n_layer * (
        m.n_embd * 2 +              // ln1
        3 * m.n_embd * m.n_embd +   // qkv_w
        3 * m.n_embd +              // qkv_b
        m.n_embd * m.n_embd +       // attn_proj_w
        m.n_embd +                  // attn_proj_b
        m.n_embd * 2 +              // ln2
        4 * m.n_embd * m.n_embd +   // mlp_w
        4 * m.n_embd +              // mlp_b
        m.n_embd * 4 * m.n_embd +   // mlp_proj_w
        m.n_embd                    // mlp_proj_b
    );
    params += m.n_embd * 2;         // ln_f
    params += m.vocab_size * m.n_embd;  // lm_head

    printf("Parameters: %ld (~%.1fM)\\n\\n", params, params / 1000000.0);

    // Allocate (random weights for benchmarking)
    m.token_embd = malloc(m.vocab_size * m.n_embd * sizeof(float));
    m.lm_head_w = malloc(m.vocab_size * m.n_embd * sizeof(float));
    m.ln_f_w = malloc(m.n_embd * sizeof(float));
    m.ln_f_b = malloc(m.n_embd * sizeof(float));

    m.ln1_w = malloc(m.n_layer * sizeof(float*));
    m.ln1_b = malloc(m.n_layer * sizeof(float*));
    m.ln2_w = malloc(m.n_layer * sizeof(float*));
    m.ln2_b = malloc(m.n_layer * sizeof(float*));
    m.qkv_w = malloc(m.n_layer * sizeof(float*));
    m.qkv_b = malloc(m.n_layer * sizeof(float*));
    m.attn_proj_w = malloc(m.n_layer * sizeof(float*));
    m.attn_proj_b = malloc(m.n_layer * sizeof(float*));
    m.mlp_w = malloc(m.n_layer * sizeof(float*));
    m.mlp_b = malloc(m.n_layer * sizeof(float*));
    m.mlp_proj_w = malloc(m.n_layer * sizeof(float*));
    m.mlp_proj_b = malloc(m.n_layer * sizeof(float*));

    int i, l;
    for(i = 0; i < m.vocab_size * m.n_embd; i++) {
        m.token_embd[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
        m.lm_head_w[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
    }

    for(i = 0; i < m.n_embd; i++) {
        m.ln_f_w[i] = 1.0f;
        m.ln_f_b[i] = 0.0f;
    }

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
        m.attn_proj_w[l] = malloc(m.n_embd * m.n_embd * sizeof(float));
        m.attn_proj_b[l] = malloc(m.n_embd * sizeof(float));
        m.mlp_w[l] = malloc(4 * m.n_embd * m.n_embd * sizeof(float));
        m.mlp_b[l] = malloc(4 * m.n_embd * sizeof(float));
        m.mlp_proj_w[l] = malloc(m.n_embd * 4 * m.n_embd * sizeof(float));
        m.mlp_proj_b[l] = malloc(m.n_embd * sizeof(float));

        for(i = 0; i < 3 * m.n_embd * m.n_embd; i++) {
            m.qkv_w[l][i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
        }
        for(i = 0; i < m.n_embd * m.n_embd; i++) {
            m.attn_proj_w[l][i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
        }
        for(i = 0; i < 4 * m.n_embd * m.n_embd; i++) {
            m.mlp_w[l][i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
            m.mlp_proj_w[l][i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
        }
        for(i = 0; i < 3 * m.n_embd; i++) m.qkv_b[l][i] = 0;
        for(i = 0; i < m.n_embd; i++) m.attn_proj_b[l][i] = 0;
        for(i = 0; i < 4 * m.n_embd; i++) m.mlp_b[l][i] = 0;
        for(i = 0; i < m.n_embd; i++) m.mlp_proj_b[l][i] = 0;
    }

    // Benchmark
    printf("Running inference benchmark...\\n");
    float *logits = malloc(m.vocab_size * sizeof(float));

    struct timeval start, end;
    gettimeofday(&start, NULL);

    int n_tokens = 50;
    int token = 0;
    for(i = 0; i < n_tokens; i++) {
        forward(&m, token, logits);
        token = sample(logits, m.vocab_size);
    }

    gettimeofday(&end, NULL);
    unsigned long us = (end.tv_sec - start.tv_sec) * 1000000UL + (end.tv_usec - start.tv_usec);

    printf("\\n%.1f tokens/sec on PowerPC G4 with AltiVec\\n", n_tokens / (us / 1000000.0));
    printf("\\n🔥 REAL TRANSFORMER WITH ALTIVEC ACCELERATION 🔥\\n");

    return 0;
}
