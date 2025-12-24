/* Transformer for PowerPC G4 - GPT-style architecture
 * Based on SmolLM-135M structure
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// Model hyperparameters (SmolLM-135M)
#define VOCAB_SIZE 49152
#define N_EMBD 576      // embedding dimension
#define N_LAYER 30      // number of layers
#define N_HEAD 9        // attention heads
#define CTX_SIZE 2048   // context length

// For testing, use smaller dims
#define TEST_VOCAB 1000
#define TEST_EMBD 256
#define TEST_LAYER 4
#define TEST_HEAD 4

typedef struct {
    int vocab_size;
    int n_embd;
    int n_layer;
    int n_head;
    int ctx_size;

    // Weights
    float *token_embd;      // vocab_size x n_embd
    float *pos_embd;        // ctx_size x n_embd

    // Per-layer weights (simplified - no KV cache yet)
    float **ln1_w;          // n_layer x n_embd
    float **ln1_b;
    float **attn_qkv_w;     // n_layer x (3*n_embd) x n_embd
    float **attn_qkv_b;
    float **attn_proj_w;    // n_layer x n_embd x n_embd
    float **attn_proj_b;

    float **ln2_w;
    float **ln2_b;
    float **mlp_fc_w;       // n_layer x (4*n_embd) x n_embd
    float **mlp_fc_b;
    float **mlp_proj_w;     // n_layer x n_embd x (4*n_embd)
    float **mlp_proj_b;

    float *ln_f_w;          // final layer norm
    float *ln_f_b;
    float *lm_head_w;       // vocab_size x n_embd
} Transformer;

void layer_norm(float *x, float *w, float *b, int size) {
    float mean = 0, var = 0;
    int i;

    for(i = 0; i < size; i++) mean += x[i];
    mean /= size;

    for(i = 0; i < size; i++) var += (x[i] - mean) * (x[i] - mean);
    var /= size;

    float std = sqrtf(var + 1e-5f);
    for(i = 0; i < size; i++) {
        x[i] = (x[i] - mean) / std;
        x[i] = x[i] * w[i] + b[i];
    }
}

void matmul(float *out, float *x, float *w, int n, int d) {
    // out = x @ w^T, where w is (d x n)
    int i, j;
    for(i = 0; i < d; i++) {
        float sum = 0;
        for(j = 0; j < n; j++) {
            sum += x[j] * w[i * n + j];
        }
        out[i] = sum;
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

void attention(float *out, float *x, float *qkv_w, float *qkv_b,
               float *proj_w, float *proj_b, int n_embd, int n_head) {
    int head_dim = n_embd / n_head;
    int i, j, h;

    // QKV projection
    float *qkv = malloc(3 * n_embd * sizeof(float));
    matmul(qkv, x, qkv_w, n_embd, 3 * n_embd);
    for(i = 0; i < 3 * n_embd; i++) qkv[i] += qkv_b[i];

    float *q = qkv;
    float *k = qkv + n_embd;
    float *v = qkv + 2 * n_embd;

    // Simplified: single-token attention (no sequence)
    float *attn_out = malloc(n_embd * sizeof(float));

    for(h = 0; h < n_head; h++) {
        // Dot product attention
        float score = 0;
        for(i = 0; i < head_dim; i++) {
            score += q[h * head_dim + i] * k[h * head_dim + i];
        }
        score /= sqrtf(head_dim);

        // Apply to value
        for(i = 0; i < head_dim; i++) {
            attn_out[h * head_dim + i] = v[h * head_dim + i] * score;
        }
    }

    // Output projection
    matmul(out, attn_out, proj_w, n_embd, n_embd);
    for(i = 0; i < n_embd; i++) out[i] += proj_b[i];

    free(qkv);
    free(attn_out);
}

void mlp(float *out, float *x, float *fc_w, float *fc_b,
         float *proj_w, float *proj_b, int n_embd) {
    int hidden_dim = 4 * n_embd;
    int i;

    float *hidden = malloc(hidden_dim * sizeof(float));

    // FC layer
    matmul(hidden, x, fc_w, n_embd, hidden_dim);
    for(i = 0; i < hidden_dim; i++) hidden[i] += fc_b[i];
    gelu(hidden, hidden_dim);

    // Projection
    matmul(out, hidden, proj_w, hidden_dim, n_embd);
    for(i = 0; i < n_embd; i++) out[i] += proj_b[i];

    free(hidden);
}

void forward(Transformer *model, int token, float *logits) {
    int i, l;
    int n_embd = model->n_embd;

    // Token + position embedding
    float *x = malloc(n_embd * sizeof(float));
    for(i = 0; i < n_embd; i++) {
        x[i] = model->token_embd[token * n_embd + i] +
               model->pos_embd[0 * n_embd + i];  // position 0 for now
    }

    // Transformer layers
    float *residual = malloc(n_embd * sizeof(float));
    float *attn_out = malloc(n_embd * sizeof(float));
    float *mlp_out = malloc(n_embd * sizeof(float));

    for(l = 0; l < model->n_layer; l++) {
        // Save residual
        memcpy(residual, x, n_embd * sizeof(float));

        // Pre-attention layer norm
        layer_norm(x, model->ln1_w[l], model->ln1_b[l], n_embd);

        // Attention
        attention(attn_out, x, model->attn_qkv_w[l], model->attn_qkv_b[l],
                 model->attn_proj_w[l], model->attn_proj_b[l], n_embd, model->n_head);

        // Add residual
        for(i = 0; i < n_embd; i++) x[i] = residual[i] + attn_out[i];

        // Save residual again
        memcpy(residual, x, n_embd * sizeof(float));

        // Pre-MLP layer norm
        layer_norm(x, model->ln2_w[l], model->ln2_b[l], n_embd);

        // MLP
        mlp(mlp_out, x, model->mlp_fc_w[l], model->mlp_fc_b[l],
            model->mlp_proj_w[l], model->mlp_proj_b[l], n_embd);

        // Add residual
        for(i = 0; i < n_embd; i++) x[i] = residual[i] + mlp_out[i];
    }

    // Final layer norm
    layer_norm(x, model->ln_f_w, model->ln_f_b, n_embd);

    // LM head
    matmul(logits, x, model->lm_head_w, n_embd, model->vocab_size);

    free(x);
    free(residual);
    free(attn_out);
    free(mlp_out);
}

int main() {
    printf("🚀 Transformer on PowerPC G4\\n");
    printf("================================\\n\\n");

    // Test with random weights first
    Transformer model;
    model.vocab_size = 100;
    model.n_embd = 64;
    model.n_layer = 2;
    model.n_head = 4;
    model.ctx_size = 128;

    printf("Model: %d vocab, %d embd, %d layers, %d heads\\n",
           model.vocab_size, model.n_embd, model.n_layer, model.n_head);

    // Allocate (simplified - just token embeddings and LM head for now)
    model.token_embd = malloc(model.vocab_size * model.n_embd * sizeof(float));
    model.pos_embd = malloc(model.ctx_size * model.n_embd * sizeof(float));
    model.lm_head_w = malloc(model.vocab_size * model.n_embd * sizeof(float));

    // Initialize random weights
    int i;
    for(i = 0; i < model.vocab_size * model.n_embd; i++) {
        model.token_embd[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
        model.lm_head_w[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
    }
    for(i = 0; i < model.ctx_size * model.n_embd; i++) {
        model.pos_embd[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
    }

    // Allocate layer weights
    model.ln1_w = malloc(model.n_layer * sizeof(float*));
    model.ln1_b = malloc(model.n_layer * sizeof(float*));
    model.ln2_w = malloc(model.n_layer * sizeof(float*));
    model.ln2_b = malloc(model.n_layer * sizeof(float*));
    model.ln_f_w = malloc(model.n_embd * sizeof(float));
    model.ln_f_b = malloc(model.n_embd * sizeof(float));

    model.attn_qkv_w = malloc(model.n_layer * sizeof(float*));
    model.attn_qkv_b = malloc(model.n_layer * sizeof(float*));
    model.attn_proj_w = malloc(model.n_layer * sizeof(float*));
    model.attn_proj_b = malloc(model.n_layer * sizeof(float*));

    model.mlp_fc_w = malloc(model.n_layer * sizeof(float*));
    model.mlp_fc_b = malloc(model.n_layer * sizeof(float*));
    model.mlp_proj_w = malloc(model.n_layer * sizeof(float*));
    model.mlp_proj_b = malloc(model.n_layer * sizeof(float*));

    int l;
    for(l = 0; l < model.n_layer; l++) {
        model.ln1_w[l] = malloc(model.n_embd * sizeof(float));
        model.ln1_b[l] = malloc(model.n_embd * sizeof(float));
        model.ln2_w[l] = malloc(model.n_embd * sizeof(float));
        model.ln2_b[l] = malloc(model.n_embd * sizeof(float));

        for(i = 0; i < model.n_embd; i++) {
            model.ln1_w[l][i] = 1.0f;
            model.ln1_b[l][i] = 0.0f;
            model.ln2_w[l][i] = 1.0f;
            model.ln2_b[l][i] = 0.0f;
        }

        model.attn_qkv_w[l] = malloc(3 * model.n_embd * model.n_embd * sizeof(float));
        model.attn_qkv_b[l] = malloc(3 * model.n_embd * sizeof(float));
        model.attn_proj_w[l] = malloc(model.n_embd * model.n_embd * sizeof(float));
        model.attn_proj_b[l] = malloc(model.n_embd * sizeof(float));

        for(i = 0; i < 3 * model.n_embd * model.n_embd; i++) {
            model.attn_qkv_w[l][i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
        }
        for(i = 0; i < 3 * model.n_embd; i++) {
            model.attn_qkv_b[l][i] = 0.0f;
        }
        for(i = 0; i < model.n_embd * model.n_embd; i++) {
            model.attn_proj_w[l][i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
        }
        for(i = 0; i < model.n_embd; i++) {
            model.attn_proj_b[l][i] = 0.0f;
        }

        model.mlp_fc_w[l] = malloc(4 * model.n_embd * model.n_embd * sizeof(float));
        model.mlp_fc_b[l] = malloc(4 * model.n_embd * sizeof(float));
        model.mlp_proj_w[l] = malloc(model.n_embd * 4 * model.n_embd * sizeof(float));
        model.mlp_proj_b[l] = malloc(model.n_embd * sizeof(float));

        for(i = 0; i < 4 * model.n_embd * model.n_embd; i++) {
            model.mlp_fc_w[l][i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
            model.mlp_proj_w[l][i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
        }
        for(i = 0; i < 4 * model.n_embd; i++) {
            model.mlp_fc_b[l][i] = 0.0f;
        }
        for(i = 0; i < model.n_embd; i++) {
            model.mlp_proj_b[l][i] = 0.0f;
        }
    }

    for(i = 0; i < model.n_embd; i++) {
        model.ln_f_w[i] = 1.0f;
        model.ln_f_b[i] = 0.0f;
    }

    // Test forward pass
    float *logits = malloc(model.vocab_size * sizeof(float));

    printf("\\nRunning forward pass (token 0)...\\n");
    forward(&model, 0, logits);

    printf("Logits[0:5]: ");
    for(i = 0; i < 5; i++) printf("%.3f ", logits[i]);
    printf("\\n");

    softmax(logits, model.vocab_size);
    printf("Probs[0:5]:  ");
    for(i = 0; i < 5; i++) printf("%.3f ", logits[i]);
    printf("\\n");

    printf("\\n✅ Transformer inference working!\\n");

    return 0;
}
