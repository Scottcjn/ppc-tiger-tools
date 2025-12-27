/*
 * HRM Inference for PowerPC G5 - Accelerate BLAS Optimized
 * Samsung Hierarchical Reasoning Model for Sudoku
 *
 * Compile with GCC 10:
 *   g++-10 -O3 -mcpu=970 -mtune=970 -ffast-math \
 *          -framework Accelerate -o hrm_inference hrm_inference.cpp
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

// Model configuration
static const int HIDDEN_SIZE = 512;
static const int NUM_HEADS = 8;
static const int HEAD_DIM = 64;
static const int H_LAYERS = 4;
static const int L_LAYERS = 4;
static const int VOCAB_SIZE = 11;
static const int SEQ_LEN = 81;
static const int MAX_STEPS = 16;
static const int MLP_DIM = HIDDEN_SIZE * 3;

struct Weights {
    float* H_init;
    float* L_init;
    float* embed;
    float* lm_head;
    float* qkv_proj[2][4];
    float* o_proj[2][4];
    float* gate_up[2][4];
    float* down_proj[2][4];
};

static float rope_cos[SEQ_LEN][HEAD_DIM];
static float rope_sin[SEQ_LEN][HEAD_DIM];

inline float* alloc_float(size_t count) {
    float* ptr = (float*)malloc(count * sizeof(float));
    memset(ptr, 0, count * sizeof(float));
    return ptr;
}

float* load_weight(const char* weights_dir, const char* name, int* out_size) {
    char fname[512];
    char safe_name[256];

    strncpy(safe_name, name, sizeof(safe_name) - 1);
    safe_name[sizeof(safe_name) - 1] = '\0';
    for (char* p = safe_name; *p; p++) {
        if (*p == '.') *p = '_';
    }

    snprintf(fname, sizeof(fname), "%s/%s.bin", weights_dir, safe_name);

    FILE* f = fopen(fname, "rb");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open %s\n", fname);
        exit(1);
    }

    unsigned char buf[4];
    fread(buf, 1, 4, f);
    int ndim = (buf[0] << 24) | (buf[1] << 16) | (buf[2] << 8) | buf[3];

    int total_size = 1;
    for (int i = 0; i < ndim; i++) {
        fread(buf, 1, 4, f);
        int dim = (buf[0] << 24) | (buf[1] << 16) | (buf[2] << 8) | buf[3];
        total_size *= dim;
    }

    float* data = alloc_float(total_size);
    unsigned char* raw = (unsigned char*)malloc(total_size * 4);
    fread(raw, 4, total_size, f);
    fclose(f);

    for (int i = 0; i < total_size; i++) {
        unsigned char* p = raw + i * 4;
        unsigned int bits = (p[0] << 24) | (p[1] << 16) | (p[2] << 8) | p[3];
        memcpy(&data[i], &bits, 4);
    }

    free(raw);
    *out_size = total_size;
    return data;
}

void init_rope() {
    for (int pos = 0; pos < SEQ_LEN; pos++) {
        for (int i = 0; i < HEAD_DIM / 2; i++) {
            float freq = 1.0f / powf(10000.0f, (float)(2 * i) / HEAD_DIM);
            float angle = pos * freq;
            rope_cos[pos][i] = cosf(angle);
            rope_cos[pos][i + HEAD_DIM/2] = cosf(angle);
            rope_sin[pos][i] = sinf(angle);
            rope_sin[pos][i + HEAD_DIM/2] = sinf(angle);
        }
    }
}

Weights load_all_weights(const char* weights_dir) {
    Weights w;
    int size;

    printf("Loading weights from %s\n", weights_dir);

    w.H_init = load_weight(weights_dir, "H_init", &size);
    w.L_init = load_weight(weights_dir, "L_init", &size);
    w.embed = load_weight(weights_dir, "embed_tokens_embedding_weight", &size);
    w.lm_head = load_weight(weights_dir, "lm_head_weight", &size);

    const char* level_names[2] = {"H_level", "L_level"};

    for (int lvl = 0; lvl < 2; lvl++) {
        for (int layer = 0; layer < 4; layer++) {
            char name[128];

            snprintf(name, sizeof(name), "%s_layers_%d_self_attn_qkv_proj_weight",
                     level_names[lvl], layer);
            w.qkv_proj[lvl][layer] = load_weight(weights_dir, name, &size);

            snprintf(name, sizeof(name), "%s_layers_%d_self_attn_o_proj_weight",
                     level_names[lvl], layer);
            w.o_proj[lvl][layer] = load_weight(weights_dir, name, &size);

            snprintf(name, sizeof(name), "%s_layers_%d_mlp_gate_up_proj_weight",
                     level_names[lvl], layer);
            w.gate_up[lvl][layer] = load_weight(weights_dir, name, &size);

            snprintf(name, sizeof(name), "%s_layers_%d_mlp_down_proj_weight",
                     level_names[lvl], layer);
            w.down_proj[lvl][layer] = load_weight(weights_dir, name, &size);
        }
    }

    printf("Loaded all weights\n");
    return w;
}

void rms_norm(float* out, const float* x, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += x[i] * x[i];
    }
    float rms = 1.0f / sqrtf(sum / size + 1e-5f);
    for (int i = 0; i < size; i++) {
        out[i] = x[i] * rms;
    }
}

void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < size; i++) {
        x[i] *= inv_sum;
    }
}

// Matrix multiply using Accelerate BLAS - THIS IS THE KEY OPTIMIZATION
void matmul(float* C, const float* A, const float* B, int M, int N, int K) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K,
                1.0f, A, K, B, K,
                0.0f, C, N);
}

void add_vectors(float* out, const float* a, const float* b, int size) {
    for (int i = 0; i < size; i++) {
        out[i] = a[i] + b[i];
    }
}

void apply_rope(float* q, float* k, int seq_len, int num_heads, int head_dim) {
    for (int pos = 0; pos < seq_len; pos++) {
        for (int h = 0; h < num_heads; h++) {
            float* q_head = q + pos * num_heads * head_dim + h * head_dim;
            float* k_head = k + pos * num_heads * head_dim + h * head_dim;

            for (int i = 0; i < head_dim / 2; i++) {
                float cos_val = rope_cos[pos][i];
                float sin_val = rope_sin[pos][i];

                float q0 = q_head[i];
                float q1 = q_head[i + head_dim/2];
                q_head[i] = q0 * cos_val - q1 * sin_val;
                q_head[i + head_dim/2] = q1 * cos_val + q0 * sin_val;

                float k0 = k_head[i];
                float k1 = k_head[i + head_dim/2];
                k_head[i] = k0 * cos_val - k1 * sin_val;
                k_head[i + head_dim/2] = k1 * cos_val + k0 * sin_val;
            }
        }
    }
}

float dot_product(const float* a, const float* b, int size) {
    float result = 0.0f;
    for (int i = 0; i < size; i++) {
        result += a[i] * b[i];
    }
    return result;
}

void attention(float* out, const float* x, const float* qkv_w, const float* o_w,
               float* q_buf, float* k_buf, float* v_buf, float* attn_buf, float* head_out) {

    float* qkv = alloc_float(SEQ_LEN * HIDDEN_SIZE * 3);
    matmul(qkv, x, qkv_w, SEQ_LEN, HIDDEN_SIZE * 3, HIDDEN_SIZE);

    for (int pos = 0; pos < SEQ_LEN; pos++) {
        for (int h = 0; h < NUM_HEADS; h++) {
            for (int d = 0; d < HEAD_DIM; d++) {
                int src_idx = pos * HIDDEN_SIZE * 3;
                int dst_idx = pos * NUM_HEADS * HEAD_DIM + h * HEAD_DIM + d;

                q_buf[dst_idx] = qkv[src_idx + h * HEAD_DIM + d];
                k_buf[dst_idx] = qkv[src_idx + HIDDEN_SIZE + h * HEAD_DIM + d];
                v_buf[dst_idx] = qkv[src_idx + HIDDEN_SIZE * 2 + h * HEAD_DIM + d];
            }
        }
    }

    apply_rope(q_buf, k_buf, SEQ_LEN, NUM_HEADS, HEAD_DIM);

    float scale = 1.0f / sqrtf((float)HEAD_DIM);

    for (int h = 0; h < NUM_HEADS; h++) {
        for (int i = 0; i < SEQ_LEN; i++) {
            for (int j = 0; j < SEQ_LEN; j++) {
                const float* qi = q_buf + i * NUM_HEADS * HEAD_DIM + h * HEAD_DIM;
                const float* kj = k_buf + j * NUM_HEADS * HEAD_DIM + h * HEAD_DIM;
                float score = dot_product(qi, kj, HEAD_DIM);
                attn_buf[i * SEQ_LEN + j] = score * scale;
            }
            softmax(attn_buf + i * SEQ_LEN, SEQ_LEN);
        }

        for (int i = 0; i < SEQ_LEN; i++) {
            for (int d = 0; d < HEAD_DIM; d++) {
                float val = 0.0f;
                for (int j = 0; j < SEQ_LEN; j++) {
                    val += attn_buf[i * SEQ_LEN + j] *
                           v_buf[j * NUM_HEADS * HEAD_DIM + h * HEAD_DIM + d];
                }
                head_out[i * HIDDEN_SIZE + h * HEAD_DIM + d] = val;
            }
        }
    }

    matmul(out, head_out, o_w, SEQ_LEN, HIDDEN_SIZE, HIDDEN_SIZE);
    free(qkv);
}

void mlp(float* out, const float* x, const float* gate_up_w, const float* down_w) {
    float* gate_up = alloc_float(SEQ_LEN * MLP_DIM * 2);
    matmul(gate_up, x, gate_up_w, SEQ_LEN, MLP_DIM * 2, HIDDEN_SIZE);

    float* hidden = alloc_float(SEQ_LEN * MLP_DIM);
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < MLP_DIM; j++) {
            float gate = gate_up[i * MLP_DIM * 2 + j];
            float up = gate_up[i * MLP_DIM * 2 + MLP_DIM + j];
            float silu_gate = gate / (1.0f + expf(-gate));
            hidden[i * MLP_DIM + j] = silu_gate * up;
        }
    }

    matmul(out, hidden, down_w, SEQ_LEN, HIDDEN_SIZE, MLP_DIM);
    free(gate_up);
    free(hidden);
}

void transformer_block(float* x, int level, int layer, const Weights& w,
                       float* q_buf, float* k_buf, float* v_buf,
                       float* attn_buf, float* head_out, float* temp) {

    attention(temp, x, w.qkv_proj[level][layer], w.o_proj[level][layer],
              q_buf, k_buf, v_buf, attn_buf, head_out);

    for (int i = 0; i < SEQ_LEN; i++) {
        add_vectors(x + i * HIDDEN_SIZE, x + i * HIDDEN_SIZE,
                   temp + i * HIDDEN_SIZE, HIDDEN_SIZE);
        rms_norm(x + i * HIDDEN_SIZE, x + i * HIDDEN_SIZE, HIDDEN_SIZE);
    }

    mlp(temp, x, w.gate_up[level][layer], w.down_proj[level][layer]);

    for (int i = 0; i < SEQ_LEN; i++) {
        add_vectors(x + i * HIDDEN_SIZE, x + i * HIDDEN_SIZE,
                   temp + i * HIDDEN_SIZE, HIDDEN_SIZE);
        rms_norm(x + i * HIDDEN_SIZE, x + i * HIDDEN_SIZE, HIDDEN_SIZE);
    }
}

void forward(float* logits, const int* puzzle, const Weights& w) {
    float* input_emb = alloc_float(SEQ_LEN * HIDDEN_SIZE);
    float* z_h = alloc_float(SEQ_LEN * HIDDEN_SIZE);
    float* z_l = alloc_float(SEQ_LEN * HIDDEN_SIZE);
    float* l_state = alloc_float(SEQ_LEN * HIDDEN_SIZE);
    float* h_state = alloc_float(SEQ_LEN * HIDDEN_SIZE);

    float* q_buf = alloc_float(SEQ_LEN * NUM_HEADS * HEAD_DIM);
    float* k_buf = alloc_float(SEQ_LEN * NUM_HEADS * HEAD_DIM);
    float* v_buf = alloc_float(SEQ_LEN * NUM_HEADS * HEAD_DIM);
    float* attn_buf = alloc_float(SEQ_LEN * SEQ_LEN);
    float* head_out = alloc_float(SEQ_LEN * HIDDEN_SIZE);
    float* temp = alloc_float(SEQ_LEN * HIDDEN_SIZE);

    float embed_scale = sqrtf((float)HIDDEN_SIZE);

    for (int i = 0; i < SEQ_LEN; i++) {
        int tok = puzzle[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            input_emb[i * HIDDEN_SIZE + j] = w.embed[tok * HIDDEN_SIZE + j] * embed_scale;
        }
    }

    for (int i = 0; i < SEQ_LEN; i++) {
        memcpy(z_h + i * HIDDEN_SIZE, w.H_init, HIDDEN_SIZE * sizeof(float));
        memcpy(z_l + i * HIDDEN_SIZE, w.L_init, HIDDEN_SIZE * sizeof(float));
    }

    for (int step = 0; step < MAX_STEPS; step++) {
        for (int i = 0; i < SEQ_LEN * HIDDEN_SIZE; i++) {
            l_state[i] = z_l[i] + z_h[i] + input_emb[i];
        }
        for (int layer = 0; layer < L_LAYERS; layer++) {
            transformer_block(l_state, 1, layer, w, q_buf, k_buf, v_buf, attn_buf, head_out, temp);
        }
        memcpy(z_l, l_state, SEQ_LEN * HIDDEN_SIZE * sizeof(float));

        for (int i = 0; i < SEQ_LEN * HIDDEN_SIZE; i++) {
            h_state[i] = z_h[i] + z_l[i];
        }
        for (int layer = 0; layer < H_LAYERS; layer++) {
            transformer_block(h_state, 0, layer, w, q_buf, k_buf, v_buf, attn_buf, head_out, temp);
        }
        memcpy(z_h, h_state, SEQ_LEN * HIDDEN_SIZE * sizeof(float));

        if ((step + 1) % 4 == 0 || step == 0) {
            printf("  Step %d/%d\n", step + 1, MAX_STEPS);
        }
    }

    matmul(logits, z_h, w.lm_head, SEQ_LEN, VOCAB_SIZE, HIDDEN_SIZE);

    free(input_emb); free(z_h); free(z_l); free(l_state); free(h_state);
    free(q_buf); free(k_buf); free(v_buf); free(attn_buf); free(head_out); free(temp);
}

void print_grid(const int* grid, const char* title) {
    printf("\n%s:\n", title);
    for (int i = 0; i < 9; i++) {
        if (i > 0 && i % 3 == 0) printf("---------------------\n");
        for (int j = 0; j < 9; j++) {
            if (j > 0 && j % 3 == 0) printf("| ");
            int val = grid[i * 9 + j];
            if (val > 0) printf("%d ", val);
            else printf(". ");
        }
        printf("\n");
    }
}

int main(int argc, char** argv) {
    printf("============================================================\n");
    printf("Samsung HRM Sudoku - PowerPC G5 C++ (Accelerate BLAS)\n");
    printf("============================================================\n");

    const char* weights_dir = "weights_be";
    if (argc > 1) weights_dir = argv[1];

    init_rope();
    Weights w = load_all_weights(weights_dir);

    int puzzle[81] = {
        5,3,0,0,7,0,0,0,0, 6,0,0,1,9,5,0,0,0, 0,9,8,0,0,0,0,6,0,
        8,0,0,0,6,0,0,0,3, 4,0,0,8,0,3,0,0,1, 7,0,0,0,2,0,0,0,6,
        0,6,0,0,0,0,2,8,0, 0,0,0,4,1,9,0,0,5, 0,0,0,0,8,0,0,7,9
    };

    int correct[81] = {
        5,3,4,6,7,8,9,1,2, 6,7,2,1,9,5,3,4,8, 1,9,8,3,4,2,5,6,7,
        8,5,9,7,6,1,4,2,3, 4,2,6,8,5,3,7,9,1, 7,1,3,9,2,4,8,5,6,
        9,6,1,5,3,7,2,8,4, 2,8,7,4,1,9,6,3,5, 3,4,5,2,8,6,1,7,9
    };

    print_grid(puzzle, "Input");
    printf("\nRunning inference (Accelerate BLAS)...\n");

    float* logits = alloc_float(SEQ_LEN * VOCAB_SIZE);

    clock_t start = clock();
    forward(logits, puzzle, w);
    clock_t end = clock();

    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;

    int solution[81];
    for (int i = 0; i < SEQ_LEN; i++) {
        int best = 0;
        float best_score = logits[i * VOCAB_SIZE];
        for (int j = 1; j < VOCAB_SIZE; j++) {
            if (logits[i * VOCAB_SIZE + j] > best_score) {
                best_score = logits[i * VOCAB_SIZE + j];
                best = j;
            }
        }
        solution[i] = best;
    }

    int matches = 0;
    for (int i = 0; i < 81; i++) {
        if (solution[i] == correct[i]) matches++;
    }

    printf("\nSolution (%.2fs):\n", elapsed);
    for (int i = 0; i < 9; i++) {
        if (i > 0 && i % 3 == 0) printf("---------------------\n");
        for (int j = 0; j < 9; j++) {
            if (j > 0 && j % 3 == 0) printf("| ");
            int pred = solution[i * 9 + j];
            int corr = correct[i * 9 + j];
            if (pred == corr) printf("%d ", pred);
            else printf("X ");
        }
        printf("\n");
    }

    printf("\nAccuracy: %d/81 = %.1f%%\n", matches, 100.0 * matches / 81);
    printf("Speedup vs Python (76s): %.1fx\n", 76.0 / elapsed);

    free(logits);
    return 0;
}
