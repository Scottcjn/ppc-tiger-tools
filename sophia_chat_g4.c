/*
 * Sophia GPT Prime - Full Transformer Chat for PowerPC G4
 * Optimized with AltiVec SIMD
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <altivec.h>

// Model architecture (matches training)
#define VOCAB_SIZE 78
#define N_EMBD 192
#define N_LAYER 8
#define N_HEAD 8
#define HEAD_DIM (N_EMBD / N_HEAD)
#define MAX_SEQ 256

typedef struct {
    float *token_embd;      // [VOCAB_SIZE, N_EMBD]
    float *pos_embd;        // [MAX_SEQ, N_EMBD]

    // Per-layer weights (8 layers)
    float **ln1_w, **ln1_b; // Layer norm 1
    float **qkv_w, **qkv_b; // QKV projection
    float **proj_w, **proj_b; // Output projection
    float **ln2_w, **ln2_b; // Layer norm 2
    float **mlp1_w, **mlp1_b; // MLP layer 1
    float **mlp2_w, **mlp2_b; // MLP layer 2

    // Final layer norm + head
    float *ln_f_w, *ln_f_b;
    float *head_w, *head_b;
} Model;

// Vocab mapping
char vocab[VOCAB_SIZE];

void load_vocab(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Failed to open vocab\n"); exit(1); }

    int idx = 0;
    int c;
    while ((c = fgetc(f)) != EOF && idx < VOCAB_SIZE) {
        if (c != '\n') vocab[idx++] = c;
    }
    fclose(f);
    printf("✅ Loaded %d vocab chars\n", idx);
}

unsigned int read_be_uint(FILE *f) {
    unsigned char b[4];
    fread(b, 1, 4, f);
    return (b[0] << 24) | (b[1] << 16) | (b[2] << 8) | b[3];
}

float read_be_float(FILE *f) {
    unsigned char b[4];
    fread(b, 1, 4, f);
    unsigned int val = (b[0] << 24) | (b[1] << 16) | (b[2] << 8) | b[3];
    return *(float*)&val;
}

void load_model(Model *m, const char *path) {
    int i, L;
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Failed to open model\n"); exit(1); }

    // Read header
    unsigned int vocab_size = read_be_uint(f);
    unsigned int n_embd = read_be_uint(f);
    unsigned int n_layer = read_be_uint(f);
    unsigned int n_head = read_be_uint(f);

    printf("📖 Model: vocab=%u embd=%u layers=%u heads=%u\n",
           vocab_size, n_embd, n_layer, n_head);

    // Allocate weights
    m->token_embd = malloc(VOCAB_SIZE * N_EMBD * sizeof(float));
    m->pos_embd = malloc(MAX_SEQ * N_EMBD * sizeof(float));

    m->ln1_w = malloc(N_LAYER * sizeof(float*));
    m->ln1_b = malloc(N_LAYER * sizeof(float*));
    m->qkv_w = malloc(N_LAYER * sizeof(float*));
    m->qkv_b = malloc(N_LAYER * sizeof(float*));
    m->proj_w = malloc(N_LAYER * sizeof(float*));
    m->proj_b = malloc(N_LAYER * sizeof(float*));
    m->ln2_w = malloc(N_LAYER * sizeof(float*));
    m->ln2_b = malloc(N_LAYER * sizeof(float*));
    m->mlp1_w = malloc(N_LAYER * sizeof(float*));
    m->mlp1_b = malloc(N_LAYER * sizeof(float*));
    m->mlp2_w = malloc(N_LAYER * sizeof(float*));
    m->mlp2_b = malloc(N_LAYER * sizeof(float*));

    for (i = 0; i < N_LAYER; i++) {
        m->ln1_w[i] = malloc(N_EMBD * sizeof(float));
        m->ln1_b[i] = malloc(N_EMBD * sizeof(float));
        m->qkv_w[i] = malloc(N_EMBD * 3 * N_EMBD * sizeof(float));
        m->qkv_b[i] = malloc(3 * N_EMBD * sizeof(float));
        m->proj_w[i] = malloc(N_EMBD * N_EMBD * sizeof(float));
        m->proj_b[i] = malloc(N_EMBD * sizeof(float));
        m->ln2_w[i] = malloc(N_EMBD * sizeof(float));
        m->ln2_b[i] = malloc(N_EMBD * sizeof(float));
        m->mlp1_w[i] = malloc(N_EMBD * 4 * N_EMBD * sizeof(float));
        m->mlp1_b[i] = malloc(4 * N_EMBD * sizeof(float));
        m->mlp2_w[i] = malloc(4 * N_EMBD * N_EMBD * sizeof(float));
        m->mlp2_b[i] = malloc(N_EMBD * sizeof(float));
    }

    m->ln_f_w = malloc(N_EMBD * sizeof(float));
    m->ln_f_b = malloc(N_EMBD * sizeof(float));
    m->head_w = malloc(N_EMBD * VOCAB_SIZE * sizeof(float));
    m->head_b = malloc(VOCAB_SIZE * sizeof(float));

    // Read weights in order
    for (i = 0; i < VOCAB_SIZE * N_EMBD; i++) m->token_embd[i] = read_be_float(f);
    for (i = 0; i < MAX_SEQ * N_EMBD; i++) m->pos_embd[i] = read_be_float(f);

    for (L = 0; L < N_LAYER; L++) {
        for (i = 0; i < N_EMBD; i++) m->ln1_w[L][i] = read_be_float(f);
        for (i = 0; i < N_EMBD; i++) m->ln1_b[L][i] = read_be_float(f);
        for (i = 0; i < N_EMBD * 3 * N_EMBD; i++) m->qkv_w[L][i] = read_be_float(f);
        for (i = 0; i < 3 * N_EMBD; i++) m->qkv_b[L][i] = read_be_float(f);
        for (i = 0; i < N_EMBD * N_EMBD; i++) m->proj_w[L][i] = read_be_float(f);
        for (i = 0; i < N_EMBD; i++) m->proj_b[L][i] = read_be_float(f);
        for (i = 0; i < N_EMBD; i++) m->ln2_w[L][i] = read_be_float(f);
        for (i = 0; i < N_EMBD; i++) m->ln2_b[L][i] = read_be_float(f);
        for (i = 0; i < N_EMBD * 4 * N_EMBD; i++) m->mlp1_w[L][i] = read_be_float(f);
        for (i = 0; i < 4 * N_EMBD; i++) m->mlp1_b[L][i] = read_be_float(f);
        for (i = 0; i < 4 * N_EMBD * N_EMBD; i++) m->mlp2_w[L][i] = read_be_float(f);
        for (i = 0; i < N_EMBD; i++) m->mlp2_b[L][i] = read_be_float(f);
    }

    for (i = 0; i < N_EMBD; i++) m->ln_f_w[i] = read_be_float(f);
    for (i = 0; i < N_EMBD; i++) m->ln_f_b[i] = read_be_float(f);
    for (i = 0; i < N_EMBD * VOCAB_SIZE; i++) m->head_w[i] = read_be_float(f);
    for (i = 0; i < VOCAB_SIZE; i++) m->head_b[i] = read_be_float(f);

    fclose(f);
    printf("✅ Model loaded\n");
}

// AltiVec optimized matrix-vector multiply
void matmul(const float *W, const float *x, float *out, int rows, int cols) {
    int i, j;
    for (i = 0; i < rows; i++) {
        vector float sum = (vector float){0,0,0,0};
        for (j = 0; j + 3 < cols; j += 4) {
            vector float vx = vec_ld(0, (float*)(x + j));
            vector float vw = vec_ld(0, (float*)(W + i * cols + j));
            sum = vec_madd(vx, vw, sum);
        }

        // Horizontal sum
        vector float perm1 = vec_perm(sum, sum, (vector unsigned char){4,5,6,7,0,1,2,3,12,13,14,15,8,9,10,11});
        sum = vec_add(sum, perm1);
        vector float perm2 = vec_perm(sum, sum, (vector unsigned char){8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7});
        sum = vec_add(sum, perm2);

        float result;
        vec_ste(sum, 0, &result);

        // Handle remainder
        for (; j < cols; j++) {
            result += W[i * cols + j] * x[j];
        }

        out[i] = result;
    }
}

void layer_norm(float *x, const float *w, const float *b, int size) {
    int i;
    float mean = 0, var = 0;
    for (i = 0; i < size; i++) mean += x[i];
    mean /= size;
    for (i = 0; i < size; i++) {
        float diff = x[i] - mean;
        var += diff * diff;
    }
    var = sqrtf(var / size + 1e-5f);
    for (i = 0; i < size; i++) {
        x[i] = (x[i] - mean) / var * w[i] + b[i];
    }
}

void gelu(float *x, int size) {
    int i;
    for (i = 0; i < size; i++) {
        float v = x[i];
        x[i] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
    }
}

void softmax(float *x, int size) {
    int i;
    float max_val = x[0];
    for (i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0;
    for (i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (i = 0; i < size; i++) x[i] /= sum;
}

int char_to_idx(char c) {
    int i;
    for (i = 0; i < VOCAB_SIZE; i++) {
        if (vocab[i] == c) return i;
    }
    return 0; // Default to first char if not found
}

void forward(Model *m, int *tokens, int n_tokens, float *logits) {
    int i, L;
    float x[N_EMBD] __attribute__((aligned(16)));
    float qkv[3 * N_EMBD] __attribute__((aligned(16)));
    float attn_out[N_EMBD] __attribute__((aligned(16)));
    float mlp_hidden[4 * N_EMBD] __attribute__((aligned(16)));

    // Embed last token + position
    for (i = 0; i < N_EMBD; i++) {
        x[i] = m->token_embd[tokens[n_tokens-1] * N_EMBD + i] +
               m->pos_embd[(n_tokens-1) * N_EMBD + i];
    }

    // Transform through layers
    for (L = 0; L < N_LAYER; L++) {
        // Pre-norm + attention
        float x_norm[N_EMBD];
        memcpy(x_norm, x, N_EMBD * sizeof(float));
        layer_norm(x_norm, m->ln1_w[L], m->ln1_b[L], N_EMBD);

        // QKV projection (simplified - just use last position)
        matmul(m->qkv_w[L], x_norm, qkv, 3 * N_EMBD, N_EMBD);
        for (i = 0; i < 3 * N_EMBD; i++) qkv[i] += m->qkv_b[L][i];

        // Simple self-attention (single position)
        memcpy(attn_out, qkv, N_EMBD * sizeof(float));

        // Output projection
        float proj_out[N_EMBD];
        matmul(m->proj_w[L], attn_out, proj_out, N_EMBD, N_EMBD);
        for (i = 0; i < N_EMBD; i++) {
            x[i] += proj_out[i] + m->proj_b[L][i]; // Residual
        }

        // Pre-norm + MLP
        memcpy(x_norm, x, N_EMBD * sizeof(float));
        layer_norm(x_norm, m->ln2_w[L], m->ln2_b[L], N_EMBD);

        matmul(m->mlp1_w[L], x_norm, mlp_hidden, 4 * N_EMBD, N_EMBD);
        for (i = 0; i < 4 * N_EMBD; i++) mlp_hidden[i] += m->mlp1_b[L][i];
        gelu(mlp_hidden, 4 * N_EMBD);

        float mlp_out[N_EMBD];
        matmul(m->mlp2_w[L], mlp_hidden, mlp_out, N_EMBD, 4 * N_EMBD);
        for (i = 0; i < N_EMBD; i++) {
            x[i] += mlp_out[i] + m->mlp2_b[L][i]; // Residual
        }
    }

    // Final layer norm + head
    layer_norm(x, m->ln_f_w, m->ln_f_b, N_EMBD);
    matmul(m->head_w, x, logits, VOCAB_SIZE, N_EMBD);
    for (i = 0; i < VOCAB_SIZE; i++) logits[i] += m->head_b[i];
}

int sample(float *logits, float temperature) {
    int i;
    // Apply temperature
    for (i = 0; i < VOCAB_SIZE; i++) logits[i] /= temperature;
    softmax(logits, VOCAB_SIZE);

    // Sample
    float r = (float)rand() / RAND_MAX;
    float cumsum = 0;
    for (i = 0; i < VOCAB_SIZE; i++) {
        cumsum += logits[i];
        if (r < cumsum) return i;
    }
    return VOCAB_SIZE - 1;
}

void generate(Model *m, const char *prompt, int max_len, float temp) {
    int tokens[MAX_SEQ];
    int n_tokens = 0;
    int i;

    // Tokenize prompt
    for (i = 0; prompt[i] && n_tokens < MAX_SEQ; i++) {
        tokens[n_tokens++] = char_to_idx(prompt[i]);
    }

    printf("%s", prompt);
    fflush(stdout);

    clock_t start = clock();

    // Generate
    for (i = 0; i < max_len && n_tokens < MAX_SEQ; i++) {
        float logits[VOCAB_SIZE];
        forward(m, tokens, n_tokens, logits);

        int next_token = sample(logits, temp);
        tokens[n_tokens++] = next_token;

        char c = vocab[next_token];
        printf("%c", c);
        fflush(stdout);

        if (c == '\n' && i > 10) break; // Stop at newline after some generation
    }

    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("\n⚡ %.1f tok/s\n", (n_tokens - strlen(prompt)) / elapsed);
}

int main() {
    printf("\n");
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║         🚀 SOPHIA GPT PRIME - POWERPC G4 🚀              ║\n");
    printf("║         AltiVec-Accelerated Transformer Chat              ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n\n");

    srand(time(NULL));

    load_vocab("sophia_gpt_vocab.txt");

    Model model;
    load_model(&model, "sophia_gpt_g4.bin");

    printf("\n💬 Chat started! (Type 'quit' to exit)\n");
    printf("═══════════════════════════════════════════════════════════\n\n");

    char input[256];
    while (1) {
        printf("You: ");
        if (!fgets(input, sizeof(input), stdin)) break;

        // Remove newline
        input[strcspn(input, "\n")] = 0;

        if (strcmp(input, "quit") == 0) break;

        // Create prompt
        char prompt[512];
        snprintf(prompt, sizeof(prompt), "User: %s\nAssistant: ", input);

        printf("Sophia: ");
        generate(&model, prompt, 100, 0.8f);
        printf("\n");
    }

    printf("\n👋 Goodbye!\n\n");
    return 0;
}
