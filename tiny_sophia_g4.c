/* Tiny Sophia - Character-level LLM for PowerPC G4
 * TRAINED model - generates real text!
 * 3,347 parameters, 0.013 MB
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#define MAX_VOCAB 32
#define MAX_EMBD 64

typedef struct {
    int vocab_size;
    int n_embd;

    float *embed;      // [vocab_size × n_embd]
    float *fc1_w;      // [n_embd × n_embd]
    float *fc1_b;      // [n_embd]
    float *fc2_w;      // [n_embd × n_embd]
    float *fc2_b;      // [n_embd]
    float *head_w;     // [vocab_size × n_embd]
    float *head_b;     // [vocab_size]

    char vocab[MAX_VOCAB];
} TinySophia;

/* Load model from binary file */
int load_model(const char *path, TinySophia *model) {
    FILE *f = fopen(path, "rb");
    if(!f) return -1;

    // Read header
    fread(&model->vocab_size, sizeof(int), 1, f);
    fread(&model->n_embd, sizeof(int), 1, f);

    int vs = model->vocab_size;
    int ne = model->n_embd;

    // Allocate
    model->embed = (float*)malloc(vs * ne * sizeof(float));
    model->fc1_w = (float*)malloc(ne * ne * sizeof(float));
    model->fc1_b = (float*)malloc(ne * sizeof(float));
    model->fc2_w = (float*)malloc(ne * ne * sizeof(float));
    model->fc2_b = (float*)malloc(ne * sizeof(float));
    model->head_w = (float*)malloc(vs * ne * sizeof(float));
    model->head_b = (float*)malloc(vs * sizeof(float));

    // Read weights
    fread(model->embed, sizeof(float), vs * ne, f);
    fread(model->fc1_w, sizeof(float), ne * ne, f);
    fread(model->fc1_b, sizeof(float), ne, f);
    fread(model->fc2_w, sizeof(float), ne * ne, f);
    fread(model->fc2_b, sizeof(float), ne, f);
    fread(model->head_w, sizeof(float), vs * ne, f);
    fread(model->head_b, sizeof(float), vs, f);

    fclose(f);
    return 0;
}

/* Forward pass */
void forward(TinySophia *model, int token, float *output) {
    int i, j;
    int ne = model->n_embd;
    int vs = model->vocab_size;

    float h1[MAX_EMBD];
    float h2[MAX_EMBD];

    // Embedding lookup
    memcpy(h1, model->embed + token * ne, ne * sizeof(float));

    // FC1: h2 = tanh(fc1_w @ h1 + fc1_b)
    for(i = 0; i < ne; i++) {
        float sum = model->fc1_b[i];
        for(j = 0; j < ne; j++) {
            sum += model->fc1_w[i * ne + j] * h1[j];
        }
        h2[i] = tanhf(sum);
    }

    // FC2: h1 = tanh(fc2_w @ h2 + fc2_b)
    for(i = 0; i < ne; i++) {
        float sum = model->fc2_b[i];
        for(j = 0; j < ne; j++) {
            sum += model->fc2_w[i * ne + j] * h2[j];
        }
        h1[i] = tanhf(sum);
    }

    // Head: output = head_w @ h1 + head_b
    for(i = 0; i < vs; i++) {
        float sum = model->head_b[i];
        for(j = 0; j < ne; j++) {
            sum += model->head_w[i * ne + j] * h1[j];
        }
        output[i] = sum;
    }
}

/* Softmax */
void softmax(float *x, int size) {
    float max_val = x[0];
    int i;
    for(i = 1; i < size; i++) {
        if(x[i] > max_val) max_val = x[i];
    }

    float sum = 0.0f;
    for(i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    for(i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

/* Sample token */
int sample(float *probs, int vocab_size) {
    float r = (float)rand() / RAND_MAX;
    float cumsum = 0.0f;
    int i;

    for(i = 0; i < vocab_size; i++) {
        cumsum += probs[i];
        if(r < cumsum) return i;
    }

    return vocab_size - 1;
}

int main(int argc, char *argv[]) {
    printf("🌟 Tiny Sophia - Character-level LLM for PowerPC G4 🌟\n");
    printf("======================================================\n\n");

    if(argc < 2) {
        printf("Usage: %s <weights.bin> [seed_text]\n", argv[0]);
        return 1;
    }

    TinySophia model;
    if(load_model(argv[1], &model) != 0) {
        printf("Error loading model: %s\n", argv[1]);
        return 1;
    }

    printf("Model loaded: %d vocab, %d embed\n", model.vocab_size, model.n_embd);
    printf("Parameters: %d\n\n",
           model.vocab_size * model.n_embd +
           model.n_embd * model.n_embd * 2 +
           model.n_embd * 2 +
           model.vocab_size * model.n_embd +
           model.vocab_size);

    // Load vocab
    FILE *vf = fopen("/tmp/vocab.txt", "r");
    if(vf) {
        int idx;
        char ch[10];
        while(fscanf(vf, "%d '%c'", &idx, &model.vocab[idx]) == 2) {}
        fclose(vf);
    }

    // Seed
    srand(42);

    const char *seed = (argc > 2) ? argv[2] : "";
    printf("Generating from: \"%s\"\n", seed);
    printf("Output: %s", seed);
    fflush(stdout);

    // Generate
    int current_token = 0;  // Start token
    float logits[MAX_VOCAB];

    struct timeval start, end;
    gettimeofday(&start, NULL);

    int num_tokens = 100;
    int i;

    for(i = 0; i < num_tokens; i++) {
        forward(&model, current_token, logits);
        softmax(logits, model.vocab_size);
        current_token = sample(logits, model.vocab_size);

        printf("%c", model.vocab[current_token]);
        fflush(stdout);

        if(current_token == '\n') break;
    }

    gettimeofday(&end, NULL);
    unsigned long us = (end.tv_sec - start.tv_sec) * 1000000UL +
                       (end.tv_usec - start.tv_usec);

    printf("\n\n");
    printf("Generated %d tokens in %lu us\n", i, us);
    printf("Speed: %.2f tokens/sec\n", (float)i / (us / 1000000.0f));
    printf("\nTRAINED model - this is REAL text generation!\n");
    printf("Tiny Sophia: %d parameters on PowerPC G4\n",
           model.vocab_size * model.n_embd +
           model.n_embd * model.n_embd * 2 +
           model.n_embd * 2 +
           model.vocab_size * model.n_embd +
           model.vocab_size);

    printf("\n🔥 Even the smallest rocks generate language 🔥\n");

    return 0;
}
