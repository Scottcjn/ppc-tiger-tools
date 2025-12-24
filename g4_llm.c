/* Real LLM on PowerPC G4 - Character-level generation
 * Trained model: 16,221 parameters
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

#define MAX_VOCAB 32
#define MAX_EMBD 128

char vocab[MAX_VOCAB];
int vocab_size, n_embd;

typedef struct {
    float *embed, *l1_w, *l1_b, *l2_w, *l2_b, *l3_w, *l3_b, *head_w, *head_b;
} Model;

void forward(Model *m, int token, float *out) {
    float h1[MAX_EMBD], h2[MAX_EMBD], h3[MAX_EMBD];
    int i, j;

    for(i=0; i<n_embd; i++) h1[i] = m->embed[token*n_embd + i];

    for(i=0; i<n_embd; i++) {
        float s = m->l1_b[i];
        for(j=0; j<n_embd; j++) s += m->l1_w[i*n_embd+j] * h1[j];
        h2[i] = tanhf(s);
    }

    for(i=0; i<n_embd; i++) {
        float s = m->l2_b[i];
        for(j=0; j<n_embd; j++) s += m->l2_w[i*n_embd+j] * h2[j];
        h3[i] = tanhf(s);
    }

    for(i=0; i<n_embd; i++) {
        float s = m->l3_b[i];
        for(j=0; j<n_embd; j++) s += m->l3_w[i*n_embd+j] * h3[j];
        h1[i] = tanhf(s);
    }

    for(i=0; i<vocab_size; i++) {
        float s = m->head_b[i];
        for(j=0; j<n_embd; j++) s += m->head_w[i*n_embd+j] * h1[j];
        out[i] = s;
    }
}

int sample(float *logits) {
    float max = logits[0];
    int i;
    for(i=1; i<vocab_size; i++) if(logits[i]>max) max=logits[i];

    float sum=0;
    for(i=0; i<vocab_size; i++) {
        logits[i] = expf(logits[i]-max);
        sum += logits[i];
    }
    for(i=0; i<vocab_size; i++) logits[i] /= sum;

    float r = (float)rand()/RAND_MAX, c=0;
    for(i=0; i<vocab_size; i++) {
        c += logits[i];
        if(r < c) return i;
    }
    return 0;
}

int main() {
    printf("🔥 LLM on PowerPC G4 - TRAINED MODEL 🔥\n");
    printf("========================================\n\n");

    FILE *f = fopen("g4_model.bin", "rb");
    if(!f) { printf("Error: g4_model.bin not found\n"); return 1; }

    fread(&vocab_size, 4, 1, f);
    fread(&n_embd, 4, 1, f);

    printf("Model: %d vocab, %d embed\n", vocab_size, n_embd);

    Model m;
    m.embed = malloc(vocab_size*n_embd*sizeof(float));
    m.l1_w = malloc(n_embd*n_embd*sizeof(float));
    m.l1_b = malloc(n_embd*sizeof(float));
    m.l2_w = malloc(n_embd*n_embd*sizeof(float));
    m.l2_b = malloc(n_embd*sizeof(float));
    m.l3_w = malloc(n_embd*n_embd*sizeof(float));
    m.l3_b = malloc(n_embd*sizeof(float));
    m.head_w = malloc(vocab_size*n_embd*sizeof(float));
    m.head_b = malloc(vocab_size*sizeof(float));

    fread(m.embed, sizeof(float), vocab_size*n_embd, f);
    fread(m.l1_w, sizeof(float), n_embd*n_embd, f);
    fread(m.l1_b, sizeof(float), n_embd, f);
    fread(m.l2_w, sizeof(float), n_embd*n_embd, f);
    fread(m.l2_b, sizeof(float), n_embd, f);
    fread(m.l3_w, sizeof(float), n_embd*n_embd, f);
    fread(m.l3_b, sizeof(float), n_embd, f);
    fread(m.head_w, sizeof(float), vocab_size*n_embd, f);
    fread(m.head_b, sizeof(float), vocab_size, f);
    fclose(f);

    FILE *vf = fopen("g4_vocab.txt", "r");
    if(vf) {
        fread(vocab, 1, vocab_size, vf);
        fclose(vf);
    }

    printf("Params: %d\n",
           vocab_size*n_embd + n_embd*n_embd*3 + n_embd*3 + vocab_size*n_embd + vocab_size);
    printf("\nGenerating text on PowerPC G4...\n\n");

    srand(42);
    float logits[MAX_VOCAB];
    int token = 0;  // Start token

    struct timeval start, end;
    gettimeofday(&start, NULL);

    int i;
    for(i=0; i<200; i++) {
        forward(&m, token, logits);
        token = sample(logits);
        printf("%c", vocab[token]);
        fflush(stdout);
    }

    gettimeofday(&end, NULL);
    unsigned long us = (end.tv_sec-start.tv_sec)*1000000UL + (end.tv_usec-start.tv_usec);

    printf("\n\n%.1f tokens/sec on PowerPC G4\n", 200.0/(us/1000000.0));
    printf("\n🔥 REAL TEXT GENERATION ON VINTAGE HARDWARE 🔥\n");

    return 0;
}
