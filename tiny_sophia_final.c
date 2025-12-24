/* Tiny Sophia - WORKING trained model on G4! */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

// Fixed from model export
#define VOCAB_SIZE 19
#define N_EMBD 32

char vocab[] = " abdehilmnoprstwy";

typedef struct {
    float *embed, *fc1_w, *fc1_b, *fc2_w, *fc2_b, *head_w, *head_b;
} Model;

void forward(Model *m, int token, float *out) {
    float h1[N_EMBD], h2[N_EMBD];
    int i, j;
    
    // Embedding
    for(i=0; i<N_EMBD; i++) h1[i] = m->embed[token * N_EMBD + i];
    
    // FC1
    for(i=0; i<N_EMBD; i++) {
        float s = m->fc1_b[i];
        for(j=0; j<N_EMBD; j++) s += m->fc1_w[i*N_EMBD+j] * h1[j];
        h2[i] = tanhf(s);
    }
    
    // FC2
    for(i=0; i<N_EMBD; i++) {
        float s = m->fc2_b[i];
        for(j=0; j<N_EMBD; j++) s += m->fc2_w[i*N_EMBD+j] * h2[j];
        h1[i] = tanhf(s);
    }
    
    // Head
    for(i=0; i<VOCAB_SIZE; i++) {
        float s = m->head_b[i];
        for(j=0; j<N_EMBD; j++) s += m->head_w[i*N_EMBD+j] * h1[j];
        out[i] = s;
    }
}

int sample(float *logits) {
    float max = logits[0];
    int i;
    for(i=1; i<VOCAB_SIZE; i++) if(logits[i]>max) max=logits[i];
    
    float sum=0;
    for(i=0; i<VOCAB_SIZE; i++) {
        logits[i] = expf(logits[i]-max);
        sum += logits[i];
    }
    for(i=0; i<VOCAB_SIZE; i++) logits[i] /= sum;
    
    float r = (float)rand()/RAND_MAX, c=0;
    for(i=0; i<VOCAB_SIZE; i++) {
        c += logits[i];
        if(r < c) return i;
    }
    return VOCAB_SIZE-1;
}

int main() {
    printf("🌟 Tiny Sophia on PowerPC G4 🌟\n\n");
    
    FILE *f = fopen("tiny_sophia.bin", "rb");
    if(!f) { printf("Error loading model\n"); return 1; }
    
    fseek(f, 8, SEEK_SET);  // Skip header
    
    Model m;
    m.embed = malloc(VOCAB_SIZE*N_EMBD*sizeof(float));
    m.fc1_w = malloc(N_EMBD*N_EMBD*sizeof(float));
    m.fc1_b = malloc(N_EMBD*sizeof(float));
    m.fc2_w = malloc(N_EMBD*N_EMBD*sizeof(float));
    m.fc2_b = malloc(N_EMBD*sizeof(float));
    m.head_w = malloc(VOCAB_SIZE*N_EMBD*sizeof(float));
    m.head_b = malloc(VOCAB_SIZE*sizeof(float));
    
    fread(m.embed, sizeof(float), VOCAB_SIZE*N_EMBD, f);
    fread(m.fc1_w, sizeof(float), N_EMBD*N_EMBD, f);
    fread(m.fc1_b, sizeof(float), N_EMBD, f);
    fread(m.fc2_w, sizeof(float), N_EMBD*N_EMBD, f);
    fread(m.fc2_b, sizeof(float), N_EMBD, f);
    fread(m.head_w, sizeof(float), VOCAB_SIZE*N_EMBD, f);
    fread(m.head_b, sizeof(float), VOCAB_SIZE, f);
    fclose(f);
    
    printf("Model: 3,347 parameters\n");
    printf("Generating: ");
    fflush(stdout);
    
    srand(42);
    float logits[VOCAB_SIZE];
    int token = 0;
    
    struct timeval start, end;
    gettimeofday(&start, NULL);
    
    int i;
    for(i=0; i<80; i++) {
        forward(&m, token, logits);
        token = sample(logits);
        printf("%c", vocab[token]);
        fflush(stdout);
    }
    
    gettimeofday(&end, NULL);
    unsigned long us = (end.tv_sec-start.tv_sec)*1000000UL + (end.tv_usec-start.tv_usec);
    
    printf("\n\n%.1f tokens/sec\n", 80.0/(us/1000000.0));
    printf("\n🔥 TRAINED model on PowerPC G4! 🔥\n");
    
    return 0;
}
