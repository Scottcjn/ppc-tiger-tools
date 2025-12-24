/* Simplified Tiny Sophia for G4 - hardcoded vocab */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define VOCAB_SIZE 19
#define N_EMBD 32

char vocab[] = " abdehilmnoprstwy";  // 19 chars from training

int main() {
    printf("Tiny Sophia on G4\n");
    
    FILE *f = fopen("tiny_sophia.bin", "rb");
    if(!f) { printf("Error\n"); return 1; }
    
    // Skip header (8 bytes)
    fseek(f, 8, SEEK_SET);
    
    // Read embed table
    float *embed = malloc(VOCAB_SIZE * N_EMBD * sizeof(float));
    fread(embed, sizeof(float), VOCAB_SIZE * N_EMBD, f);
    
    printf("Embed[0][0] = %f\n", embed[0]);
    printf("Model loaded!\n");
    
    fclose(f);
    free(embed);
    return 0;
}
