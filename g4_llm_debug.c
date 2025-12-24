#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

int main() {
    printf("LLM on PowerPC G4\n");
    
    FILE *f = fopen("g4_model.bin", "rb");
    if(!f) { 
        printf("Cannot open g4_model.bin\n");
        return 1;
    }
    
    int vs, ne;
    fread(&vs, 4, 1, f);
    fread(&ne, 4, 1, f);
    
    printf("Vocab: %d, Embed: %d\n", vs, ne);
    printf("File opened successfully!\n");
    
    fclose(f);
    return 0;
}
