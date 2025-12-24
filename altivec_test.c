#include <altivec.h>
#include <stdio.h>

int main() {
    // Test vec_perm with thalamic gating pattern (pairs collapse)
    vector unsigned char v1 = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    vector unsigned char thalamic = {0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7};
    vector unsigned char result = vec_perm(v1, v1, thalamic);
    
    unsigned char *r = (unsigned char*)&result;
    printf("AltiVec thalamic collapse test:\n");
    printf("Input:  ");
    unsigned char *in = (unsigned char*)&v1;
    for(int i=0; i<16; i++) printf("%d ", in[i]);
    printf("\nOutput: ");
    for(int i=0; i<16; i++) printf("%d ", r[i]);
    printf("\n\nPattern: pairs collapse (many-to-one, 1-cycle non-bijective)\n");
    printf("Hardware: PowerPC G4 7400 @ 1.5GHz\n");
    printf("Even the rocks praise Him through structured forgetting.\n");
    return 0;
}
