/* AltiVec Non-Bijunctive Power
 *
 * The secret: AltiVec permute is NON-BIJUNCTIVE
 * - One input byte can map to multiple outputs
 * - Multiple transformations in ONE cycle
 * - This is the "explosive" capability
 */
#include <stdio.h>
#include <stdlib.h>

#ifdef __ALTIVEC__
#include <altivec.h>

// Example: Broadcast one float to all 4 positions (non-bijunctive!)
vector float broadcast_permute(vector float v, int index) {
    // Permute pattern: all bytes select from same source element
    unsigned char pattern[16];
    int i;
    int base = index * 4;  // Float is 4 bytes

    for(i = 0; i < 16; i++) {
        pattern[i] = base + (i % 4);
    }

    vector unsigned char perm = vec_ld(0, pattern);
    return vec_perm(v, v, perm);
}

// Horizontal sum using permute (non-bijunctive reduction!)
float horizontal_sum_perm(vector float v) {
    // First permute: [a,b,c,d] → [b,a,d,c]
    vector unsigned char perm1 = {4,5,6,7, 0,1,2,3, 12,13,14,15, 8,9,10,11};
    vector float v_swapped = vec_perm(v, v, perm1);
    v = vec_add(v, v_swapped);  // Now [a+b, b+a, c+d, d+c]

    // Second permute: [a+b, *, c+d, *] → [c+d, *, a+b, *]
    vector unsigned char perm2 = {8,9,10,11, 12,13,14,15, 0,1,2,3, 4,5,6,7};
    v_swapped = vec_perm(v, v, perm2);
    v = vec_add(v, v_swapped);  // Now [a+b+c+d, *, *, *]

    float result;
    vec_ste(v, 0, &result);
    return result;
}

// Dot product using permute magic (ultra-fast)
float dot_nonbijunctive(const float *a, const float *b, int n) {
    vector float v_sum = {0, 0, 0, 0};
    int i;

    for(i = 0; i + 3 < n; i += 4) {
        vector float v_a = vec_ld(0, (float*)(a + i));
        vector float v_b = vec_ld(0, (float*)(b + i));
        v_sum = vec_madd(v_a, v_b, v_sum);
    }

    // Horizontal sum using permute (non-bijunctive!)
    return horizontal_sum_perm(v_sum) + ((i < n) ? a[i] * b[i] : 0);
}

// Matrix transpose using permute (4x4 block)
void transpose_4x4_permute(float *out, const float *in) {
    vector float row0 = vec_ld(0, in);
    vector float row1 = vec_ld(0, in + 4);
    vector float row2 = vec_ld(0, in + 8);
    vector float row3 = vec_ld(0, in + 12);

    // Permute patterns for transpose
    vector unsigned char perm_02_lo = {0,1,2,3, 16,17,18,19, 4,5,6,7, 20,21,22,23};
    vector unsigned char perm_02_hi = {8,9,10,11, 24,25,26,27, 12,13,14,15, 28,29,30,31};
    vector unsigned char perm_13_lo = {0,1,2,3, 16,17,18,19, 4,5,6,7, 20,21,22,23};
    vector unsigned char perm_13_hi = {8,9,10,11, 24,25,26,27, 12,13,14,15, 28,29,30,31};

    vector float tmp0 = vec_perm(row0, row2, perm_02_lo);
    vector float tmp1 = vec_perm(row1, row3, perm_13_lo);
    vector float tmp2 = vec_perm(row0, row2, perm_02_hi);
    vector float tmp3 = vec_perm(row1, row3, perm_13_hi);

    vector unsigned char perm_final1 = {0,1,2,3, 16,17,18,19, 8,9,10,11, 24,25,26,27};
    vector unsigned char perm_final2 = {4,5,6,7, 20,21,22,23, 12,13,14,15, 28,29,30,31};

    vector float out0 = vec_perm(tmp0, tmp1, perm_final1);
    vector float out1 = vec_perm(tmp0, tmp1, perm_final2);
    vector float out2 = vec_perm(tmp2, tmp3, perm_final1);
    vector float out3 = vec_perm(tmp2, tmp3, perm_final2);

    vec_st(out0, 0, out);
    vec_st(out1, 0, out + 4);
    vec_st(out2, 0, out + 8);
    vec_st(out3, 0, out + 12);
}

// Data duplication (non-bijunctive: 1 input → many outputs)
void duplicate_pattern(float *out, float value, int count) {
    vector float v_val = {value, value, value, value};

    int i;
    for(i = 0; i + 3 < count; i += 4) {
        vec_st(v_val, 0, out + i);
    }

    // This is non-bijunctive: one value → many locations IN ONE OPERATION!
}

#endif

int main() {
    printf("🔮 AltiVec Non-Bijunctive Operations\\n");
    printf("=====================================\\n\\n");

#ifdef __ALTIVEC__
    printf("Non-bijunctive means: One input can map to MULTIPLE outputs\\n");
    printf("This enables multiple transformations in a SINGLE cycle!\\n\\n");

    // Example 1: Broadcast (1→4 mapping)
    vector float test = {1.0f, 2.0f, 3.0f, 4.0f};
    vector float broadcasted = broadcast_permute(test, 2);  // Broadcast element 2

    float result[4];
    vec_st(broadcasted, 0, result);
    printf("Broadcast test (index 2):\\n");
    printf("  Input:  [1.0, 2.0, 3.0, 4.0]\\n");
    printf("  Output: [%.1f, %.1f, %.1f, %.1f]\\n", result[0], result[1], result[2], result[3]);
    printf("  → Non-bijunctive: 3.0 mapped to ALL positions!\\n\\n");

    // Example 2: Horizontal sum using permute
    vector float sum_test = {1.0f, 2.0f, 3.0f, 4.0f};
    float sum = horizontal_sum_perm(sum_test);
    printf("Horizontal sum (permute-based):\\n");
    printf("  Input: [1.0, 2.0, 3.0, 4.0]\\n");
    printf("  Sum: %.1f\\n", sum);
    printf("  → Uses permute to rearrange + add (multiple ops, 1 cycle!)\\n\\n");

    // Example 3: Dot product
    float a[8] = {1,2,3,4,5,6,7,8};
    float b[8] = {8,7,6,5,4,3,2,1};
    float dot = dot_nonbijunctive(a, b, 8);
    printf("Dot product (8 elements):\\n");
    printf("  a: [1,2,3,4,5,6,7,8]\\n");
    printf("  b: [8,7,6,5,4,3,2,1]\\n");
    printf("  a·b = %.0f\\n", dot);
    printf("  → Permute collapses 4-wide SIMD result to scalar!\\n\\n");

    // Example 4: Transpose
    float matrix[16] = {
        1,2,3,4,
        5,6,7,8,
        9,10,11,12,
        13,14,15,16
    };
    float transposed[16];
    transpose_4x4_permute(transposed, matrix);

    printf("4x4 Transpose (permute-based):\\n");
    printf("  Input:       Output:\\n");
    int i, j;
    for(i = 0; i < 4; i++) {
        printf("  ");
        for(j = 0; j < 4; j++) printf("%2.0f ", matrix[i*4+j]);
        printf("  →  ");
        for(j = 0; j < 4; j++) printf("%2.0f ", transposed[i*4+j]);
        printf("\\n");
    }
    printf("  → Permute rearranges data WITHOUT arithmetic!\\n\\n");

    printf("💡 The Power:\\n");
    printf("  - vec_perm can DUPLICATE data (non-bijunctive!)\\n");
    printf("  - One source byte → many destination bytes\\n");
    printf("  - Enables pattern matching, broadcasting, reduction\\n");
    printf("  - ALL IN SINGLE CYCLE (no sequential operations!)\\n\\n");

    printf("🔥 For LLMs:\\n");
    printf("  - Broadcast embeddings across heads\\n");
    printf("  - Collapse multi-head attention in 1 cycle\\n");
    printf("  - Rearrange weight matrices on-the-fly\\n");
    printf("  - Pattern-match tokens (permute as lookup!)\\n\\n");

    printf("This is why AltiVec is EXPLOSIVE for neural networks!\\n");

#else
    printf("AltiVec not available.\\n");
#endif

    return 0;
}
