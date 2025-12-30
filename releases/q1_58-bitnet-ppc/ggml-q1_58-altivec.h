/*
 * Q1.58 (BitNet Ternary) Quantization for PowerPC AltiVec
 *
 * Implements ternary weight quantization {-1, 0, +1} optimized for
 * PowerPC G4/G5 AltiVec SIMD instructions.
 *
 * Key insight: With ternary weights, matrix multiplication becomes
 * pure integer addition/subtraction - no FP multiply needed!
 *
 * December 2025
 */

#ifndef GGML_Q1_58_ALTIVEC_H
#define GGML_Q1_58_ALTIVEC_H

#include <stdint.h>
#include <stddef.h>

#ifdef __ALTIVEC__
#include <altivec.h>
#endif

/* Q1.58 block structure: 256 ternary weights per block */
#define QK_Q1_58 256
#define Q1_58_BLOCK_SIZE 68  /* 64 packed + 2 scale + 2 zero_count */

typedef struct {
    uint8_t packed[64];     /* 256 ternary weights (2 bits each) */
    uint16_t d;             /* FP16 scale factor */
    uint16_t zero_count;    /* Sparsity hint */
} block_q1_58;

/* Ternary encoding: -1 -> 00, 0 -> 01, +1 -> 10 */
#define TERNARY_NEG  0
#define TERNARY_ZERO 1
#define TERNARY_POS  2

/*
 * FP16 to FP32 conversion (uses lookup table on G4)
 * On systems with F16C, this would use hardware instruction
 */
#ifndef GGML_FP16_TO_FP32
extern float ggml_table_f32_f16[65536];
#define GGML_FP16_TO_FP32(x) ggml_table_f32_f16[(x)]
#endif

#ifdef __ALTIVEC__

/*
 * Unpack 16 ternary weights from 4 bytes
 *
 * Input: 4 bytes containing 16 x 2-bit ternary values
 * Output: 16 x int8 values in {-1, 0, +1}
 *
 * This is where AltiVec's vec_perm really shines!
 */
static inline vector signed char unpack_ternary_16_altivec(const uint8_t* packed) {
    /* Lookup table for 2-bit to signed char conversion */
    /* Index 0=00=-1, 1=01=0, 2=10=+1, 3=11=0 (reserved) */
    static const signed char lut_data[16] __attribute__((aligned(16))) = {
        -1, 0, 1, 0,   /* for indices 0,1,2,3 */
        -1, 0, 1, 0,   /* repeated for vec_perm */
        -1, 0, 1, 0,
        -1, 0, 1, 0
    };
    vector signed char lut = vec_ld(0, lut_data);

    /* Extract 2-bit indices from 4 packed bytes */
    uint8_t b0 = packed[0];
    uint8_t b1 = packed[1];
    uint8_t b2 = packed[2];
    uint8_t b3 = packed[3];

    /* Build index vector - each index is 0-3 */
    unsigned char idx_data[16] __attribute__((aligned(16))) = {
        (b0 >> 0) & 3, (b0 >> 2) & 3, (b0 >> 4) & 3, (b0 >> 6) & 3,
        (b1 >> 0) & 3, (b1 >> 2) & 3, (b1 >> 4) & 3, (b1 >> 6) & 3,
        (b2 >> 0) & 3, (b2 >> 2) & 3, (b2 >> 4) & 3, (b2 >> 6) & 3,
        (b3 >> 0) & 3, (b3 >> 2) & 3, (b3 >> 4) & 3, (b3 >> 6) & 3
    };
    vector unsigned char indices = vec_ld(0, idx_data);

    /* Use vec_perm to lookup all 16 values at once! */
    return vec_perm(lut, lut, indices);
}

/*
 * Alternative: Faster unpacking using shift operations
 * Avoids building index array in memory
 */
static inline vector signed char unpack_ternary_16_fast(uint32_t packed_word) {
    vector signed char result;
    signed char* r = (signed char*)&result;

    /* Decode each 2-bit value */
    static const signed char decode[4] = {-1, 0, 1, 0};

    for (int i = 0; i < 16; i++) {
        int idx = (packed_word >> (i * 2)) & 3;
        r[i] = decode[idx];
    }

    return result;
}

/*
 * Ternary dot product using AltiVec
 *
 * Key insight: Since weights are {-1, 0, +1}, multiplication becomes:
 *   w=-1: subtract activation
 *   w=0:  skip (contribute 0)
 *   w=+1: add activation
 *
 * This is MUCH faster than FP multiply on G4!
 */
static inline vector signed int vec_dot_ternary_altivec(
    vector signed char activations,   /* 16 x int8 activations */
    vector signed char weights)       /* 16 x ternary weights */
{
    /*
     * Method 1: Use vec_mule/vec_mulo
     * These do element-wise multiply of even/odd elements
     * Since weights are -1,0,+1, this is efficient
     */
    vector signed short prod_even = vec_mule(weights, activations);
    vector signed short prod_odd  = vec_mulo(weights, activations);

    /* Sum pairs of int16 to int32 */
    vector signed int sum = vec_splat_s32(0);

    /* Manual accumulation (vec_sum4s only works on int8->int32) */
    /* Pack int16 pairs and sum */
    sum = vec_msum(weights, activations, sum);

    return sum;
}

/*
 * Branchless ternary dot product
 * Uses conditional select instead of multiply
 */
static inline vector signed int vec_dot_ternary_branchless(
    vector signed char activations,
    vector signed char weights)
{
    vector signed char zero = vec_splat_s8(0);
    vector signed char one = vec_splat_s8(1);
    vector signed char neg_one = vec_splat_s8(-1);

    /* Create masks for each weight value */
    vector bool char is_pos = vec_cmpeq(weights, one);
    vector bool char is_neg = vec_cmpeq(weights, neg_one);

    /* Select: if pos, use activation; else 0 */
    vector signed char pos_contrib = vec_sel(zero, activations, is_pos);

    /* Select: if neg, use -activation; else 0 */
    vector signed char neg_act = vec_sub(zero, activations);
    vector signed char neg_contrib = vec_sel(zero, neg_act, is_neg);

    /* Sum contributions */
    vector signed char total = vec_add(pos_contrib, neg_contrib);

    /* Horizontal sum to int32 using vec_sum4s */
    vector signed int sum = vec_sum4s(total, vec_splat_s32(0));

    return sum;
}

/*
 * Full Q1.58 dequantization row
 *
 * Converts Q1.58 block data to float32
 * Used for debugging and compatibility, not performance-critical
 */
static inline void dequantize_row_q1_58_altivec(
    const block_q1_58* restrict x,
    float* restrict y,
    int64_t k)
{
    const int nb = k / QK_Q1_58;

    for (int i = 0; i < nb; i++) {
        /* Get scale factor */
        const float d = GGML_FP16_TO_FP32(x[i].d);

        /* Process 256 weights, 16 at a time */
        for (int j = 0; j < QK_Q1_58; j += 16) {
            /* Unpack 16 ternary weights */
            vector signed char weights = unpack_ternary_16_altivec(&x[i].packed[j / 4]);

            /* Convert to float and scale */
            signed char* w = (signed char*)&weights;
            for (int k = 0; k < 16; k++) {
                y[i * QK_Q1_58 + j + k] = d * (float)w[k];
            }
        }
    }
}

/*
 * Q1.58 x Q8 dot product - the main workhorse!
 *
 * Computes dot product of Q1.58 weights with Q8 activations
 * Returns single float scalar
 *
 * This is integer-only until the final scaling step!
 */
static inline void ggml_vec_dot_q1_58_q8_altivec(
    int n,
    float* restrict s,
    const void* restrict vx,      /* Q1.58 weights */
    size_t bx,                     /* Block stride x */
    const void* restrict vy,      /* Q8 activations */
    size_t by,                     /* Block stride y */
    int nrc)
{
    const block_q1_58* x = (const block_q1_58*)vx;

    /* Assume Q8_0 format for activations */
    /* block_q8_0: 32 bytes = 2 (d) + 32 (qs) for 32 values */
    const int8_t* y_qs = (const int8_t*)vy;

    const int nb = n / QK_Q1_58;
    int32_t total_sum = 0;  /* Integer accumulator! */

    for (int i = 0; i < nb; i++) {
        /* Get Q1.58 scale */
        const float d_x = GGML_FP16_TO_FP32(x[i].d);

        int32_t block_sum = 0;

        /* Process 256 elements, 16 at a time */
        for (int j = 0; j < QK_Q1_58; j += 16) {
            /* Load 16 int8 activations */
            /* Note: need to handle Q8 block boundaries */
            vector signed char va;
            int8_t* va_ptr = (int8_t*)&va;
            for (int k = 0; k < 16; k++) {
                va_ptr[k] = y_qs[i * QK_Q1_58 + j + k];
            }

            /* Unpack 16 ternary weights */
            vector signed char vw = unpack_ternary_16_altivec(&x[i].packed[j / 4]);

            /* Integer dot product */
            vector signed int vsum = vec_dot_ternary_branchless(va, vw);

            /* Horizontal sum - vec_sums puts result in element 3 */
            vsum = vec_sums(vsum, vec_splat_s32(0));

            /* Extract scalar sum */
            int32_t lane_sum;
            vec_ste(vsum, 12, &lane_sum);  /* Element 3 offset = 12 bytes */
            block_sum += lane_sum;
        }

        /* Scale block sum (only FP operation!) */
        total_sum += (int32_t)(block_sum * d_x);
    }

    *s = (float)total_sum;
}

/*
 * Optimized version with loop unrolling
 * Processes 64 elements per iteration (4x unroll)
 */
static inline void ggml_vec_dot_q1_58_q8_altivec_unrolled(
    int n,
    float* restrict s,
    const void* restrict vx,
    size_t bx,
    const void* restrict vy,
    size_t by,
    int nrc)
{
    const block_q1_58* x = (const block_q1_58*)vx;
    const int8_t* y_qs = (const int8_t*)vy;

    const int nb = n / QK_Q1_58;
    int32_t total_sum = 0;

    for (int i = 0; i < nb; i++) {
        const float d_x = GGML_FP16_TO_FP32(x[i].d);

        /* Process in chunks of 64 (4 vectors of 16) */
        vector signed int acc0 = vec_splat_s32(0);
        vector signed int acc1 = vec_splat_s32(0);
        vector signed int acc2 = vec_splat_s32(0);
        vector signed int acc3 = vec_splat_s32(0);

        for (int j = 0; j < QK_Q1_58; j += 64) {
            const int8_t* y_ptr = &y_qs[i * QK_Q1_58 + j];
            const uint8_t* w_ptr = &x[i].packed[j / 4];

            /* Load 4 x 16 activations */
            vector signed char va0 = vec_ld(0, y_ptr);
            vector signed char va1 = vec_ld(16, y_ptr);
            vector signed char va2 = vec_ld(32, y_ptr);
            vector signed char va3 = vec_ld(48, y_ptr);

            /* Unpack 4 x 16 weights */
            vector signed char vw0 = unpack_ternary_16_altivec(w_ptr);
            vector signed char vw1 = unpack_ternary_16_altivec(w_ptr + 4);
            vector signed char vw2 = unpack_ternary_16_altivec(w_ptr + 8);
            vector signed char vw3 = unpack_ternary_16_altivec(w_ptr + 12);

            /* Accumulate using vec_msum (multiply-sum) */
            /* vec_msum: sums products of int8 pairs to int32 */
            acc0 = vec_msum(vw0, va0, acc0);
            acc1 = vec_msum(vw1, va1, acc1);
            acc2 = vec_msum(vw2, va2, acc2);
            acc3 = vec_msum(vw3, va3, acc3);
        }

        /* Combine accumulators */
        vector signed int acc = vec_add(vec_add(acc0, acc1), vec_add(acc2, acc3));

        /* Horizontal sum */
        acc = vec_sums(acc, vec_splat_s32(0));
        int32_t block_sum;
        vec_ste(acc, 12, &block_sum);

        total_sum += (int32_t)(block_sum * d_x);
    }

    *s = (float)total_sum;
}

#else /* !__ALTIVEC__ */

/*
 * Scalar fallback for non-AltiVec systems
 */
static inline void ggml_vec_dot_q1_58_q8_scalar(
    int n,
    float* restrict s,
    const void* restrict vx,
    size_t bx,
    const void* restrict vy,
    size_t by,
    int nrc)
{
    const block_q1_58* x = (const block_q1_58*)vx;
    const int8_t* y_qs = (const int8_t*)vy;

    /* Ternary decode lookup */
    static const int8_t ternary_decode[4] = {-1, 0, 1, 0};

    const int nb = n / QK_Q1_58;
    int32_t total_sum = 0;

    for (int i = 0; i < nb; i++) {
        const float d_x = GGML_FP16_TO_FP32(x[i].d);
        int32_t block_sum = 0;

        for (int j = 0; j < QK_Q1_58; j++) {
            /* Extract 2-bit ternary value */
            int byte_idx = j / 4;
            int bit_offset = (j % 4) * 2;
            int ternary_idx = (x[i].packed[byte_idx] >> bit_offset) & 0x3;
            int8_t w = ternary_decode[ternary_idx];

            /* Get activation */
            int8_t a = y_qs[i * QK_Q1_58 + j];

            /* Integer accumulate (no FP multiply!) */
            block_sum += (int32_t)w * (int32_t)a;
        }

        total_sum += (int32_t)(block_sum * d_x);
    }

    *s = (float)total_sum;
}

#define ggml_vec_dot_q1_58_q8_altivec ggml_vec_dot_q1_58_q8_scalar

#endif /* __ALTIVEC__ */

/*
 * Get type traits for Q1.58
 */
static inline size_t ggml_type_size_q1_58(void) {
    return Q1_58_BLOCK_SIZE;
}

static inline int ggml_blck_size_q1_58(void) {
    return QK_Q1_58;
}

#endif /* GGML_Q1_58_ALTIVEC_H */
