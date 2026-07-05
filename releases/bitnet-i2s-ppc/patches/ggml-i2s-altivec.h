/*
 * AltiVec/VSX Kernels for Microsoft BitNet I2_S (Type 36)
 *
 * Native ternary inference: weights are {-1, 0, +1}
 * No per-block scale - pure integer arithmetic until final output!
 *
 * I2_S Format (reverse-engineered):
 *   - 64 bytes packed data per 256 weights
 *   - 32 bytes per-tensor header (scale factor)
 *   - 2 bits per weight: 00=-1, 01=0, 10=+1, 11=unused
 *
 * Key insight: With ternary weights, matrix multiply becomes:
 *   y = sum(w_i * x_i) where w_i ∈ {-1, 0, +1}
 *     = sum(x_i where w_i=+1) - sum(x_i where w_i=-1)
 *
 * This is pure integer addition/subtraction - perfect for AltiVec!
 *
 * Author: PPC-Tiger-Tools
 * Date: December 2025
 */

#ifndef GGML_I2S_ALTIVEC_H
#define GGML_I2S_ALTIVEC_H

#include <altivec.h>
#include <stdint.h>

/* I2_S Format Constants */
#define I2S_TYPE_ID         36
#define I2S_BLOCK_WEIGHTS   256
#define I2S_BLOCK_BYTES     64
#define I2S_TENSOR_HEADER   32

/* Block structure for I2_S
 * Note: Unlike Q4_K, I2_S has no per-block scale.
 * Scale is stored in tensor header (per-tensor).
 */
typedef struct {
    uint8_t qs[64];  /* 256 weights @ 2 bits = 64 bytes */
} block_i2s;

/* Tensor header (at start of each I2_S tensor) */
typedef struct {
    union {
        uint16_t scale_f16;  /* FP16 scale factor */
        uint8_t  raw[32];    /* Raw header bytes */
    };
} i2s_tensor_header;

#define QK_I2S  256
#define QR_I2S  1

/*
 * Unpack 16 ternary weights from 4 packed bytes using vec_perm
 *
 * Input: 4 bytes = 16 weights @ 2 bits each
 * Output: 16 x int8 values in {-1, 0, +1}
 *
 * Encoding: 00 -> -1, 01 -> 0, 10 -> +1, 11 -> 0 (unused)
 */
static inline vector signed char unpack_i2s_16(const uint8_t* packed) {
    /* Lookup table: 2-bit index -> ternary value */
    static const vector signed char lut = {
        -1, 0, 1, 0,   /* For indices 0,1,2,3 repeated 4x */
        -1, 0, 1, 0,
        -1, 0, 1, 0,
        -1, 0, 1, 0
    };

    /* Extract 2-bit indices from 4 packed bytes */
    uint8_t b0 = packed[0], b1 = packed[1], b2 = packed[2], b3 = packed[3];

    vector unsigned char indices = (vector unsigned char){
        (b0 >> 0) & 3, (b0 >> 2) & 3, (b0 >> 4) & 3, (b0 >> 6) & 3,
        (b1 >> 0) & 3, (b1 >> 2) & 3, (b1 >> 4) & 3, (b1 >> 6) & 3,
        (b2 >> 0) & 3, (b2 >> 2) & 3, (b2 >> 4) & 3, (b2 >> 6) & 3,
        (b3 >> 0) & 3, (b3 >> 2) & 3, (b3 >> 4) & 3, (b3 >> 6) & 3
    };

    /* Use vec_perm for fast lookup - G4's specialty! */
    return vec_perm(lut, lut, indices);
}

/*
 * Integer dot product: ternary weights × int8 activations
 *
 * Since weights are {-1, 0, +1}, this is purely addition/subtraction!
 * Result is int32 - no floating point until final scale.
 *
 * IMPORTANT: Input arrays must be 16-byte aligned for vec_ld!
 */
static inline vector signed int vec_dot_i2s_q8(
    vector signed char weights,     /* 16 x ternary {-1,0,+1} */
    vector signed char activations) /* 16 x int8 quantized activations */
{
    /*
     * Method: Use vec_mule/vec_mulo for signed multiply
     * -1 * x = -x, 0 * x = 0, +1 * x = x
     * This is mathematically correct for ternary!
     *
     * vec_mule/vec_mulo produce int16 products which we properly
     * unpack to int32 using vec_unpackh/vec_unpackl.
     */

    /* Multiply even/odd pairs: 16 x int8 -> 8 x int16 each */
    vector signed short prod_even = vec_mule(weights, activations);
    vector signed short prod_odd = vec_mulo(weights, activations);

    /* Unpack int16 to int32 and sum
     * vec_unpackh: high 4 int16 -> 4 int32 (sign-extended)
     * vec_unpackl: low 4 int16 -> 4 int32 (sign-extended)
     */
    vector signed int sum = vec_splat_s32(0);

    sum = vec_add(sum, vec_unpackh(prod_even));
    sum = vec_add(sum, vec_unpackl(prod_even));
    sum = vec_add(sum, vec_unpackh(prod_odd));
    sum = vec_add(sum, vec_unpackl(prod_odd));

    return sum;
}

/*
 * Alternative: Branchless conditional accumulation
 * May be faster on some G4 variants
 */
static inline vector signed int vec_dot_i2s_q8_branchless(
    vector signed char weights,
    vector signed char activations)
{
    /* Create masks for +1 and -1 weights */
    vector bool char is_pos = vec_cmpeq(weights, vec_splat_s8(1));
    vector bool char is_neg = vec_cmpeq(weights, vec_splat_s8(-1));

    /* Select activations where weight is +1 */
    vector signed char pos_vals = vec_sel(vec_splat_s8(0), activations, is_pos);

    /* Select negated activations where weight is -1 */
    vector signed char neg_activations = vec_sub(vec_splat_s8(0), activations);
    vector signed char neg_vals = vec_sel(vec_splat_s8(0), neg_activations, is_neg);

    /* Sum: pos_vals contributes +, neg_vals already negated */
    vector signed char combined = vec_add(pos_vals, neg_vals);

    /* Horizontal sum to int32 */
    return vec_sum4s(combined, vec_splat_s32(0));
}

/*
 * Horizontal sum: vector int32 -> scalar int32
 */
static inline int32_t vec_reduce_i32(vector signed int v) {
    /* Sum all 4 elements */
    vector signed int zero = vec_splat_s32(0);
    v = vec_sums(v, zero);  /* Result in element 3 */

    /* Extract element 3 */
    union { vector signed int v; int32_t s[4]; } u;
    u.v = v;
    return u.s[3];
}

/*
 * Full I2_S block dot product with Q8 activations
 *
 * Computes: sum(i2s_weights[0:256] * q8_activations[0:256])
 * Result is integer - apply tensor scale at the end!
 *
 * IMPORTANT: activations must be 16-byte aligned!
 */
static inline int32_t ggml_vec_dot_i2s_q8_block(
    const block_i2s* weights,
    const int8_t* activations)  /* 256 Q8 values, must be 16-byte aligned */
{
    vector signed int acc = vec_splat_s32(0);
    int j;

    /* Process 16 weights at a time (4 bytes packed -> 16 ternary) */
    for (j = 0; j < 256; j += 16) {
        /* Unpack 16 ternary weights */
        vector signed char vw = unpack_i2s_16(&weights->qs[j / 4]);

        /* Load 16 int8 activations (requires 16-byte alignment) */
        vector signed char va = vec_ld(0, (const signed char*)&activations[j]);

        /* Integer dot product */
        vector signed int partial = vec_dot_i2s_q8(vw, va);

        /* Accumulate */
        acc = vec_add(acc, partial);
    }

    /* Reduce to scalar */
    return vec_reduce_i32(acc);
}

/*
 * Full tensor dot product for I2_S
 *
 * Input:
 *   vx - I2_S tensor data (with 32-byte header)
 *   vy - Q8_K quantized activations
 *   n  - number of elements (multiple of 256)
 *
 * Output:
 *   s  - dot product result
 */
static inline void ggml_vec_dot_i2s_q8_K(
    int n,
    float* restrict s,
    const void* restrict vx,
    const void* restrict vy)
{
    const uint8_t* x_raw = (const uint8_t*)vx;
    const int8_t* y = (const int8_t*)vy;

    /* Read tensor scale from header (first 2 bytes as FP16) */
    uint16_t scale_bits = *(const uint16_t*)x_raw;
    /* Convert FP16 to float (simplified) */
    union { uint16_t u; int16_t i; } scale_u;
    scale_u.u = scale_bits;
    /* TODO: Proper FP16 to F32 conversion for big-endian */
    float tensor_scale = (float)scale_u.i / 32768.0f;  /* Rough approx */

    /* Skip header to get to block data */
    const block_i2s* blocks = (const block_i2s*)(x_raw + I2S_TENSOR_HEADER);

    int nb = n / I2S_BLOCK_WEIGHTS;
    int64_t total_sum = 0;  /* Use int64 to avoid overflow */

    for (int i = 0; i < nb; i++) {
        int32_t block_sum = ggml_vec_dot_i2s_q8_block(
            &blocks[i],
            &y[i * I2S_BLOCK_WEIGHTS]
        );
        total_sum += block_sum;
    }

    /* Apply tensor-level scale only at the end! */
    *s = (float)total_sum * tensor_scale;
}

/*
 * Optimized path: Skip zeros entirely
 *
 * Since ~40% of weights are zero, we can skip those computations.
 * This version counts and skips zeros for even faster execution.
 */
static inline int32_t ggml_vec_dot_i2s_q8_sparse(
    const block_i2s* weights,
    const int8_t* activations)
{
    int32_t sum = 0;

    /* Process byte by byte for zero-skipping */
    for (int i = 0; i < 64; i++) {
        uint8_t packed = weights->qs[i];

        /* Skip if all zeros (01010101 = 0x55 when all weights are 0) */
        if (packed == 0x55) continue;

        /* Extract 4 weights from this byte */
        for (int j = 0; j < 4; j++) {
            int w = (packed >> (j * 2)) & 0x03;
            int8_t a = activations[i * 4 + j];

            if (w == 0) {       /* -1 */
                sum -= a;
            } else if (w == 2) { /* +1 */
                sum += a;
            }
            /* w == 1 or 3: zero, skip */
        }
    }

    return sum;
}

/*
 * Dequantize I2_S block to float (for debugging/verification)
 */
static inline void dequantize_block_i2s(
    const block_i2s* x,
    float* y,
    float scale)
{
    for (int j = 0; j < 64; j++) {
        uint8_t packed = x->qs[j];
        for (int k = 0; k < 4; k++) {
            int idx = j * 4 + k;
            int bits = (packed >> (k * 2)) & 0x03;
            int ternary;
            switch (bits) {
                case 0: ternary = -1; break;
                case 1: ternary =  0; break;
                case 2: ternary = +1; break;
                default: ternary = 0; break;  /* 3 = unused */
            }
            y[idx] = (float)ternary * scale;
        }
    }
}

/*
 * Full row dequantization for I2_S tensor
 */
static inline void dequantize_row_i2s(
    const void* vx,
    float* y,
    int64_t k)
{
    const uint8_t* x_raw = (const uint8_t*)vx;

    /* Get scale from header */
    uint16_t scale_bits = *(const uint16_t*)x_raw;
    float scale = (float)(scale_bits) / 32768.0f;  /* Approx FP16->F32 */

    const block_i2s* blocks = (const block_i2s*)(x_raw + I2S_TENSOR_HEADER);
    int nb = k / I2S_BLOCK_WEIGHTS;

    for (int i = 0; i < nb; i++) {
        dequantize_block_i2s(&blocks[i], &y[i * I2S_BLOCK_WEIGHTS], scale);
    }
}

/*
 * Row-wise multiply-add: y += I2_S_row * x
 * Core operation for matrix-vector multiplication
 */
static inline void ggml_axpy_i2s_q8(
    int n,
    float* restrict y,     /* Output accumulator [n] */
    const void* restrict vw, /* I2_S weight row */
    const void* restrict vx, /* Q8 input vector */
    float x_scale)           /* Q8 input scale */
{
    /* This performs: y += scale * (W * x) where W is ternary */
    float dot;
    ggml_vec_dot_i2s_q8_K(n, &dot, vw, vx);

    /* Accumulate - for matrix ops, this would be per-output element */
    y[0] += dot * x_scale;
}

#endif /* GGML_I2S_ALTIVEC_H */
