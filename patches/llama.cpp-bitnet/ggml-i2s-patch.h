/*
 * I2_S (BitNet) Support Patch for llama.cpp b2000
 * Add this to ggml.h after the GGML_TYPE enum
 *
 * For PowerPC G4/G5 big-endian systems
 */

#ifndef GGML_I2S_PATCH_H
#define GGML_I2S_PATCH_H

/* Add to ggml.h enum ggml_type (after existing types, before COUNT) */
#define GGML_TYPE_I2_S 36

/* I2_S block structure - 256 weights packed into 64 bytes */
#define QK_I2S 256

typedef struct {
    uint8_t qs[64];  /* 256 ternary weights @ 2 bits each */
} block_i2s;

/* Per-tensor header for I2_S (32 bytes at start of each tensor) */
typedef struct {
    uint16_t scale;     /* FP16 scale factor */
    uint8_t  pad[30];   /* Reserved */
} i2s_tensor_header_t;

#define I2S_TENSOR_HEADER_SIZE 32

/*
 * Dequantize I2_S block to float
 * Encoding: 00=-1, 01=0, 10=+1, 11=unused
 */
static inline void dequantize_block_i2s(const block_i2s* x, float* y, float scale) {
    static const int8_t decode[4] = {-1, 0, 1, 0};
    int j;
    uint8_t packed;

    for (j = 0; j < 64; j++) {
        packed = x->qs[j];
        y[j*4 + 0] = scale * decode[(packed >> 0) & 3];
        y[j*4 + 1] = scale * decode[(packed >> 2) & 3];
        y[j*4 + 2] = scale * decode[(packed >> 4) & 3];
        y[j*4 + 3] = scale * decode[(packed >> 6) & 3];
    }
}

/*
 * Integer dot product for I2_S × Q8
 * Returns sum of (ternary_weight * int8_activation)
 */
static inline int32_t vec_dot_i2s_q8_block(const block_i2s* w, const int8_t* a) {
    static const int8_t decode[4] = {-1, 0, 1, 0};
    int32_t sum = 0;
    int j;
    uint8_t packed;
    int8_t w0, w1, w2, w3;

    for (j = 0; j < 64; j++) {
        packed = w->qs[j];
        w0 = decode[(packed >> 0) & 3];
        w1 = decode[(packed >> 2) & 3];
        w2 = decode[(packed >> 4) & 3];
        w3 = decode[(packed >> 6) & 3];

        sum += w0 * a[j*4 + 0];
        sum += w1 * a[j*4 + 1];
        sum += w2 * a[j*4 + 2];
        sum += w3 * a[j*4 + 3];
    }

    return sum;
}

/*
 * FP16 to float conversion (big-endian safe)
 */
static inline float fp16_to_fp32_be(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    float val;
    int e, i;

    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        /* Denormal */
        val = mant / 1024.0f;
        val *= (1.0f / 16384.0f);  /* 2^-14 */
        return sign ? -val : val;
    } else if (exp == 31) {
        if (mant == 0) return sign ? -INFINITY : INFINITY;
        return NAN;
    }

    val = (1.0f + mant / 1024.0f);
    e = (int)exp - 15;
    if (e > 0) {
        for (i = 0; i < e; i++) val *= 2.0f;
    } else {
        for (i = 0; i < -e; i++) val *= 0.5f;
    }
    return sign ? -val : val;
}

/*
 * Get scale from I2_S tensor header (handles big-endian)
 */
static inline float i2s_get_tensor_scale(const void* tensor_data) {
    const uint8_t* p = (const uint8_t*)tensor_data;
#ifdef __BIG_ENDIAN__
    /* Big-endian: bytes are already in correct order after conversion */
    uint16_t scale_bits = (p[0] << 8) | p[1];
#else
    /* Little-endian */
    uint16_t scale_bits = p[0] | (p[1] << 8);
#endif
    return fp16_to_fp32_be(scale_bits);
}

/*
 * Get pointer to I2_S blocks (skip 32-byte header)
 */
static inline const block_i2s* i2s_get_blocks(const void* tensor_data) {
    return (const block_i2s*)((const uint8_t*)tensor_data + I2S_TENSOR_HEADER_SIZE);
}

#endif /* GGML_I2S_PATCH_H */
