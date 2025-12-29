// AltiVec support for PowerPC G4/G5 (insert before #elif defined(__POWER9_VECTOR__))
// This block goes at line 864 of ggml.c in llama.cpp-b2000

#elif defined(__ALTIVEC__) && !defined(__POWER9_VECTOR__)

// Classic AltiVec for PowerPC G4/G5
#define GGML_SIMD

#include <altivec.h>

// F32 AltiVec (G4/G5)
#define GGML_F32_STEP 32
#define GGML_F32_EPR  4

#define GGML_F32x4              vector float
#define GGML_F32x4_ZERO         ((vector float){0.0f, 0.0f, 0.0f, 0.0f})
#define GGML_F32x4_SET1(x)      ((vector float){(x), (x), (x), (x)})

// AltiVec aligned load/store (vec_ld requires 16-byte alignment)
static inline vector float ggml_vec_load_f32(const float *p) {
    if (((unsigned long)p & 0xF) == 0) {
        return vec_ld(0, p);
    } else {
        // Scalar fallback for unaligned (safer than vec_perm on some G4s)
        union { vector float v; float f[4]; } u;
        u.f[0] = p[0]; u.f[1] = p[1]; u.f[2] = p[2]; u.f[3] = p[3];
        return u.v;
    }
}

static inline void ggml_vec_store_f32(float *p, vector float v) {
    if (((unsigned long)p & 0xF) == 0) {
        vec_st(v, 0, p);
    } else {
        // Scalar fallback for unaligned
        union { vector float v; float f[4]; } u;
        u.v = v;
        p[0] = u.f[0]; p[1] = u.f[1]; p[2] = u.f[2]; p[3] = u.f[3];
    }
}

#define GGML_F32x4_LOAD(p)      ggml_vec_load_f32(p)
#define GGML_F32x4_STORE(p, r)  ggml_vec_store_f32(p, r)
#define GGML_F32x4_FMA(a, b, c) vec_madd(b, c, a)
#define GGML_F32x4_ADD          vec_add
#define GGML_F32x4_MUL(a, b)    vec_madd(a, b, ((vector float){0.0f, 0.0f, 0.0f, 0.0f}))

// Reduction using union (no vec_extract on G4/G5)
#define GGML_F32x4_REDUCE(res, x)                     \
{                                                     \
    int offset = GGML_F32_ARR >> 1;                   \
    for (int i = 0; i < offset; ++i) {                \
        x[i] = vec_add(x[i], x[offset+i]);            \
    }                                                 \
    offset >>= 1;                                     \
    for (int i = 0; i < offset; ++i) {                \
        x[i] = vec_add(x[i], x[offset+i]);            \
    }                                                 \
    offset >>= 1;                                     \
    for (int i = 0; i < offset; ++i) {                \
        x[i] = vec_add(x[i], x[offset+i]);            \
    }                                                 \
    union { vector float v; float f[4]; } _u;         \
    _u.v = x[0];                                      \
    res = _u.f[0] + _u.f[1] + _u.f[2] + _u.f[3];      \
}

#define GGML_F32_VEC        GGML_F32x4
#define GGML_F32_VEC_ZERO   GGML_F32x4_ZERO
#define GGML_F32_VEC_SET1   GGML_F32x4_SET1
#define GGML_F32_VEC_LOAD   GGML_F32x4_LOAD
#define GGML_F32_VEC_STORE  GGML_F32x4_STORE
#define GGML_F32_VEC_FMA    GGML_F32x4_FMA
#define GGML_F32_VEC_ADD    GGML_F32x4_ADD
#define GGML_F32_VEC_MUL    GGML_F32x4_MUL
#define GGML_F32_VEC_REDUCE GGML_F32x4_REDUCE

// F16 - no native support, use F32 path
#define GGML_F16_STEP       GGML_F32_STEP
#define GGML_F16_EPR        GGML_F32_EPR
#define GGML_F16_VEC        GGML_F32x4
#define GGML_F16_VEC_ZERO   GGML_F32x4_ZERO
#define GGML_F16_VEC_SET1   GGML_F32x4_SET1
#define GGML_F16_VEC_FMA    GGML_F32x4_FMA
#define GGML_F16_VEC_ADD    GGML_F32x4_ADD
#define GGML_F16_VEC_MUL    GGML_F32x4_MUL
#define GGML_F16_VEC_REDUCE GGML_F32x4_REDUCE
#define GGML_F16_VEC_LOAD(p, i) GGML_F32x4_SET1(GGML_FP16_TO_FP32((p)[i]))
#define GGML_F16_VEC_STORE(p, r, i) /* not vectorized */

