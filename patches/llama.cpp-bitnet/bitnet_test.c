/*
 * BitNet I2_S Test for PowerPC G4
 * Standalone test that loads and runs inference on native BitNet model
 *
 * Compile: gcc -O3 -maltivec -o bitnet_test bitnet_test.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef __ALTIVEC__
#include <altivec.h>
#define HAS_ALTIVEC 1
#else
#define HAS_ALTIVEC 0
#endif

/* I2_S Constants */
#define I2S_TYPE 36
#define I2S_BLOCK_WEIGHTS 256
#define I2S_BLOCK_BYTES 64
#define I2S_HEADER_SIZE 32

/* Ternary decode table */
static const int8_t TERNARY_DECODE[4] = {-1, 0, 1, 0};

/* FP16 to float (big-endian) */
static float fp16_to_f32_be(const uint8_t* p) {
    uint16_t h = ((uint16_t)p[0] << 8) | p[1];
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    float val;
    int e;

    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        val = mant / 1024.0f * (1.0f / 16384.0f);
        return sign ? -val : val;
    }
    if (exp == 31) return mant ? NAN : (sign ? -INFINITY : INFINITY);

    val = (1.0f + mant / 1024.0f);
    e = (int)exp - 15;
    while (e > 0) { val *= 2.0f; e--; }
    while (e < 0) { val *= 0.5f; e++; }
    return sign ? -val : val;
}

/* Read big-endian uint32 */
static uint32_t read_be32(const uint8_t* p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8) | p[3];
}

/* Read big-endian uint64 */
static uint64_t read_be64(const uint8_t* p) {
    return ((uint64_t)read_be32(p) << 32) | read_be32(p + 4);
}

/* Unpack ternary weights from I2_S block */
static void unpack_i2s_block(const uint8_t* packed, int8_t* out) {
    int i;
    uint8_t byte;
    for (i = 0; i < 64; i++) {
        byte = packed[i];
        out[i*4 + 0] = TERNARY_DECODE[(byte >> 0) & 3];
        out[i*4 + 1] = TERNARY_DECODE[(byte >> 2) & 3];
        out[i*4 + 2] = TERNARY_DECODE[(byte >> 4) & 3];
        out[i*4 + 3] = TERNARY_DECODE[(byte >> 6) & 3];
    }
}

/* Integer dot product (scalar) */
static int32_t dot_i2s_scalar(const int8_t* w, const int8_t* a, int n) {
    int32_t sum = 0;
    int i;
    for (i = 0; i < n; i++) {
        sum += w[i] * a[i];
    }
    return sum;
}

#if HAS_ALTIVEC
/* AltiVec dot product for ternary weights
 * vec_mule/vec_mulo produce int16, which we properly unpack to int32.
 * Compatible with GCC 4.0.1 on Tiger.
 */
static int32_t dot_i2s_altivec(const int8_t* w, const int8_t* a, int n) {
    vector signed int acc = vec_splat_s32(0);
    vector signed char vw, va;
    vector signed short pe, po;
    vector signed int hi, lo;
    union { vector signed int v; int32_t s[4]; } u;
    int i;

    for (i = 0; i < n; i += 16) {
        vw = vec_ld(0, (const signed char*)&w[i]);
        va = vec_ld(0, (const signed char*)&a[i]);

        /* Multiply even/odd pairs: 16 x int8 -> 8 x int16 each */
        pe = vec_mule(vw, va);  /* 8 int16 products from even indices */
        po = vec_mulo(vw, va);  /* 8 int16 products from odd indices */

        /* Unpack int16 to int32 and sum
         * vec_unpackh: high 4 int16 -> 4 int32
         * vec_unpackl: low 4 int16 -> 4 int32
         */
        hi = vec_unpackh(pe);
        lo = vec_unpackl(pe);
        acc = vec_add(acc, vec_add(hi, lo));

        hi = vec_unpackh(po);
        lo = vec_unpackl(po);
        acc = vec_add(acc, vec_add(hi, lo));
    }

    /* Horizontal sum of 4 int32 lanes */
    acc = vec_sums(acc, vec_splat_s32(0));

    /* Result is in element 3 (big-endian) */
    u.v = acc;
    return u.s[3];
}
#endif

/* Main test */
int main(int argc, char** argv) {
    const char* model_path = argc > 1 ? argv[1] : "ggml-model-i2_s-BE.gguf";
    FILE* f;
    uint8_t header[24];
    uint32_t version;
    uint64_t tensor_count, metadata_count, i2s_offset;
    uint8_t tensor_header[I2S_HEADER_SIZE];
    float scale;
    uint8_t packed[I2S_BLOCK_BYTES];
    int8_t weights[I2S_BLOCK_WEIGHTS] __attribute__((aligned(16)));
    int8_t activations[I2S_BLOCK_WEIGHTS] __attribute__((aligned(16)));
    int neg, zero, pos, i;
    clock_t start, end;
    int iterations = 100000;
    int32_t result;
    double scalar_time;
    int32_t pos_sum, neg_sum;
#if HAS_ALTIVEC
    double altivec_time;
#endif

    printf("BitNet I2_S Test for PowerPC\n");
    printf("============================\n");
    printf("Model: %s\n", model_path);
    printf("AltiVec: %s\n\n", HAS_ALTIVEC ? "YES" : "NO");

    f = fopen(model_path, "rb");
    if (!f) {
        fprintf(stderr, "Cannot open model: %s\n", model_path);
        return 1;
    }

    /* Read GGUF header */
    if (fread(header, 1, 24, f) != 24) {
        fprintf(stderr, "Cannot read header\n");
        fclose(f);
        return 1;
    }

    /* Verify magic */
    if (memcmp(header, "GGUF", 4) != 0) {
        fprintf(stderr, "Invalid GGUF magic\n");
        fclose(f);
        return 1;
    }

    version = read_be32(header + 4);
    tensor_count = read_be64(header + 8);
    metadata_count = read_be64(header + 16);

    printf("GGUF Version: %u\n", version);
    printf("Tensors: %llu\n", (unsigned long long)tensor_count);
    printf("Metadata: %llu\n\n", (unsigned long long)metadata_count);

    /* Seek to first I2_S tensor data (hardcoded offset from analysis) */
    /* data_start=8351360, first I2_S offset=656680960 */
    i2s_offset = 8351360 + 656680960;

    printf("Seeking to first I2_S tensor at offset %llu...\n", (unsigned long long)i2s_offset);
    fseek(f, i2s_offset, SEEK_SET);

    /* Read tensor header */
    if (fread(tensor_header, 1, I2S_HEADER_SIZE, f) != I2S_HEADER_SIZE) {
        fprintf(stderr, "Cannot read tensor header\n");
        fclose(f);
        return 1;
    }

    scale = fp16_to_f32_be(tensor_header);
    printf("Tensor scale: %.4f\n\n", scale);

    /* Read first block */
    if (fread(packed, 1, I2S_BLOCK_BYTES, f) != I2S_BLOCK_BYTES) {
        fprintf(stderr, "Cannot read block\n");
        fclose(f);
        return 1;
    }

    /* Unpack ternary weights */
    unpack_i2s_block(packed, weights);

    /* Count distribution */
    neg = 0; zero = 0; pos = 0;
    for (i = 0; i < I2S_BLOCK_WEIGHTS; i++) {
        if (weights[i] == -1) neg++;
        else if (weights[i] == 0) zero++;
        else if (weights[i] == 1) pos++;
    }

    printf("Weight distribution (first block):\n");
    printf("  -1: %d (%.1f%%)\n", neg, 100.0 * neg / I2S_BLOCK_WEIGHTS);
    printf("   0: %d (%.1f%%)\n", zero, 100.0 * zero / I2S_BLOCK_WEIGHTS);
    printf("  +1: %d (%.1f%%)\n\n", pos, 100.0 * pos / I2S_BLOCK_WEIGHTS);

    /* Create test activations */
    srand(42);
    for (i = 0; i < I2S_BLOCK_WEIGHTS; i++) {
        activations[i] = (rand() % 256) - 128;
    }

    /* Benchmark dot product */
    printf("=== Dot Product Benchmark ===\n");

    /* Scalar */
    start = clock();
    for (i = 0; i < iterations; i++) {
        result = dot_i2s_scalar(weights, activations, I2S_BLOCK_WEIGHTS);
    }
    end = clock();
    scalar_time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Scalar:  %d iterations in %.3f s (%.2f M ops/s)\n",
           iterations, scalar_time, iterations * I2S_BLOCK_WEIGHTS / scalar_time / 1e6);
    printf("         Result: %d\n", result);

#if HAS_ALTIVEC
    /* AltiVec */
    start = clock();
    for (i = 0; i < iterations; i++) {
        result = dot_i2s_altivec(weights, activations, I2S_BLOCK_WEIGHTS);
    }
    end = clock();
    altivec_time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("AltiVec: %d iterations in %.3f s (%.2f M ops/s)\n",
           iterations, altivec_time, iterations * I2S_BLOCK_WEIGHTS / altivec_time / 1e6);
    printf("         Result: %d\n", result);
    printf("         Speedup: %.2fx\n", scalar_time / altivec_time);
#endif

    /* Verify ternary math */
    printf("\n=== Ternary Math Verification ===\n");
    pos_sum = 0; neg_sum = 0;
    for (i = 0; i < I2S_BLOCK_WEIGHTS; i++) {
        if (weights[i] == 1) pos_sum += activations[i];
        else if (weights[i] == -1) neg_sum += activations[i];
    }
    printf("Sum(a where w=+1): %d\n", pos_sum);
    printf("Sum(a where w=-1): %d\n", neg_sum);
    printf("Ternary result (pos-neg): %d\n", pos_sum - neg_sum);
    printf("Dot product result: %d\n", result);

    if (pos_sum - neg_sum == result) {
        printf("\nPASS: Ternary math verified on PowerPC G4!\n");
    } else {
        printf("\nFAIL: Result mismatch!\n");
    }

    fclose(f);

    printf("\n============================\n");
    printf("BitNet I2_S test complete!\n");

    return 0;
}
