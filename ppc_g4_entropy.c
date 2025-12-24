/*
 * PowerPC G4 Entropy Collector for RustChain
 * Collects timing entropy from CPU-specific features
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
#include <string.h>

#ifdef __APPLE__
#include <mach/mach_time.h>
#endif

/* PowerPC timebase register access */
static inline uint64_t read_timebase() {
    uint32_t tbl, tbu0, tbu1;

    do {
        __asm__ volatile ("mftbu %0" : "=r"(tbu0));
        __asm__ volatile ("mftb %0" : "=r"(tbl));
        __asm__ volatile ("mftbu %0" : "=r"(tbu1));
    } while (tbu0 != tbu1);

    return (((uint64_t)tbu0) << 32) | tbl;
}

/* AltiVec SIMD timing (G4 specific) */
static void altivec_entropy(uint64_t *samples, int count) {
#ifdef __ALTIVEC__
    vector unsigned int v1 = {1, 2, 3, 4};
    vector unsigned int v2 = {5, 6, 7, 8};
    vector unsigned int result;

    for (int i = 0; i < count; i++) {
        uint64_t start = read_timebase();

        /* AltiVec operations for timing variance */
        for (int j = 0; j < 100; j++) {
            result = vec_add(v1, v2);
            v1 = vec_xor(result, v2);
            v2 = vec_sl(v1, result);
        }

        uint64_t end = read_timebase();
        samples[i] = end - start;
    }
#else
    /* Fallback for non-AltiVec */
    for (int i = 0; i < count; i++) {
        uint64_t start = read_timebase();
        volatile int x = 0;
        for (int j = 0; j < 1000; j++) x ^= j;
        samples[i] = read_timebase() - start;
    }
#endif
}

/* Cache timing entropy */
static void cache_entropy(uint64_t *samples, int count) {
    char *buffer = malloc(32 * 1024); /* 32KB - L1 cache size on G4 */
    if (!buffer) return;

    for (int i = 0; i < count; i++) {
        /* Flush cache by accessing large memory */
        memset(buffer, i, 32 * 1024);

        uint64_t start = read_timebase();

        /* Time cache-sensitive operations */
        volatile int sum = 0;
        for (int j = 0; j < 1024; j++) {
            sum += buffer[j * 32]; /* Stride to hit different cache lines */
        }

        samples[i] = read_timebase() - start;
    }

    free(buffer);
}

/* Branch predictor entropy */
static void branch_entropy(uint64_t *samples, int count) {
    for (int i = 0; i < count; i++) {
        uint64_t start = read_timebase();

        /* Unpredictable branch pattern */
        volatile int x = 0;
        for (int j = 0; j < 100; j++) {
            if ((j * 7 + i * 13) % 17 > 8) {
                x += j * 3;
            } else {
                x ^= j * 5;
            }
        }

        samples[i] = read_timebase() - start;
    }
}

/* Calculate statistics */
static void calculate_stats(uint64_t *samples, int count, double *mean, double *variance) {
    double sum = 0.0;
    for (int i = 0; i < count; i++) {
        sum += samples[i];
    }
    *mean = sum / count;

    double var_sum = 0.0;
    for (int i = 0; i < count; i++) {
        double diff = samples[i] - *mean;
        var_sum += diff * diff;
    }
    *variance = var_sum / count;
}

int main(int argc, char *argv[]) {
    int sample_count = 1000;
    if (argc > 1) sample_count = atoi(argv[1]);

    uint64_t *altivec_samples = calloc(sample_count, sizeof(uint64_t));
    uint64_t *cache_samples = calloc(sample_count, sizeof(uint64_t));
    uint64_t *branch_samples = calloc(sample_count, sizeof(uint64_t));

    /* Collect entropy from different sources */
    altivec_entropy(altivec_samples, sample_count);
    cache_entropy(cache_samples, sample_count);
    branch_entropy(branch_samples, sample_count);

    /* Calculate statistics */
    double altivec_mean, altivec_var;
    double cache_mean, cache_var;
    double branch_mean, branch_var;

    calculate_stats(altivec_samples, sample_count, &altivec_mean, &altivec_var);
    calculate_stats(cache_samples, sample_count, &cache_mean, &cache_var);
    calculate_stats(branch_samples, sample_count, &branch_mean, &branch_var);

    /* Combine for overall CPU drift metrics */
    double cpu_drift_mean = (altivec_mean + cache_mean + branch_mean) / 3.0;
    double cpu_drift_var = (altivec_var + cache_var + branch_var) / 3.0;

    /* Output for RustChain attestation */
    printf("cpu_drift_mean=%f\n", cpu_drift_mean);
    printf("cpu_drift_var=%f\n", cpu_drift_var);
    printf("altivec_mean=%f\n", altivec_mean);
    printf("cache_mean=%f\n", cache_mean);
    printf("branch_mean=%f\n", branch_mean);
    printf("samples=%d\n", sample_count);

    /* Optional: Write raw samples to file for commitment */
    if (argc > 2 && strcmp(argv[2], "--raw") == 0) {
        FILE *fp = fopen("g4_entropy_raw.bin", "wb");
        if (fp) {
            fwrite(altivec_samples, sizeof(uint64_t), sample_count, fp);
            fwrite(cache_samples, sizeof(uint64_t), sample_count, fp);
            fwrite(branch_samples, sizeof(uint64_t), sample_count, fp);
            fclose(fp);
            printf("raw_file=g4_entropy_raw.bin\n");
        }
    }

    free(altivec_samples);
    free(cache_samples);
    free(branch_samples);

    return 0;
}