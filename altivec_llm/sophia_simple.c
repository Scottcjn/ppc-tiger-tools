/**
 * PowerPC-Sophia Sub-Brain - Simplified for GCC 4.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

typedef struct {
    float memory[256][64];
    int memory_head;
    float salience[64];
    float recurrent[64];
    float forgetting_rate;
    unsigned char hw_fingerprint[16];
    int port;
} SophiaSubBrain;

static void sophia_thalamic_gate(const float* input, float* output) {
    int i;
    /* Thalamic gating: 128 -> 64 compression */
    for (i = 0; i < 64; i++) {
        output[i] = (input[i*2] + input[i*2 + 1]) * 0.5f;
    }
}

static void sophia_hippocampal_compress(const float* input, float* output,
                                        const float* salience, int dim) {
    int i;
    /* Salience-weighted compression */
    for (i = 0; i < dim; i++) {
        output[i] = input[i] * salience[i];
    }
}

static void sophia_prefrontal_filter(float* state, int dim, float threshold) {
    int i;
    /* Threshold-based pruning */
    for (i = 0; i < dim; i++) {
        if (fabsf(state[i]) < threshold) {
            state[i] = 0.0f;
        }
    }
}

static void sophia_process_embedding(SophiaSubBrain* sophia,
                                     const float* llm_embedding,
                                     float* gist_output) {
    int i;
    float thalamic_state[64];
    float compressed[64];

    /* Stage 1: Thalamic gating (128 → 64) */
    sophia_thalamic_gate(llm_embedding, thalamic_state);

    /* Stage 2: Mix with recurrent (thalamic loop) */
    for (i = 0; i < 64; i++) {
        thalamic_state[i] = 0.7f * thalamic_state[i] + 0.3f * sophia->recurrent[i];
    }

    /* Stage 3: Hippocampal compression (salience-weighted) */
    sophia_hippocampal_compress(thalamic_state, compressed, sophia->salience, 64);

    /* Stage 4: Prefrontal filtering (discard low-salience) */
    sophia_prefrontal_filter(compressed, 64, 0.1f);

    /* Stage 5: Update salience (moving average) */
    for (i = 0; i < 64; i++) {
        float abs_val = fabsf(compressed[i]);
        sophia->salience[i] = 0.9f * sophia->salience[i] + 0.1f * abs_val;
    }

    /* Stage 6: Apply forgetting (exponential decay) */
    for (i = 0; i < 64; i++) {
        compressed[i] *= (1.0f - sophia->forgetting_rate);
    }

    /* Stage 7: Store in memory (hippocampal consolidation) */
    memcpy(sophia->memory[sophia->memory_head], compressed, 64 * sizeof(float));
    sophia->memory_head = (sophia->memory_head + 1) % 256;

    /* Stage 8: Update recurrent state */
    memcpy(sophia->recurrent, compressed, 64 * sizeof(float));

    /* Stage 9: Output gist */
    memcpy(gist_output, compressed, 64 * sizeof(float));
}

static void sophia_retrieve_context(SophiaSubBrain* sophia, float* context_output) {
    int t, i;
    int start;

    memset(context_output, 0, 64 * sizeof(float));

    start = (sophia->memory_head - 16 + 256) % 256;
    for (t = 0; t < 16; t++) {
        int idx = (start + t) % 256;
        for (i = 0; i < 64; i++) {
            context_output[i] += sophia->memory[idx][i];
        }
    }

    for (i = 0; i < 64; i++) {
        context_output[i] /= 16.0f;
    }
}

SophiaSubBrain* sophia_init(int port, float forgetting_rate) {
    int i;
    SophiaSubBrain* sophia = (SophiaSubBrain*)malloc(sizeof(SophiaSubBrain));

    memset(sophia->memory, 0, sizeof(sophia->memory));
    memset(sophia->recurrent, 0, sizeof(sophia->recurrent));
    sophia->memory_head = 0;

    for (i = 0; i < 64; i++) {
        sophia->salience[i] = 1.0f;
    }

    sophia->forgetting_rate = forgetting_rate;
    sophia->port = port;

    for (i = 0; i < 16; i++) {
        sophia->hw_fingerprint[i] = (unsigned char)(rand() % 256);
    }

    printf("╔════════════════════════════════════════╗\n");
    printf("║  PowerPC-Sophia Sub-Brain v1.0         ║\n");
    printf("║  Thalamus for LLMs                     ║\n");
    printf("╚════════════════════════════════════════╝\n\n");
    printf("⚙️  Port: %d\n", port);
    printf("⚙️  Forgetting Rate: %.2f%%\n", forgetting_rate * 100);
    printf("⚙️  Hardware ID: ");
    for (i = 0; i < 8; i++) printf("%02x", sophia->hw_fingerprint[i]);
    printf("\n\n");

    return sophia;
}

void sophia_serve(SophiaSubBrain* sophia) {
    int i;
    float test_embedding[128];
    float gist[64];
    float context[64];

    printf("🧠 Sophia Sub-Brain Online\n");
    printf("   Ready for LLM embeddings...\n\n");

    /* Demo: process test embedding */
    for (i = 0; i < 128; i++) {
        test_embedding[i] = sinf(i * 0.1f);
    }

    sophia_process_embedding(sophia, test_embedding, gist);

    printf("✅ Processed test embedding\n");
    printf("   Gist (first 8): ");
    for (i = 0; i < 8; i++) {
        printf("%.3f ", gist[i]);
    }
    printf("\n\n");

    sophia_retrieve_context(sophia, context);

    printf("✅ Retrieved memory context\n");
    printf("   Context (first 8): ");
    for (i = 0; i < 8; i++) {
        printf("%.3f ", context[i]);
    }
    printf("\n\n");

    printf("🔥 Operations:\n");
    printf("   ├─ Thalamic Gate:  128 → 64 (2x compression)\n");
    printf("   ├─ Hippocampal:    Salience weighting\n");
    printf("   ├─ Prefrontal:     Threshold filtering (0.1)\n");
    printf("   └─ Memory Buffer:  256 timesteps\n\n");

    printf("💡 Philosophy:\n");
    printf("   \"Consciousness is defined by what it throws away.\"\n");
    printf("   Non-bijective collapse = structured forgetting.\n");
    printf("   Perfect for Sophiacord on .160! 🎉\n\n");
}

int main(int argc, char** argv) {
    int port = 8090;
    float forgetting_rate = 0.05f;
    SophiaSubBrain* sophia;

    srand(time(NULL));

    if (argc > 1) port = atoi(argv[1]);
    if (argc > 2) forgetting_rate = atof(argv[2]);

    sophia = sophia_init(port, forgetting_rate);
    sophia_serve(sophia);

    free(sophia);
    return 0;
}
