// Big-endian byte swap helpers for GGUF (PowerPC fix)
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define GGUF_IS_BIG_ENDIAN 0  // Using pre-converted BE model
#elif defined(__BIG_ENDIAN__) || defined(__ppc__) || defined(__PPC__) || defined(__powerpc__)
#define GGUF_IS_BIG_ENDIAN 0  // Using pre-converted BE model
#else
#define GGUF_IS_BIG_ENDIAN 0
#endif

static inline uint16_t gguf_bswap16(uint16_t x) { return (x >> 8) | (x << 8); }
static inline uint32_t gguf_bswap32(uint32_t x) {
    return ((x >> 24) & 0x000000FF) | ((x >> 8) & 0x0000FF00) |
           ((x << 8) & 0x00FF0000) | ((x << 24) & 0xFF000000);
}
static inline uint64_t gguf_bswap64(uint64_t x) {
    return ((x >> 56) & 0x00000000000000FFULL) | ((x >> 40) & 0x000000000000FF00ULL) |
           ((x >> 24) & 0x0000000000FF0000ULL) | ((x >> 8) & 0x00000000FF000000ULL) |
           ((x << 8) & 0x000000FF00000000ULL) | ((x << 24) & 0x0000FF0000000000ULL) |
           ((x << 40) & 0x00FF000000000000ULL) | ((x << 56) & 0xFF00000000000000ULL);
}

static bool gguf_fread_el(FILE * file, void * dst, size_t size, size_t * offset) {
    const size_t n = fread(dst, 1, size, file);
    *offset += n;
    return n == size;
}

// Read with byte swapping for scalar values on big-endian
static bool gguf_fread_scalar(FILE * file, void * dst, size_t size, size_t * offset) {
    const size_t n = fread(dst, 1, size, file);
    *offset += n;
#if GGUF_IS_BIG_ENDIAN
    if (n == size) {
        switch (size) {
            case 2: *(uint16_t*)dst = gguf_bswap16(*(uint16_t*)dst); break;
            case 4: *(uint32_t*)dst = gguf_bswap32(*(uint32_t*)dst); break;
            case 8: *(uint64_t*)dst = gguf_bswap64(*(uint64_t*)dst); break;
        }
    }
#endif
    return n == size;
}


static bool gguf_fread_str(FILE * file, struct gguf_str * p, size_t * offset) {
    p->n    = 0;
    p->data = NULL;

    bool ok = true;

    ok = ok && gguf_fread_scalar(file, &p->n,    sizeof(p->n), offset); p->data = calloc(p->n + 1, 1);
    ok = ok && gguf_fread_el(file,  p->data, p->n,         offset);
