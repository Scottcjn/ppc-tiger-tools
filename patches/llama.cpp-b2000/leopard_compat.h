/* Leopard (macOS 10.5) compatibility shims for llama.cpp */
#ifndef LEOPARD_COMPAT_H
#define LEOPARD_COMPAT_H

#ifdef __APPLE__
#include <AvailabilityMacros.h>
#include <sys/time.h>
#include <stdlib.h>
#include <errno.h>

/* clock_gettime not available before macOS 10.12 */
#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
#define CLOCK_REALTIME 0

static inline int clock_gettime(int clk_id, struct timespec *ts) {
    (void)clk_id;
    struct timeval tv;
    gettimeofday(&tv, NULL);
    ts->tv_sec = tv.tv_sec;
    ts->tv_nsec = tv.tv_usec * 1000;
    return 0;
}
#endif

/*
 * posix_memalign - use valloc on older macOS
 * valloc allocates page-aligned memory which satisfies any alignment
 */
#if !defined(HAVE_POSIX_MEMALIGN) && (MAC_OS_X_VERSION_MIN_REQUIRED < 1060)
static inline int posix_memalign(void **memptr, size_t alignment, size_t size) {
    (void)alignment; /* valloc always page-aligns */
    *memptr = valloc(size);
    if (*memptr == NULL) {
        return ENOMEM;
    }
    return 0;
}
#endif

#endif /* __APPLE__ */
#endif /* LEOPARD_COMPAT_H */
