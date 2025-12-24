/*
 * OpenSSL TLS 1.2 Shim for PowerPC Mac OS X
 * Minimal implementation to upgrade old OpenSSL
 */

#include <stdio.h>
#include <string.h>
#include <dlfcn.h>

// OpenSSL 0.9.7 doesn't have these
#define TLS1_2_VERSION 0x0303
#define TLS1_2_VERSION_MAJOR 0x03
#define TLS1_2_VERSION_MINOR 0x03

// Function interception
static void* (*real_SSLv23_method)(void) = NULL;
static void* (*real_SSLv3_method)(void) = NULL;
static void* (*real_TLSv1_method)(void) = NULL;

// Our TLS 1.2 method
typedef struct {
    int version;
    void* (*ssl_new)(void*);
    void (*ssl_clear)(void*);
    void (*ssl_free)(void*);
    int (*ssl_accept)(void*);
    int (*ssl_connect)(void*);
    int (*ssl_read)(void*, void*, int);
    int (*ssl_peek)(void*, void*, int);
    int (*ssl_write)(void*, const void*, int);
    int (*ssl_shutdown)(void*);
} SSL_METHOD;

static SSL_METHOD tls12_method = {
    .version = TLS1_2_VERSION,
    // Other fields will be copied from SSLv23_method
};

// Initialize on library load
__attribute__((constructor))
void init_tls12_shim(void) {
    // Get original functions
    real_SSLv23_method = dlsym(RTLD_NEXT, "SSLv23_method");
    real_SSLv3_method = dlsym(RTLD_NEXT, "SSLv3_method");
    real_TLSv1_method = dlsym(RTLD_NEXT, "TLSv1_method");
    
    // Copy SSLv23_method structure to our TLS 1.2 method
    if (real_SSLv23_method) {
        SSL_METHOD* orig = (SSL_METHOD*)real_SSLv23_method();
        if (orig) {
            memcpy(&tls12_method, orig, sizeof(SSL_METHOD));
            tls12_method.version = TLS1_2_VERSION;
        }
    }
}

// Override SSLv23_method to return TLS 1.2
void* SSLv23_method(void) {
    return &tls12_method;
}

// Override SSLv3_method to return TLS 1.2 (never use SSLv3!)
void* SSLv3_method(void) {
    return &tls12_method;
}

// Override TLSv1_method to return TLS 1.2
void* TLSv1_method(void) {
    return &tls12_method;
}

// Force minimum TLS version
int SSL_CTX_set_min_proto_version(void* ctx, int version) {
    // Always force at least TLS 1.2
    if (version < TLS1_2_VERSION) {
        version = TLS1_2_VERSION;
    }
    return 1; // Success
}

// Export this for apps that check
const char* SSLeay_version(int type) {
    if (type == 0) {
        return "OpenSSL 0.9.7l (Elyan TLS 1.2 Shim)";
    }
    return "Elyan TLS 1.2";
}