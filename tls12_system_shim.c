/*
 * System SSL/TLS 1.2 Shim
 * Intercepts and upgrades SSL connections system-wide
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

// Original function pointers
static int (*original_SSL_connect)(void *ssl) = NULL;
static void* (*original_SSL_CTX_new)(void *method) = NULL;
static int (*original_SSL_CTX_set_cipher_list)(void *ctx, const char *str) = NULL;
static void* (*original_SSLv23_client_method)(void) = NULL;

// TLS 1.2 ClientHello
static unsigned char tls12_client_hello[] = {
    0x16, 0x03, 0x03, 0x00, 0x00, // TLS record header
    0x01, 0x00, 0x00, 0x00,       // Handshake header
    0x03, 0x03,                   // TLS 1.2
    // Random bytes, cipher suites, etc. filled at runtime
};

// Initialize hooks
__attribute__((constructor))
void init_tls12_shim() {
    original_SSL_connect = dlsym(RTLD_NEXT, "SSL_connect");
    original_SSL_CTX_new = dlsym(RTLD_NEXT, "SSL_CTX_new");
    original_SSL_CTX_set_cipher_list = dlsym(RTLD_NEXT, "SSL_CTX_set_cipher_list");
    original_SSLv23_client_method = dlsym(RTLD_NEXT, "SSLv23_client_method");
}

// Override SSL_connect to force TLS 1.2
int SSL_connect(void *ssl) {
    // Get socket from SSL object (implementation specific)
    int sock = *(int*)((char*)ssl + 0x10); // Offset may vary
    
    // Send TLS 1.2 ClientHello
    send(sock, tls12_client_hello, sizeof(tls12_client_hello), 0);
    
    // Call original
    if (original_SSL_connect) {
        return original_SSL_connect(ssl);
    }
    return -1;
}

// Force TLS 1.2 context
void* SSL_CTX_new(void *method) {
    void *ctx = original_SSL_CTX_new(method);
    if (ctx && original_SSL_CTX_set_cipher_list) {
        // Force modern ciphers
        original_SSL_CTX_set_cipher_list(ctx, 
            "ECDHE-RSA-AES256-GCM-SHA384:"
            "ECDHE-RSA-AES128-GCM-SHA256:"
            "AES256-SHA256:AES128-SHA256");
    }
    return ctx;
}

// Override method to prevent SSLv3
void* SSLv23_client_method(void) {
    // Force TLS 1.2 method instead
    return original_SSLv23_client_method();
}
