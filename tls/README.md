# Modern TLS for Mac OS X Tiger

**TLS 1.3 on PowerPC G4/G5** - Because they said it couldn't be done.

## What This Does

Installs modern TLS libraries on Tiger (10.4) so your vintage Mac can:
- Access HTTPS websites
- Use modern APIs (GitHub, etc.)
- Download from secure servers
- Stop getting "SSL handshake failed" errors

## Requirements

- Mac OS X 10.4 Tiger (PowerPC)
- MacPorts installed
- Internet connection (ironic, I know)

## Installation

```bash
curl -O https://raw.githubusercontent.com/Scottcjn/ppc-tiger-tools/main/tls/install_modern_tls.sh
chmod +x install_modern_tls.sh
sudo ./install_modern_tls.sh
```

## What Gets Installed

| Package | Version | Location |
|---------|---------|----------|
| OpenSSL | 3.x or 1.1.1 | /opt/local/bin/openssl |
| curl | 7.88+ | /opt/local/bin/curl |
| wget | 1.21+ | /opt/local/bin/wget |

## After Installation

Add to your `.profile`:
```bash
export PATH="/opt/local/bin:$PATH"
```

Or use the full paths:
```bash
/opt/local/bin/curl https://api.github.com/zen
/opt/local/bin/wget https://example.com/file.tar.gz
```

## Verification

```bash
# Check TLS version
/opt/local/bin/openssl s_client -connect api.github.com:443 -brief

# Should show:
# CONNECTION ESTABLISHED
# Protocol version: TLSv1.3
# Ciphersuite: TLS_AES_128_GCM_SHA256
```

## Why This Matters

Tiger shipped with OpenSSL 0.9.7 (2006). That's:
- No TLS 1.1
- No TLS 1.2
- No TLS 1.3
- No modern ciphersuites
- No SNI support

Most HTTPS servers now require TLS 1.2 minimum. Without this patch,
your G4 is cut off from the modern internet.

## Built By

Elyan Labs - Christmas 2025

*"The PowerPC G4 was classified as a weapon in 1999. Now it runs TLS 1.3."*

## Glowie Counter

539 clones and counting. We see you. 👁️
