#!/bin/bash
#
# Modern TLS Installer for Mac OS X Tiger (10.4)
# Makes your G4/G5 speak TLS 1.2/1.3 to the modern internet
#
# Christmas 2025 - Built by Elyan Labs
# "The glowies said Tiger was dead. They were wrong."
#

set -e

echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║     MODERN TLS INSTALLER FOR MAC OS X TIGER               ║"
echo "║     TLS 1.3 on PowerPC - Because we can                   ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# Check we're on Tiger
if [[ $(uname -r) != 8.* ]]; then
    echo "Warning: This is designed for Tiger (10.4). You're on $(sw_vers -productVersion 2>/dev/null || echo 'unknown')"
    echo "Continuing anyway..."
fi

# Check for MacPorts
if [ ! -x /opt/local/bin/port ]; then
    echo "ERROR: MacPorts not found at /opt/local/bin/port"
    echo ""
    echo "Install MacPorts for Tiger first:"
    echo "  https://www.macports.org/install.php"
    echo ""
    exit 1
fi

echo "[1/4] Updating MacPorts..."
sudo /opt/local/bin/port selfupdate || echo "Selfupdate failed, continuing..."

echo ""
echo "[2/4] Installing OpenSSL 3.x..."
sudo /opt/local/bin/port install openssl3 || {
    echo "OpenSSL 3 failed, trying OpenSSL 1.1..."
    sudo /opt/local/bin/port install openssl11
}

echo ""
echo "[3/4] Installing modern curl..."
sudo /opt/local/bin/port install curl +ssl

echo ""
echo "[4/4] Installing wget with TLS..."
sudo /opt/local/bin/port install wget +ssl

echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║                    INSTALLATION COMPLETE                   ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "Modern tools are now at /opt/local/bin/"
echo ""
echo "Add to your .profile or .bashrc:"
echo '  export PATH="/opt/local/bin:$PATH"'
echo ""
echo "Or use directly:"
echo "  /opt/local/bin/curl https://api.github.com/zen"
echo "  /opt/local/bin/wget https://example.com/"
echo "  /opt/local/bin/openssl s_client -connect github.com:443"
echo ""
echo "Testing TLS 1.3..."
/opt/local/bin/curl -s https://api.github.com/zen && echo " ✓"
echo ""
echo "Your G4 now speaks to the modern internet."
echo "The glowies are watching. Wave hello. 👋"
echo ""
