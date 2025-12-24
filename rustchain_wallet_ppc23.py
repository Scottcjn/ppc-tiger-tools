#!/usr/bin/env python
"""
RustChain Wallet for PowerPC G4 - Python 2.3 Compatible
Simple wallet operations for checking balance and viewing transactions

Usage:
    export RUSTCHAIN_NODE="http://50.28.86.131:8088"
    export WALLET_ADDRESS="98ad7c5973eb4a3173090b9e66011a6b7b8c42cf9RTC"

    python rustchain_wallet_ppc23.py balance
    python rustchain_wallet_ppc23.py stats
"""

import sys
import os
import urllib2
import json

# Configuration
RUSTCHAIN_NODE = os.getenv('RUSTCHAIN_NODE', 'http://50.28.86.131:8088')
WALLET_ADDRESS = os.getenv('WALLET_ADDRESS', '98ad7c5973eb4a3173090b9e66011a6b7b8c42cf9RTC')

def api_call(endpoint):
    """Make API call to RustChain node"""
    try:
        url = RUSTCHAIN_NODE + endpoint
        response = urllib2.urlopen(url, None, 30)
        return json.loads(response.read())
    except Exception, e:
        print "[ERROR] API call failed:", str(e)
        return None

def get_balance():
    """Get wallet balance"""
    result = api_call('/api/balance/' + WALLET_ADDRESS)

    if result:
        print "="*60
        print "Wallet Balance"
        print "="*60
        print "Address:", WALLET_ADDRESS
        print "Balance:", result.get('balance', 0.0), "RTC"
        print "Pending:", result.get('pending', 0.0), "RTC"
        print "="*60
    else:
        print "[ERROR] Could not retrieve balance"

def get_stats():
    """Get network stats"""
    result = api_call('/api/stats')

    if result:
        print "="*60
        print "RustChain Network Stats"
        print "="*60
        print "Version:", result.get('version', 'unknown')
        print "Epoch:", result.get('epoch', 0)
        print "Active Miners:", result.get('total_miners', 0)
        print "Block Time:", result.get('block_time', 0), "seconds"
        print "Features:", ', '.join(result.get('features', []))
        print "="*60
    else:
        print "[ERROR] Could not retrieve network stats"

def show_help():
    """Show help message"""
    print "RustChain Wallet for PowerPC G4 (Python 2.3)"
    print ""
    print "Usage:"
    print "  python rustchain_wallet_ppc23.py balance  - Show wallet balance"
    print "  python rustchain_wallet_ppc23.py stats    - Show network stats"
    print ""
    print "Current Configuration:"
    print "  Node:", RUSTCHAIN_NODE
    print "  Wallet:", WALLET_ADDRESS

if __name__ == '__main__':
    if len(sys.argv) < 2:
        show_help()
        sys.exit(0)

    command = sys.argv[1].lower()

    if command == 'balance':
        get_balance()
    elif command == 'stats':
        get_stats()
    elif command == 'help':
        show_help()
    else:
        print "[ERROR] Unknown command:", command
        print "Run 'python rustchain_wallet_ppc23.py help' for usage"
        sys.exit(1)
