#!/usr/bin/env python
"""
RustChain Wallet for PowerPC G4
Compatible with Python 2.7+ and Python 3.x
Simple wallet operations for checking balance and viewing transactions

Usage:
    export RUSTCHAIN_NODE="http://50.28.86.131:8088"
    export WALLET_ADDRESS="98ad7c5973eb4a3173090b9e66011a6b7b8c42cf9RTC"

    python rustchain_wallet_ppc.py balance
    python rustchain_wallet_ppc.py history
"""

from __future__ import print_function
import sys
import os
import json

# Python 2/3 compatibility
if sys.version_info[0] >= 3:
    import urllib.request as urllib2
    import urllib.error
else:
    import urllib2

# Configuration
RUSTCHAIN_NODE = os.getenv('RUSTCHAIN_NODE', 'http://50.28.86.131:8088')
WALLET_ADDRESS = os.getenv('WALLET_ADDRESS', '98ad7c5973eb4a3173090b9e66011a6b7b8c42cf9RTC')

def api_call(endpoint):
    """Make API call to RustChain node"""
    try:
        url = RUSTCHAIN_NODE + endpoint
        response = urllib2.urlopen(url, timeout=30)
        return json.loads(response.read().decode('utf-8'))
    except Exception, e:
        print("[ERROR] API call failed: {}".format(str(e)))
        return None

def get_balance():
    """Get wallet balance"""
    result = api_call('/api/balance/{}'.format(WALLET_ADDRESS))

    if result:
        print("="*60)
        print("Wallet Balance")
        print("="*60)
        print("Address: {}".format(WALLET_ADDRESS))
        print("Balance: {} RTC".format(result.get('balance', 0.0)))
        print("Pending: {} RTC".format(result.get('pending', 0.0)))
        print("="*60)
    else:
        print("[ERROR] Could not retrieve balance")

def get_history():
    """Get transaction history"""
    result = api_call('/api/transactions/{}'.format(WALLET_ADDRESS))

    if result and 'transactions' in result:
        txs = result['transactions']
        print("="*60)
        print("Transaction History ({} total)".format(len(txs)))
        print("="*60)

        if not txs:
            print("No transactions yet")
        else:
            for i, tx in enumerate(txs[-10:], 1):  # Show last 10
                print("\n{}. {} RTC - {}".format(
                    i,
                    tx.get('amount', 0.0),
                    tx.get('type', 'unknown')
                ))
                print("   Time: {}".format(tx.get('timestamp', 'unknown')))
                if 'block' in tx:
                    print("   Block: {}".format(tx['block']))

        print("\n" + "="*60)
    else:
        print("[ERROR] Could not retrieve transaction history")

def get_stats():
    """Get network stats"""
    result = api_call('/api/stats')

    if result:
        print("="*60)
        print("RustChain Network Stats")
        print("="*60)
        print("Blockchain Height: {}".format(result.get('height', 0)))
        print("Active Miners: {}".format(result.get('active_miners', 0)))
        print("Total Supply: {} RTC".format(result.get('total_supply', 0.0)))
        print("Last Block: {}".format(result.get('last_block_time', 'unknown')))
        print("="*60)
    else:
        print("[ERROR] Could not retrieve network stats")

def show_help():
    """Show help message"""
    print("RustChain Wallet for PowerPC G4")
    print("")
    print("Usage:")
    print("  python rustchain_wallet_ppc.py balance  - Show wallet balance")
    print("  python rustchain_wallet_ppc.py history  - Show transaction history")
    print("  python rustchain_wallet_ppc.py stats    - Show network stats")
    print("")
    print("Environment Variables:")
    print("  RUSTCHAIN_NODE  - Node URL (default: http://50.28.86.131:8088)")
    print("  WALLET_ADDRESS  - Your wallet address")
    print("")
    print("Current Configuration:")
    print("  Node: {}".format(RUSTCHAIN_NODE))
    print("  Wallet: {}".format(WALLET_ADDRESS))

if __name__ == '__main__':
    if len(sys.argv) < 2:
        show_help()
        sys.exit(0)

    command = sys.argv[1].lower()

    if command == 'balance':
        get_balance()
    elif command == 'history':
        get_history()
    elif command == 'stats':
        get_stats()
    elif command == 'help':
        show_help()
    else:
        print("[ERROR] Unknown command: {}".format(command))
        print("Run 'python rustchain_wallet_ppc.py help' for usage")
        sys.exit(1)
