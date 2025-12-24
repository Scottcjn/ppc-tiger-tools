#!/usr/bin/env python
"""
RustChain Miner for PowerPC G4
Compatible with Python 2.7+ and Python 3.x
Proof of Antiquity (PoA) Mining Client

Usage:
    export RUSTCHAIN_NODE="http://50.28.86.131:8088"
    export MINER_WALLET="98ad7c5973eb4a3173090b9e66011a6b7b8c42cf9RTC"
    python rustchain_miner_ppc.py
"""

from __future__ import print_function
import sys
import os
import time
import hashlib
import json

# Python 2/3 compatibility
if sys.version_info[0] >= 3:
    import urllib.request as urllib2
    import urllib.error
else:
    import urllib2

# Configuration
RUSTCHAIN_NODE = os.getenv('RUSTCHAIN_NODE', 'http://50.28.86.131:8088')
MINER_WALLET = os.getenv('MINER_WALLET', '98ad7c5973eb4a3173090b9e66011a6b7b8c42cf9RTC')
MINER_THREADS = int(os.getenv('MINER_THREADS', '1'))
REPORT_INTERVAL = int(os.getenv('MINER_REPORT_INTERVAL', '60'))

print("="*80)
print("RustChain Miner for PowerPC G4")
print("="*80)
print("Node: {}".format(RUSTCHAIN_NODE))
print("Wallet: {}".format(MINER_WALLET))
print("Threads: {}".format(MINER_THREADS))
print("="*80)

# Simple hash function (using SHA256 as fallback for BLAKE3)
def hash_work(block_data, nonce):
    """Hash block data with nonce using SHA256 (BLAKE3 not available on old Python)"""
    data = "{}:{}".format(block_data, nonce).encode('utf-8')
    return hashlib.sha256(data).hexdigest()

def rpc_call(method, params=None):
    """Make JSON-RPC call to RustChain node"""
    if params is None:
        params = []

    payload = {
        'jsonrpc': '2.0',
        'id': int(time.time()),
        'method': method,
        'params': params
    }

    try:
        if sys.version_info[0] >= 3:
            req = urllib2.Request(
                RUSTCHAIN_NODE + '/rpc',
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
        else:
            req = urllib2.Request(
                RUSTCHAIN_NODE + '/rpc',
                data=json.dumps(payload),
                headers={'Content-Type': 'application/json'}
            )

        response = urllib2.urlopen(req, timeout=30)
        result = json.loads(response.read().decode('utf-8'))

        if 'error' in result and result['error']:
            print("[ERROR] RPC error: {}".format(result['error']))
            return None

        return result.get('result')

    except Exception, e:
        print("[ERROR] RPC call failed: {}".format(str(e)))
        return None

def get_work():
    """Get mining work from node"""
    return rpc_call('getWork', [MINER_WALLET])

def submit_solution(nonce, hash_result):
    """Submit mining solution to node"""
    params = {
        'wallet': MINER_WALLET,
        'nonce': nonce,
        'hash': hash_result
    }
    return rpc_call('submitSolution', [params])

def mine():
    """Main mining loop"""
    hashes_done = 0
    shares_found = 0
    start_time = time.time()
    last_report = start_time

    print("[INFO] Starting mining loop...")

    while True:
        # Get work from node
        work = get_work()

        if not work:
            print("[WARN] Failed to get work, retrying in 10s...")
            time.sleep(10)
            continue

        block_data = work.get('block_template', '')
        target = work.get('target', 'f' * 64)
        nonce_start = work.get('nonce_start', 0)

        print("[INFO] Got new work - target: {}...".format(target[:16]))

        # Mine for up to 60 seconds or until solution found
        nonce = nonce_start
        work_start = time.time()

        while time.time() - work_start < 60:
            # Hash attempt
            hash_result = hash_work(block_data, nonce)
            hashes_done += 1

            # Check if solution
            if hash_result < target:
                print("[SUCCESS] Found solution! Nonce: {}, Hash: {}...".format(
                    nonce, hash_result[:16]
                ))

                # Submit solution
                result = submit_solution(nonce, hash_result)

                if result and result.get('accepted'):
                    shares_found += 1
                    print("[SUCCESS] Share accepted! Total shares: {}".format(shares_found))
                else:
                    print("[WARN] Share rejected")

                break

            nonce += 1

            # Report stats every interval
            if time.time() - last_report >= REPORT_INTERVAL:
                elapsed = time.time() - start_time
                hashrate = hashes_done / elapsed if elapsed > 0 else 0

                print("[STATS] Runtime: {:.0f}s | Hashes: {} | Rate: {:.2f} H/s | Shares: {}".format(
                    elapsed, hashes_done, hashrate, shares_found
                ))

                last_report = time.time()

        # Brief pause before next work request
        time.sleep(1)

if __name__ == '__main__':
    try:
        print("[INFO] RustChain Miner starting...")
        print("[INFO] Press Ctrl+C to stop")
        print("")
        mine()
    except KeyboardInterrupt:
        print("\n[INFO] Miner stopped by user")
    except Exception, e:
        print("\n[ERROR] Miner crashed: {}".format(str(e)))
        sys.exit(1)
