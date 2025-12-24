#!/usr/bin/env python
"""
RustChain Miner for PowerPC G4 - Python 2.3 Compatible
Compatible with Mac OS X 10.4 Tiger and Python 2.3.5

Usage:
    export RUSTCHAIN_NODE="http://50.28.86.131:8088"
    export MINER_WALLET="98ad7c5973eb4a3173090b9e66011a6b7b8c42cf9RTC"

    python rustchain_miner_ppc23.py
"""

import sys
import os
import time
import hashlib
import urllib2
import json

# Configuration
RUSTCHAIN_NODE = os.getenv('RUSTCHAIN_NODE', 'http://50.28.86.131:8088')
MINER_WALLET = os.getenv('MINER_WALLET', '98ad7c5973eb4a3173090b9e66011a6b7b8c42cf9RTC')

def api_call(endpoint, data=None):
    """Make API call to RustChain node"""
    try:
        url = RUSTCHAIN_NODE + endpoint
        if data:
            req = urllib2.Request(url, json.dumps(data), {'Content-Type': 'application/json'})
        else:
            req = urllib2.Request(url)
        response = urllib2.urlopen(req, None, 30)
        return json.loads(response.read())
    except Exception, e:
        print "[ERROR] API call failed:", str(e)
        return None

def hash_work(block_data, nonce):
    """Hash using SHA256 (Python 2.3 compatible)"""
    data = str(block_data) + ":" + str(nonce)
    return hashlib.sha256(data).hexdigest()

def get_work():
    """Get mining work from node"""
    result = api_call('/api/mine/work')
    if result and 'block_data' in result:
        return result
    return None

def submit_solution(nonce, hash_result):
    """Submit found solution"""
    data = {
        'wallet': MINER_WALLET,
        'nonce': nonce,
        'hash': hash_result
    }
    result = api_call('/api/mine/submit', data)
    return result

def mine():
    """Main mining loop"""
    print "="*80
    print "RustChain Miner for PowerPC G4 (Python 2.3)"
    print "="*80
    print "Node:", RUSTCHAIN_NODE
    print "Wallet:", MINER_WALLET
    print "="*80
    print "[INFO] Starting mining loop..."
    print ""

    total_hashes = 0
    total_shares = 0
    start_time = time.time()

    while True:
        try:
            # Get work
            work = get_work()
            if not work:
                print "[WARN] Could not get work, retrying in 10s..."
                time.sleep(10)
                continue

            block_data = work.get('block_data', '')
            target = work.get('target', 'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff')
            nonce_start = work.get('nonce_start', 0)

            print "[INFO] Got new work - target:", target[:12] + "..."

            # Mine
            nonce = nonce_start
            work_start = time.time()
            hashes_this_round = 0

            while time.time() - work_start < 60:  # 60 second rounds
                hash_result = hash_work(block_data, nonce)
                hashes_this_round = hashes_this_round + 1
                total_hashes = total_hashes + 1

                # Check if solution
                if hash_result < target:
                    print "[SUCCESS] Found solution! Nonce:", nonce, "Hash:", hash_result[:16] + "..."

                    # Submit
                    result = submit_solution(nonce, hash_result)
                    if result and result.get('accepted'):
                        total_shares = total_shares + 1
                        print "[SUCCESS] Share accepted! Total shares:", total_shares
                    else:
                        print "[WARN] Share rejected:", result
                    break

                nonce = nonce + 1

                # Stats every 60 seconds
                if hashes_this_round % 100 == 0:
                    elapsed = time.time() - start_time
                    if elapsed > 0:
                        hash_rate = total_hashes / elapsed
                        print "[STATS] Runtime: %ds | Hashes: %d | Rate: %.2f H/s | Shares: %d" % (
                            int(elapsed), total_hashes, hash_rate, total_shares
                        )

        except KeyboardInterrupt:
            print ""
            print "[INFO] Mining stopped by user"
            elapsed = time.time() - start_time
            if elapsed > 0:
                print "[STATS] Final - Runtime: %ds | Total Hashes: %d | Avg Rate: %.2f H/s | Shares: %d" % (
                    int(elapsed), total_hashes, total_hashes/elapsed, total_shares
                )
            sys.exit(0)
        except Exception, e:
            print "[ERROR] Mining error:", str(e)
            time.sleep(5)

if __name__ == '__main__':
    mine()
