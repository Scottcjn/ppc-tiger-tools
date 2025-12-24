#!/usr/bin/env python
"""
RustChain PowerPC G4 Miner
For Mac OS X PowerPC systems
"""
import os
import sys
import json
import time
import hashlib
import subprocess
import urllib2

# Configuration
SERVER_URL = "http://50.28.86.131:8088"
WALLET_ADDRESS = "98ad7c5973eb4a3173090b9e66011a6b7b8c42cf9RTC"

def get_hardware_info():
    """Get PowerPC G4 hardware details"""
    try:
        # Get system profiler info
        hw_info = subprocess.check_output(['system_profiler', 'SPHardwareDataType'])

        # Parse key values
        info = {}
        for line in hw_info.split('\n'):
            if 'Machine Model:' in line:
                info['model'] = line.split(':')[1].strip()
            elif 'CPU Type:' in line:
                info['cpu'] = line.split(':')[1].strip()
            elif 'CPU Speed:' in line:
                info['speed'] = line.split(':')[1].strip()
            elif 'Serial Number:' in line:
                info['serial'] = line.split(':')[1].strip()

        # Get MAC address
        en0_info = subprocess.check_output(['ifconfig', 'en0'])
        for line in en0_info.split('\n'):
            if 'ether' in line:
                info['mac'] = line.split()[1]

        return info
    except Exception as e:
        print("Error getting hardware info: %s" % e)
        return {}

def collect_entropy():
    """Collect timing entropy from PowerPC G4"""
    samples = []

    # Use mach_absolute_time for high-res timing
    for i in range(100):
        start = time.time()

        # CPU-intensive operations for timing variance
        x = 0
        for j in range(10000):
            x ^= hash(str(j * i))

        end = time.time()
        samples.append(int((end - start) * 1000000))  # microseconds

    # Calculate statistics
    mean = sum(samples) / float(len(samples))
    variance = sum((x - mean) ** 2 for x in samples) / float(len(samples))

    return {
        'cpu_drift_mean': mean,
        'cpu_drift_var': variance,
        'samples': samples[:10]  # Send subset
    }

def create_fingerprint(hw_info):
    """Create hardware fingerprint"""
    fp_data = {
        'platform': {
            'system': 'Darwin',
            'machine': 'Power Macintosh',
            'processor': hw_info.get('cpu', 'PowerPC G4'),
            'node': hw_info.get('model', 'PowerBook6,8')
        },
        'cpu': {
            'speed': hw_info.get('speed', '1.5 GHz'),
            'cores': 1,
            'altivec': True
        },
        'identifiers': {
            'serial': hw_info.get('serial', ''),
            'mac': hw_info.get('mac', '')
        }
    }

    # Create deterministic ID
    fp_string = json.dumps(fp_data, sort_keys=True)
    system_id = "g4_%s" % hashlib.sha256(fp_string).hexdigest()[:12]

    return fp_data, system_id

def register_node(fingerprint, system_id):
    """Register with RustChain network"""
    data = json.dumps({
        'system_id': system_id,
        'fingerprint': fingerprint,
        'wallet': WALLET_ADDRESS
    })

    try:
        req = urllib2.Request(
            "%s/api/register" % SERVER_URL,
            data,
            {'Content-Type': 'application/json'}
        )
        response = urllib2.urlopen(req)
        result = json.loads(response.read())
        return result.get('success', False)
    except Exception as e:
        print("Registration error: %s" % e)
        return False

def submit_attestation(system_id, entropy_data, hw_info):
    """Submit Silicon Ticket attestation"""

    # Get challenge nonce
    try:
        challenge = json.loads(urllib2.urlopen("%s/attest/challenge" % SERVER_URL).read())
        nonce = challenge['nonce']
    except Exception as e:
        print("Challenge error: %s" % e)
        return None

    # Build attestation report
    report = {
        'nonce': nonce,
        'device': {
            'family': 'PowerPC',
            'model': hw_info.get('model', 'PowerBook6,8'),
            'arch': 'G4',
            'year': 2005,
            'ram_bytes': 1342177280  # 1.25 GB
        },
        'derived': entropy_data,
        'commitment': "b3:%s" % hashlib.sha256(json.dumps(entropy_data)).hexdigest(),
        'timestamp': int(time.time())
    }

    data = json.dumps({
        'report': report,
        'miner_pubkey': '',  # Would add Ed25519 key here
        'miner_sig': ''
    })

    try:
        req = urllib2.Request(
            "%s/attest/submit" % SERVER_URL,
            data,
            {'Content-Type': 'application/json'}
        )
        response = urllib2.urlopen(req)
        ticket = json.loads(response.read())
        return ticket
    except Exception as e:
        print("Attestation error: %s" % e)
        return None

def mine_block(system_id, ticket):
    """Join mining pool and participate"""

    while True:
        # Join mining pool
        data = json.dumps({'system_id': system_id})

        try:
            req = urllib2.Request(
                "%s/api/mine" % SERVER_URL,
                data,
                {'Content-Type': 'application/json'}
            )
            response = urllib2.urlopen(req)
            result = json.loads(response.read())

            print("Mining block %d - Miners in pool: %d, Your shares: %d" % (
                result.get('block_height', 0),
                result.get('miners_in_pool', 0),
                result.get('your_shares', 0)
            ))

        except Exception as e:
            print("Mining error: %s" % e)

        # Wait before next submission (don't spam)
        time.sleep(60)

def main():
    print("RustChain PowerPC G4 Miner")
    print("==========================")
    print("Server: %s" % SERVER_URL)
    print("Wallet: %s" % WALLET_ADDRESS)
    print("")

    # Get hardware info
    print("Detecting PowerPC G4 hardware...")
    hw_info = get_hardware_info()

    if not hw_info:
        print("Failed to get hardware info!")
        return 1

    print("Model: %s" % hw_info.get('model', 'Unknown'))
    print("CPU: %s @ %s" % (hw_info.get('cpu', 'PowerPC G4'), hw_info.get('speed', 'Unknown')))
    print("MAC: %s" % hw_info.get('mac', 'Unknown'))
    print("")

    # Create fingerprint and register
    print("Creating hardware fingerprint...")
    fingerprint, system_id = create_fingerprint(hw_info)
    print("System ID: %s" % system_id)

    print("Registering with RustChain network...")
    if not register_node(fingerprint, system_id):
        print("Registration failed!")
        return 1

    print("Registration successful!")
    print("")

    # Collect entropy and get Silicon Ticket
    print("Collecting entropy...")
    entropy = collect_entropy()
    print("CPU drift mean: %.6f" % entropy['cpu_drift_mean'])
    print("CPU drift variance: %.6f" % entropy['cpu_drift_var'])

    print("Submitting attestation...")
    ticket = submit_attestation(system_id, entropy, hw_info)

    if not ticket:
        print("Attestation failed!")
        return 1

    print("Got Silicon Ticket: %s" % ticket.get('ticket_id', 'Unknown'))
    print("Hardware multiplier: %.1fx (Classic tier)" % ticket.get('weight', 2.5))
    print("")

    # Save ticket for later use
    with open('ticket.json', 'w') as f:
        json.dump(ticket, f)

    # Start mining
    print("Starting mining loop...")
    print("Block time: 600 seconds (10 minutes)")
    print("Expected reward: %.2f RTC per block (with 2.5x multiplier)" % (1.5 * 2.5))
    print("")

    try:
        mine_block(system_id, ticket)
    except KeyboardInterrupt:
        print("\nMining stopped")

    return 0

if __name__ == '__main__':
    sys.exit(main())