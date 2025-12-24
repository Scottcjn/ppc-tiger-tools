#!/usr/bin/env python
"""
RustChain PowerPC G4 Miner - Ancient Python Edition
Compatible with Python 2.2/2.3 on Mac OS X 10.4 Tiger
"""
import os
import sys
import time
import sha
import subprocess
import urllib2

# Configuration
SERVER_URL = "http://50.28.86.131:8088"

def get_hardware_info():
    """Get PowerPC G4 hardware details"""
    info = {}
    try:
        # Get system profiler info
        p = subprocess.Popen(['system_profiler', 'SPHardwareDataType'], stdout=subprocess.PIPE)
        hw_info = p.communicate()[0]

        for line in hw_info.split('\n'):
            line = line.strip()
            if 'Machine Model:' in line:
                info['model'] = line.split(':')[1].strip()
            elif 'CPU Type:' in line:
                info['cpu'] = line.split(':')[1].strip()
            elif 'CPU Speed:' in line:
                info['speed'] = line.split(':')[1].strip()
            elif 'Serial Number:' in line:
                info['serial'] = line.split(':')[1].strip()

        return info
    except Exception, e:
        print "Error getting hardware info:", str(e)
        return {'model': 'PowerBook6,8', 'cpu': 'PowerPC G4', 'speed': '1.5 GHz', 'serial': 'G4-UNKNOWN'}

def collect_entropy():
    """Collect simple timing entropy"""
    samples = []
    for i in range(20):
        start = time.time()
        x = 0
        for j in range(500):
            x = x ^ hash(str(j * i))
        end = time.time()
        samples.append(int((end - start) * 1000000))

    if len(samples) > 0:
        mean = sum(samples) / float(len(samples))
        total = 0
        for s in samples:
            diff = s - mean
            total = total + (diff * diff)
        variance = total / float(len(samples))
    else:
        mean = 0.0
        variance = 0.0

    return mean, variance

def make_system_id(serial):
    """Create system ID using SHA1"""
    return "g4_" + sha.sha(serial).hexdigest()[:12]

def get_challenge():
    """Get attestation challenge"""
    try:
        response = urllib2.urlopen(SERVER_URL + "/attest/challenge")
        data = response.read()

        # Extract nonce manually
        if '"nonce":"' in data:
            nonce_start = data.find('"nonce":"') + 9
            nonce_end = data.find('"', nonce_start)
            return data[nonce_start:nonce_end]
    except Exception, e:
        print "Challenge error:", str(e)
    return ""

def submit_attestation(system_id, mean, variance, hw_info):
    """Submit attestation"""
    nonce = get_challenge()
    if not nonce:
        print "Failed to get challenge"
        return None

    # Create simple commitment
    commitment_data = str(mean) + str(variance) + nonce
    commitment = "b3:" + sha.sha(commitment_data).hexdigest()

    # Build JSON manually (no json module)
    data = '{'
    data = data + '"report":{'
    data = data + '"nonce":"' + nonce + '",'
    data = data + '"device":{'
    data = data + '"family":"PowerPC",'
    data = data + '"arch":"G4",'
    data = data + '"model":"' + hw_info.get('model', 'PowerBook6,8') + '",'
    data = data + '"year":2005'
    data = data + '},'
    data = data + '"derived":{'
    data = data + '"cpu_drift_mean":' + str(mean) + ','
    data = data + '"cpu_drift_var":' + str(variance)
    data = data + '},'
    data = data + '"commitment":"' + commitment + '",'
    data = data + '"timestamp":' + str(int(time.time()))
    data = data + '},'
    data = data + '"miner_pubkey":"' + system_id + '",'
    data = data + '"miner_sig":""'
    data = data + '}'

    try:
        req = urllib2.Request(SERVER_URL + "/attest/submit", data)
        req.add_header('Content-Type', 'application/json')
        response = urllib2.urlopen(req)
        result = response.read()

        # Extract ticket ID
        if '"ticket_id":"' in result:
            id_start = result.find('"ticket_id":"') + 13
            id_end = result.find('"', id_start)
            return result[id_start:id_end]
        else:
            print "No ticket_id in response:", result[:200]
    except Exception, e:
        print "Attestation error:", str(e)
    return None

def enroll_in_epoch(system_id, ticket_id, hw_info):
    """Enroll in current epoch"""
    current_slot = int(time.time() // 600)

    data = '{'
    data = data + '"miner_pubkey":"' + system_id + '",'
    data = data + '"weights":{"temporal":1.0,"rtc":1.0},'
    data = data + '"device":{'
    data = data + '"family":"PowerPC",'
    data = data + '"arch":"G4",'
    data = data + '"model":"' + hw_info.get('model', 'PowerBook6,8') + '",'
    data = data + '"year":2005'
    data = data + '},'
    data = data + '"ticket_id":"' + ticket_id + '",'
    data = data + '"slot":' + str(current_slot)
    data = data + '}'

    try:
        req = urllib2.Request(SERVER_URL + "/epoch/enroll", data)
        req.add_header('Content-Type', 'application/json')
        response = urllib2.urlopen(req)
        result = response.read()
        return '"ok":true' in result
    except Exception, e:
        print "Enrollment error:", str(e)
    return False

def get_balance(system_id):
    """Get balance"""
    try:
        response = urllib2.urlopen(SERVER_URL + "/balance/" + system_id)
        data = response.read()

        if '"balance_rtc":' in data:
            bal_start = data.find('"balance_rtc":') + 14
            bal_end = data.find(',', bal_start)
            if bal_end == -1:
                bal_end = data.find('}', bal_start)
            return float(data[bal_start:bal_end])
    except Exception, e:
        pass
    return 0.0

def main():
    print "RustChain PowerPC G4 Miner"
    print "=========================="
    print "Compatible with Mac OS X 10.4 Tiger"
    print "Server:", SERVER_URL
    print ""

    # Get hardware info
    print "Detecting PowerPC G4..."
    hw_info = get_hardware_info()

    print "Model:", hw_info.get('model', 'PowerBook G4')
    print "CPU:", hw_info.get('cpu', 'PowerPC G4')
    print "Speed:", hw_info.get('speed', '1.5 GHz')
    print "Serial:", hw_info.get('serial', 'Unknown')[:8] + "..."
    print ""

    # Create system ID
    system_id = make_system_id(hw_info.get('serial', 'G4-FALLBACK'))
    print "System ID:", system_id
    print ""

    # Test server connectivity
    print "Testing server connectivity..."
    try:
        test_response = urllib2.urlopen(SERVER_URL + "/api/stats")
        test_data = test_response.read()
        if '"version"' in test_data:
            print "Server reachable: RustChain v2"
        else:
            print "Server response unclear"
    except Exception, e:
        print "Server unreachable:", str(e)
        return 1

    print ""

    # Collect entropy
    print "Collecting entropy..."
    mean, variance = collect_entropy()
    print "CPU timing mean: %.6f microseconds" % mean
    print "CPU timing variance: %.6f" % variance
    print ""

    # Submit attestation
    print "Submitting Silicon Ticket attestation..."
    ticket_id = submit_attestation(system_id, mean, variance, hw_info)

    if not ticket_id:
        print "Attestation failed! Check server logs"
        return 1

    print "SUCCESS: Got Silicon Ticket:", ticket_id
    print ""

    # Enroll in epoch
    print "Enrolling in current epoch..."
    enrolled = enroll_in_epoch(system_id, ticket_id, hw_info)

    if enrolled:
        print "SUCCESS: Enrolled in epoch!"
        print "Hardware multiplier: 2.5x (Classic PowerPC G4)"
        print ""
    else:
        print "Enrollment failed!"
        return 1

    # Check initial balance
    balance = get_balance(system_id)
    print "Current balance: %.8f RTC" % balance
    print ""

    print "PowerPC G4 is now mining RustChain!"
    print "- Epoch-based pro-rata rewards"
    print "- 2.5x Classic hardware advantage"
    print "- Rewards distributed every 24 hours"
    print ""
    print "Monitor balance at:"
    print SERVER_URL + "/balance/" + system_id
    print ""
    print "Press Ctrl+C to stop monitoring..."

    # Balance monitoring loop
    try:
        count = 0
        while True:
            balance = get_balance(system_id)
            count = count + 1
            print "[%d] Balance: %.8f RTC" % (count, balance)
            time.sleep(300)  # 5 minutes
    except KeyboardInterrupt:
        print "\nMining monitor stopped."
        print "Your G4 remains enrolled and earning!"

    return 0

if __name__ == '__main__':
    sys.exit(main())