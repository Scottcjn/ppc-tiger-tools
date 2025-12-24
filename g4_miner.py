#!/usr/bin/env python
"""
RustChain PowerPC G4 Miner - Basic Edition
Compatible with very old Python on Mac OS X Tiger
"""
import os
import sys
import time
import hashlib
import subprocess
import urllib2

# Configuration
SERVER_URL = "http://50.28.86.131:8088"
WALLET_ADDRESS = "98ad7c5973eb4a3173090b9e66011a6b7b8c42cf9RTC"

def get_hardware_info():
    """Get PowerPC G4 hardware details"""
    info = {}
    try:
        hw_info = subprocess.Popen(['system_profiler', 'SPHardwareDataType'],
                                  stdout=subprocess.PIPE).communicate()[0]

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
        return {}

def collect_entropy():
    """Collect timing entropy"""
    samples = []
    for i in range(50):
        start = time.time()
        x = 0
        for j in range(1000):
            x = x ^ hash(str(j * i))
        end = time.time()
        samples.append(int((end - start) * 1000000))

    if len(samples) > 0:
        mean = sum(samples) / float(len(samples))
        variance = sum([(x - mean) ** 2 for x in samples]) / float(len(samples))
    else:
        mean = 0.0
        variance = 0.0

    return mean, variance

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
        return None

    # Build simple attestation
    commitment = "b3:" + hashlib.sha256(str(mean) + str(variance)).hexdigest()

    # Simple JSON by hand
    data = '{'
    data += '"report":{'
    data += '"nonce":"' + nonce + '",'
    data += '"device":{"family":"PowerPC","arch":"G4","model":"' + hw_info.get('model', 'PowerBook6,8') + '"},'
    data += '"derived":{"cpu_drift_mean":' + str(mean) + ',"cpu_drift_var":' + str(variance) + '},'
    data += '"commitment":"' + commitment + '",'
    data += '"timestamp":' + str(int(time.time()))
    data += '},'
    data += '"miner_pubkey":"' + system_id + '",'
    data += '"miner_sig":""'
    data += '}'

    try:
        req = urllib2.Request(SERVER_URL + "/attest/submit", data,
                             {'Content-Type': 'application/json'})
        response = urllib2.urlopen(req)
        result = response.read()

        # Extract ticket ID
        if '"ticket_id":"' in result:
            id_start = result.find('"ticket_id":"') + 13
            id_end = result.find('"', id_start)
            ticket_id = result[id_start:id_end]
            return ticket_id
    except Exception, e:
        print "Attestation error:", str(e)
    return None

def enroll_in_epoch(system_id, ticket_id, hw_info):
    """Enroll in current epoch"""
    # Build enrollment data
    data = '{'
    data += '"miner_pubkey":"' + system_id + '",'
    data += '"weights":{"temporal":1.0,"rtc":1.0},'
    data += '"device":{"family":"PowerPC","arch":"G4","model":"' + hw_info.get('model', 'PowerBook6,8') + '"},'
    data += '"ticket_id":"' + ticket_id + '",'
    data += '"slot":' + str(int(time.time() // 600))
    data += '}'

    try:
        req = urllib2.Request(SERVER_URL + "/epoch/enroll", data,
                             {'Content-Type': 'application/json'})
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
    print "==========================="
    print "Server:", SERVER_URL
    print ""

    # Get hardware info
    print "Detecting PowerPC G4..."
    hw_info = get_hardware_info()

    print "Model:", hw_info.get('model', 'PowerBook G4')
    print "CPU:", hw_info.get('cpu', 'PowerPC G4')
    print "Speed:", hw_info.get('speed', '1.5 GHz')
    print ""

    # Create system ID
    system_id = "g4_" + hashlib.sha256(hw_info.get('serial', 'unknown')).hexdigest()[:12]
    print "System ID:", system_id
    print ""

    # Collect entropy
    print "Collecting entropy..."
    mean, variance = collect_entropy()
    print "CPU timing mean: %.6f" % mean
    print "CPU timing variance: %.6f" % variance
    print ""

    # Submit attestation
    print "Submitting attestation..."
    ticket_id = submit_attestation(system_id, mean, variance, hw_info)

    if not ticket_id:
        print "Attestation failed!"
        return 1

    print "Got Silicon Ticket:", ticket_id
    print ""

    # Enroll in epoch
    print "Enrolling in epoch..."
    enrolled = enroll_in_epoch(system_id, ticket_id, hw_info)

    if enrolled:
        print "Successfully enrolled!"
        print "Hardware multiplier: 2.5x (Classic PowerPC G4)"
        print ""
    else:
        print "Enrollment failed!"
        return 1

    # Check balance
    balance = get_balance(system_id)
    print "Current balance: %.6f RTC" % balance
    print ""

    print "PowerPC G4 mining active!"
    print "Epoch-based pro-rata rewards with 2.5x advantage"
    print "Monitor: %s/balance/%s" % (SERVER_URL, system_id)
    print ""
    print "Press Ctrl+C to stop monitoring"

    # Monitor balance
    try:
        while True:
            balance = get_balance(system_id)
            print "Balance: %.6f RTC" % balance
            time.sleep(300)  # 5 minutes
    except KeyboardInterrupt:
        print "\nStopped"

    return 0

if __name__ == '__main__':
    sys.exit(main())