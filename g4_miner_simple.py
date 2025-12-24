#!/usr/bin/env python
# Ultra-simple RustChain miner for PowerPC G4 Python 2.4

import urllib
import time
import random

NODE = "http://50.28.86.131:8085"
WALLET = "98ad7c5973eb4a3173090b9e66011a6b7b8c42cf9RTC"

print "RustChain PowerPC G4 Miner Starting..."
print "Wallet:", WALLET
print "Node:", NODE
print ""

while True:
    try:
        nonce = random.randint(0, 999999999)
        data = "miner_address=%s&hardware_type=PowerPC_G4&hardware_age=22&nonce=%d" % (WALLET, nonce)
        
        print "Mining with nonce:", nonce
        req = urllib.urlopen(NODE + "/api/mining/submit", data)
        result = req.read()
        print "Result:", result
        print ""
        
    except Exception, e:
        print "Error:", e
    
    time.sleep(30)