#!/bin/bash
cd ~/altivec_llm

echo "🔥 Building AltiVec Quantum LLM on PowerPC G4..."

# Just use GCC but override the specs to remove version min
gcc -O2 -mcpu=970 -maltivec simple_quantum_test.c -o quantum_test \
    -specs=/dev/null \
    -nostartfiles \
    /usr/lib/crt1.o \
    -lSystem \
    2>&1

if [ -f quantum_test ]; then
    echo "✅ Build succeeded!"
    echo ""
    echo "Testing..."
    ./quantum_test
    echo ""
    echo "🔥 AltiVec Quantum LLM deployed on G4! 🔥"
else
    # Fallback: try simpler approach
    echo "Trying alternate build method..."
    gcc -O2 -mcpu=970 -maltivec simple_quantum_test.c -o quantum_test -Wl,-no_version_load_command 2>&1 || \
    gcc -O2 -mcpu=970 -maltivec simple_quantum_test.c -o quantum_test -mmacosx-version-min=10.4 2>&1 || \
    echo "Build failed - trying static compile..."
    gcc -O2 -mcpu=970 -maltivec simple_quantum_test.c -o quantum_test -static-libgcc 2>&1
    
    if [ -f quantum_test ]; then
        ./quantum_test
    fi
fi
